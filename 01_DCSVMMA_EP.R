# ==============================================================================
# 01_DCSVMMA_EP_独立计算.R
#
# 只计算 Epanechnikov 核的 DCSVM 家族：
# DCSVMICL、DCSVMICH、SDCL、SDCH、DCSVM-UNIF 和 DCSVMMA。
#
# 计时原则：
# - 顺序运行，不使用 foreach/doParallel。
# - DCSVMMA 时间 = 完整训练集候选模型 + 5 折 OOF + 权重求解。
# - 数据生成、测试集评价和文件写入不计入方法时间。
#
# DCSVM 特征筛选使用标准整条 lambda 路径：
# 一次 dcsvm(..., nlambda = num_steps) 完成路径拟合。
# ==============================================================================

required_packages <- c("MASS", "caret", "Rsolnp", "dcsvm")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# ==============================================================================
# 0. 统一参数：三个计算脚本必须保持一致
# ==============================================================================

C_SCALE_VALUES <- c(0.2)
TARGET_DGP <- c("DGP1","DGP2")
TARGET_MISSPEC <- c(FALSE, TRUE)

P_SIM <- 1000L
N_TRAIN_SEQ <- c(100L, 200L, 300L, 400L)
N_TEST <- 1000L
N_REPEATS <- 100L
JN <- 5L

IMBALANCE_RATIO <- 0.5
DATA_SEED_BASE <- 1000L

K_FIXED_DCSVM_BY_DGP <- c(
  DGP1 = 25L,
  DGP2 = 20L
)

KERNEL_TYPE <- "epanechnikov"
KERNEL_LABEL <- "Epanechnikov"

RESULT_ROOT <- file.path(getwd(), "Split_Results")
OUTPUT_DIR <- file.path(RESULT_ROOT, "DCSVMMA_EP")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

OUTPUT_FILE <- file.path(
  OUTPUT_DIR,
  "DCSVMMA_EP_replications.csv"
)
ERROR_FILE <- file.path(
  OUTPUT_DIR,
  "DCSVMMA_EP_errors.csv"
)

# ==============================================================================
# 1. 数据生成：与原模拟文件保持一致
# ==============================================================================

generate_data_fun <- function(
    p,
    total_n,
    ismisspecification = FALSE,
    pos_ratio = 0.5
) {
  q_true <- 5L
  p_full <- as.integer(p) + 1L
  mu_val <- 0.8
  rho <- 0.5

  if (p < q_true) {
    stop("p 必须不小于真实信号变量数 q_true = 5。")
  }
  if (pos_ratio <= 0 || pos_ratio >= 1) {
    stop("pos_ratio 必须在 0 和 1 之间。")
  }

  mu <- rep(0, p_full)
  mu[seq_len(q_true)] <- mu_val

  Sigma <- matrix(rho, nrow = p_full, ncol = p_full)
  diag(Sigma) <- 1

  Y <- stats::rbinom(total_n, size = 1, prob = pos_ratio)
  Y[Y == 0] <- -1

  X <- MASS::mvrnorm(
    n = total_n,
    mu = rep(0, p_full),
    Sigma = Sigma
  )

  X[Y == 1, ] <- sweep(
    X[Y == 1, , drop = FALSE],
    2,
    mu,
    "+"
  )
  X[Y == -1, ] <- sweep(
    X[Y == -1, , drop = FALSE],
    2,
    mu,
    "-"
  )

  observed_indices <- if (ismisspecification) {
    c(
      seq_len(q_true - 1L),
      seq.int(q_true + 1L, p_full)
    )
  } else {
    seq_len(p)
  }

  list(
    X = X[, observed_indices, drop = FALSE],
    Y = Y
  )
}

generate_data_fun2 <- function(
    p,
    total_n,
    ismisspecification = TRUE,
    pos_ratio = 0.5
) {
  q <- 5L
  beta_value <- 0.5
  rho <- 0.3

  if (pos_ratio <= 0 || pos_ratio >= 1) {
    stop("pos_ratio 必须在 0 和 1 之间。")
  }

  if (ismisspecification) {
    q <- q + 1L
    p <- p + 1L
  }

  beta <- rep(0, p)
  beta[seq_len(q)] <- beta_value

  Sigma <- rho ^ abs(outer(seq_len(p), seq_len(p), "-"))
  X <- MASS::mvrnorm(
    n = total_n,
    mu = rep(0, p),
    Sigma = Sigma
  )

  X_beta <- as.numeric(X %*% beta)

  if (abs(pos_ratio - 0.5) < 1e-12) {
    b0 <- 0
  } else {
    b0 <- stats::uniroot(
      function(intercept) {
        mean(stats::plogis(intercept + X_beta)) - pos_ratio
      },
      lower = -50,
      upper = 50
    )$root
  }

  prob <- stats::plogis(b0 + X_beta)
  Y <- stats::rbinom(total_n, size = 1, prob = prob)
  Y[Y == 0] <- -1

  inds <- if (ismisspecification) {
    c(
      seq_len(q - 1L),
      seq.int(q + 1L, p)
    )
  } else {
    seq_len(p)
  }

  list(
    X = X[, inds, drop = FALSE],
    Y = Y
  )
}

generate_simulation_data <- function(
    dgp_type,
    p,
    total_n,
    is_misspec,
    pos_ratio = 0.5
) {
  if (identical(dgp_type, "DGP1")) {
    return(generate_data_fun(
      p = p,
      total_n = total_n,
      ismisspecification = is_misspec,
      pos_ratio = pos_ratio
    ))
  }

  if (identical(dgp_type, "DGP2")) {
    return(generate_data_fun2(
      p = p,
      total_n = total_n,
      ismisspecification = is_misspec,
      pos_ratio = pos_ratio
    ))
  }

  stop("不支持的数据生成机制：", dgp_type)
}

# ==============================================================================
# 2. 损失、计时和通用函数
# ==============================================================================

get_dynamic_h <- function(n, C = 1.0) {
  C * n^(-1 / 5)
}

get_dynamic_lam2 <- function(n) {
  1 / (2 * n)
}

smoothed_hinge_values <- function(v, h, kernel) {
  v <- as.numeric(v)
  if (length(h) == 1L) {
    h <- rep(as.numeric(h), length(v))
  } else {
    h <- as.numeric(h)
  }

  if (length(h) != length(v) || any(!is.finite(h)) || any(h <= 0)) {
    stop("h 必须是正的标量或与 v 等长的正向量。")
  }

  if (identical(kernel, "gaussian")) {
    z <- (1 - v) / h
    return((1 - v) * stats::pnorm(z) + h * stats::dnorm(z))
  }

  if (!identical(kernel, "epanechnikov")) {
    stop("不支持的损失核：", kernel)
  }

  result <- numeric(length(v))
  case1 <- v <= 1 - h
  result[case1] <- 1 - v[case1]

  case2 <- (v > 1 - h) & (v <= 1 + h)
  if (any(case2)) {
    v2 <- v[case2]
    h2 <- h[case2]
    result[case2] <-
      ((1 - v2 + h2)^3 *
         (3 * h2 - (1 - v2))) /
      (16 * h2^3)
  }

  result
}

dcsvm_loss_sum <- function(v, h, kernel) {
  sum(smoothed_hinge_values(v, h, kernel))
}

dcsvm_loss_mean <- function(v, h, kernel) {
  mean(smoothed_hinge_values(v, h, kernel))
}

start_method_timer <- function() {
  proc.time()
}

stop_method_timer <- function(start_time) {
  timing_diff <- proc.time() - start_time
  c(
    elapsed = unname(as.numeric(timing_diff["elapsed"])),
    cpu = unname(as.numeric(
      timing_diff["user.self"] + timing_diff["sys.self"]
    ))
  )
}

make_data_signature <- function(X, Y) {
  paste(
    format(sum(X), digits = 17, scientific = TRUE),
    format(sum(abs(X)), digits = 17, scientific = TRUE),
    format(sum(X^2), digits = 17, scientific = TRUE),
    format(sum(Y), digits = 17, scientific = TRUE),
    sep = "|"
  )
}

get_error_rate <- function(score, truth) {
  pred_class <- ifelse(as.numeric(score) >= 0, 1, -1)
  mean(pred_class != truth)
}

solve_simplex_weights <- function(Pred_mat, Y, loss_type, h_vec = NULL, kernel = NULL) {
  model_num <- ncol(Pred_mat)

  if (model_num == 1L) {
    result <- 1
    attr(result, "solver") <- "single_model"
    attr(result, "fallback") <- FALSE
    return(result)
  }

  w0 <- rep(1 / model_num, model_num)

  if (identical(loss_type, "hinge")) {
    obj_fn <- function(w) {
      mean(pmax(0, 1 - Y * as.numeric(Pred_mat %*% w)))
    }
  } else {
    obj_fn <- function(w) {
      dcsvm_loss_mean(
        Y * as.numeric(Pred_mat %*% w),
        h_vec,
        kernel
      )
    }
  }

  individual_losses <- vapply(
    seq_len(model_num),
    function(model_index) {
      one_hot <- rep(0, model_num)
      one_hot[model_index] <- 1
      obj_fn(one_hot)
    },
    numeric(1)
  )
  best_single <- rep(0, model_num)
  best_single[which.min(individual_losses)] <- 1

  solver_weights <- tryCatch({
    fit <- suppressWarnings(
      Rsolnp::solnp(
        pars = w0,
        fun = obj_fn,
        eqfun = function(w) sum(w) - 1,
        eqB = 0,
        LB = rep(0, model_num),
        UB = rep(1, model_num),
        control = list(trace = 0)
      )
    )

    candidate <- as.numeric(fit$pars)
    if (length(candidate) != model_num || any(!is.finite(candidate))) {
      stop("权重求解器返回无效结果。")
    }
    candidate <- pmin(1, pmax(0, candidate))
    if (sum(candidate) <= 0) {
      stop("权重和为 0。")
    }
    candidate / sum(candidate)
  }, error = function(e) {
    NULL
  })

  candidates <- list(
    uniform = w0,
    best_single = best_single
  )
  if (!is.null(solver_weights)) {
    candidates$solnp <- solver_weights
  }

  candidate_losses <- vapply(candidates, obj_fn, numeric(1))
  selected_name <- names(which.min(candidate_losses))[1]
  result <- candidates[[selected_name]]
  result[abs(result) < 1e-5] <- 0
  result <- result / sum(result)

  attr(result, "solver") <- selected_name
  attr(result, "fallback") <- !identical(selected_name, "solnp")
  result
}

# ==============================================================================
# 3. DCSVM 标准整条 lambda 路径筛选
# ==============================================================================

Indexs_fun_dcsvm <- function(
    X,
    Y,
    k,
    h_val,
    kernel,
    screen_seed = NULL
) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  p <- ncol(X)
  k <- min(as.integer(k), p)

  if (!is.null(screen_seed)) {
    set.seed(as.integer(screen_seed))
  }

  num_steps <- max(50L, floor(p / 5))

  fit_path <- suppressWarnings(
    dcsvm::dcsvm(
      x = X,
      y = Y,
      nlambda = num_steps,
      lam2 = 0,
      kern = kernel,
      hval = h_val
    )
  )

  beta_path <- as.matrix(fit_path$beta)
  lambda_path <- as.numeric(fit_path$lambda)

  if (
    nrow(beta_path) != p ||
      ncol(beta_path) == 0L ||
      length(lambda_path) != ncol(beta_path)
  ) {
    stop("DCSVM 系数路径无效。")
  }

  path_order <- order(lambda_path, decreasing = FALSE)
  first_zero_step <- rep(NA_integer_, p)
  max_abs_coef <- numeric(p)

  for (step_index in seq_along(path_order)) {
    beta <- as.numeric(beta_path[, path_order[step_index]])
    beta[!is.finite(beta)] <- 0

    max_abs_coef <- pmax(max_abs_coef, abs(beta))
    newly_zero <- which(
      abs(beta) <= 1e-8 &
        is.na(first_zero_step)
    )
    first_zero_step[newly_zero] <- step_index
  }

  stability_step <- ifelse(
    is.na(first_zero_step),
    length(path_order) + 1L,
    first_zero_step
  )

  features_ind <- order(
    -stability_step,
    -max_abs_coef,
    seq_len(p)
  )[seq_len(k)]

  Indexs_mat <- matrix(FALSE, nrow = k, ncol = p)
  for (i in seq_len(k)) {
    Indexs_mat[i, features_ind[seq_len(i)]] <- TRUE
  }

  Indexs_mat
}

# ==============================================================================
# 4. 单次 DCSVM 家族计算
# ==============================================================================

run_dcsvm_family_once <- function(
    X_train,
    Y_train,
    X_test,
    Y_test,
    samplesize,
    k_fixed,
    C_scale,
    fold_seed,
    kernel,
    kernel_label,
    Jn = 5L
) {
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  Y_train <- as.numeric(as.character(Y_train))
  Y_test <- as.numeric(as.character(Y_test))
  p <- ncol(X_train)

  h_train <- get_dynamic_h(samplesize, C_scale)
  lam2_train <- get_dynamic_lam2(samplesize)

  # A. 完整训练集的特征筛选和候选模型拟合。
  t0_base <- start_method_timer()

  Indexs_dc <- Indexs_fun_dcsvm(
    X = X_train,
    Y = Y_train,
    k = k_fixed,
    h_val = h_train,
    kernel = kernel,
    screen_seed = fold_seed + 100L
  )

  Betas_dc <- matrix(0, nrow = p + 1L, ncol = nrow(Indexs_dc))

  for (i in seq_len(nrow(Indexs_dc))) {
    vars <- which(Indexs_dc[i, ])
    fit_full <- dcsvm::dcsvm(
      x = X_train[, vars, drop = FALSE],
      y = Y_train,
      lambda = 0,
      lam2 = lam2_train,
      hval = h_train,
      kern = kernel
    )

    Betas_dc[1, i] <- as.numeric(fit_full$b0)[1]
    Betas_dc[vars + 1L, i] <- as.numeric(fit_full$beta)
  }

  timing_base <- stop_method_timer(t0_base)

  # DCSVMICL/DCSVMICH/SDCL/SDCH 所需的训练损失。
  # 这些计算不属于 DCSVMMA 时间。
  train_pred_matrix <- cbind(1, X_train) %*% Betas_dc
  loss_train <- vapply(
    seq_len(ncol(train_pred_matrix)),
    function(i) {
      dcsvm_loss_sum(
        Y_train * train_pred_matrix[, i],
        h_train,
        kernel
      )
    },
    numeric(1)
  )

  idx_sums <- rowSums(Indexs_dc)

  score_dcl <- loss_train + idx_sums * log(samplesize)
  w_dcl <- rep(0, nrow(Indexs_dc))
  w_dcl[which.min(score_dcl)] <- 1

  score_dch <- loss_train + idx_sums * (log(samplesize)^(3 / 2))
  w_dch <- rep(0, nrow(Indexs_dc))
  w_dch[which.min(score_dch)] <- 1

  sdcl_scaled <- -score_dcl / 2
  w_sdcl <- exp(sdcl_scaled - max(sdcl_scaled))
  w_sdcl <- w_sdcl / sum(w_sdcl)

  sdch_scaled <- -score_dch / 2
  w_sdch <- exp(sdch_scaled - max(sdch_scaled))
  w_sdch <- w_sdch / sum(w_sdch)

  w_unif <- rep(1 / nrow(Indexs_dc), nrow(Indexs_dc))

  # B. 固定且分层的 5 折 OOF，每个训练折重新筛选特征。
  set.seed(as.integer(fold_seed))
  folds <- caret::createFolds(
    as.factor(Y_train),
    k = Jn,
    list = FALSE
  )

  Pred_OOF <- matrix(
    0,
    nrow = samplesize,
    ncol = nrow(Indexs_dc)
  )
  h_oof <- numeric(samplesize)

  t0_oof <- start_method_timer()

  for (j in seq_len(Jn)) {
    tr_idx <- folds != j
    val_idx <- folds == j
    fold_n <- sum(tr_idx)

    h_fold <- get_dynamic_h(fold_n, C_scale)
    lam2_fold <- get_dynamic_lam2(fold_n)
    h_oof[val_idx] <- h_fold

    Indexs_fold <- Indexs_fun_dcsvm(
      X = X_train[tr_idx, , drop = FALSE],
      Y = Y_train[tr_idx],
      k = ncol(Pred_OOF),
      h_val = h_fold,
      kernel = kernel,
      screen_seed = fold_seed + 1000L + j
    )

    if (nrow(Indexs_fold) != ncol(Pred_OOF)) {
      stop("DCSVM OOF 候选模型数不一致。")
    }

    for (i in seq_len(ncol(Pred_OOF))) {
      vars <- which(Indexs_fold[i, ])
      fit_fold <- dcsvm::dcsvm(
        x = X_train[tr_idx, vars, drop = FALSE],
        y = Y_train[tr_idx],
        lambda = 0,
        lam2 = lam2_fold,
        hval = h_fold,
        kern = kernel
      )

      Pred_OOF[val_idx, i] <-
        as.numeric(
          X_train[val_idx, vars, drop = FALSE] %*%
            as.numeric(fit_fold$beta) +
            as.numeric(fit_fold$b0)[1]
        )
    }
  }

  timing_oof <- stop_method_timer(t0_oof)

  # C. DCSVMMA 权重求解。
  set.seed(as.integer(fold_seed + 500000L))
  t0_solver <- start_method_timer()
  w_ma <- solve_simplex_weights(
    Pred_mat = Pred_OOF,
    Y = Y_train,
    loss_type = "smoothed",
    h_vec = h_oof,
    kernel = kernel
  )
  timing_solver <- stop_method_timer(t0_solver)

  elapsed_dcsvmma <-
    timing_base[["elapsed"]] +
    timing_oof[["elapsed"]] +
    timing_solver[["elapsed"]]
  cpu_dcsvmma <-
    timing_base[["cpu"]] +
    timing_oof[["cpu"]] +
    timing_solver[["cpu"]]

  # D. 测试集评价（不计时）。
  Pred_Test <- cbind(1, X_test) %*% Betas_dc
  w_test_opt <- solve_simplex_weights(
    Pred_mat = Pred_Test,
    Y = Y_test,
    loss_type = "smoothed",
    h_vec = get_dynamic_h(samplesize, C_scale),
    kernel = kernel
  )
  min_loss <- max(
    dcsvm_loss_mean(
      Y_test * as.numeric(Pred_Test %*% w_test_opt),
      get_dynamic_h(samplesize, C_scale),
      kernel
    ),
    0.01
  )

  weight_list <- list(
    dcl = w_dcl,
    dch = w_dch,
    sdcl = w_sdcl,
    sdch = w_sdch,
    unif = w_unif,
    dcsvmma = w_ma
  )
  method_names <- c(
    dcl = "DCSVMICL",
    dch = "DCSVMICH",
    sdcl = "SDCL",
    sdch = "SDCH",
    unif = "DCSVM-UNIF",
    dcsvmma = "DCSVMMA"
  )

  result_rows <- lapply(
    names(weight_list),
    function(method_key) {
      score <- as.numeric(Pred_Test %*% weight_list[[method_key]])
      smoothed_loss <- dcsvm_loss_mean(
        Y_test * score,
        get_dynamic_h(samplesize, C_scale),
        kernel
      )

      is_ma <- identical(method_key, "dcsvmma")

      data.frame(
        Kernel = kernel_label,
        MethodKey = method_key,
        Method = unname(method_names[[method_key]]),
        ER = get_error_rate(score, Y_test),
        SmoothedLoss = smoothed_loss,
        MinLoss = min_loss,
        NSHL = smoothed_loss / min_loss,
        Elapsed_Time = if (is_ma) elapsed_dcsvmma else NA_real_,
        CPU_Time = if (is_ma) cpu_dcsvmma else NA_real_,
        Elapsed_Base = if (is_ma) timing_base[["elapsed"]] else NA_real_,
        Elapsed_OOF = if (is_ma) timing_oof[["elapsed"]] else NA_real_,
        Elapsed_Solver = if (is_ma) timing_solver[["elapsed"]] else NA_real_,
        CPU_Base = if (is_ma) timing_base[["cpu"]] else NA_real_,
        CPU_OOF = if (is_ma) timing_oof[["cpu"]] else NA_real_,
        CPU_Solver = if (is_ma) timing_solver[["cpu"]] else NA_real_,
        Solver = if (is_ma) as.character(attr(w_ma, "solver")) else NA_character_,
        Fallback = if (is_ma) isTRUE(attr(w_ma, "fallback")) else NA,
        stringsAsFactors = FALSE
      )
    }
  )

  list(
    results = do.call(rbind, result_rows),
    fold_signature = paste0("F", paste(folds, collapse = ""))
  )
}

# ==============================================================================
# 5. 顺序主循环
# ==============================================================================

cat("============================================================\n")
cat("Epanechnikov 核 DCSVM 家族独立计算\n")
cat("外层并行：关闭\n")
cat("DCSVM 筛选：标准整条 lambda 路径\n")
cat("============================================================\n")

all_results <- list()
all_errors <- list()
result_counter <- 0L
error_counter <- 0L

for (DGP_TYPE in TARGET_DGP) {
  K_FIXED <- as.integer(K_FIXED_DCSVM_BY_DGP[[DGP_TYPE]])

  for (IS_MISSPEC in TARGET_MISSPEC) {
    MODEL_TYPE <- if (IS_MISSPEC) "Misspec" else "Correct"

    for (n_train in N_TRAIN_SEQ) {
      for (rep_idx in seq_len(N_REPEATS)) {
        dgp_code <- if (identical(DGP_TYPE, "DGP1")) 1L else 2L
        data_seed <- as.integer(
          DATA_SEED_BASE +
            dgp_code * 1000000L +
            n_train * 100L +
            rep_idx
        )
        fold_seed <- as.integer(data_seed + 50000000L)

        set.seed(data_seed)
        total_n <- n_train + N_TEST
        sim_data <- generate_simulation_data(
          dgp_type = DGP_TYPE,
          p = P_SIM,
          total_n = total_n,
          is_misspec = IS_MISSPEC,
          pos_ratio = IMBALANCE_RATIO
        )

        split_order <- sample.int(total_n)
        train_idx <- split_order[seq_len(n_train)]
        test_idx <- split_order[n_train + seq_len(N_TEST)]

        X_train <- sim_data$X[train_idx, , drop = FALSE]
        Y_train <- sim_data$Y[train_idx]
        X_test <- sim_data$X[test_idx, , drop = FALSE]
        Y_test <- sim_data$Y[test_idx]

        if (
          length(unique(Y_train)) < 2L ||
            length(unique(Y_test)) < 2L
        ) {
          stop("训练集或测试集只有一个类别。")
        }

        data_signature <- make_data_signature(X_train, Y_train)

        for (C_SCALE in C_SCALE_VALUES) {
          cat(sprintf(
            "[%s | %s | n=%d | rep=%d/%d | C=%s]\n",
            DGP_TYPE,
            MODEL_TYPE,
            n_train,
            rep_idx,
            N_REPEATS,
            format(C_SCALE, trim = TRUE)
          ))

          one_result <- tryCatch(
            run_dcsvm_family_once(
              X_train = X_train,
              Y_train = Y_train,
              X_test = X_test,
              Y_test = Y_test,
              samplesize = n_train,
              k_fixed = K_FIXED,
              C_scale = C_SCALE,
              fold_seed = fold_seed,
              kernel = KERNEL_TYPE,
              kernel_label = KERNEL_LABEL,
              Jn = JN
            ),
            error = function(e) e
          )

          if (inherits(one_result, "error")) {
            error_counter <- error_counter + 1L
            all_errors[[error_counter]] <- data.frame(
              Kernel = KERNEL_LABEL,
              C = C_SCALE,
              DGP = DGP_TYPE,
              ModelType = MODEL_TYPE,
              n = n_train,
              p = P_SIM,
              K = K_FIXED,
              Replication = rep_idx,
              DataSeed = data_seed,
              FoldSeed = fold_seed,
              Error = conditionMessage(one_result),
              stringsAsFactors = FALSE
            )
            message("  FAILED: ", conditionMessage(one_result))
          } else {
            result_counter <- result_counter + 1L
            scenario_columns <- data.frame(
              C = C_SCALE,
              DGP = DGP_TYPE,
              ModelType = MODEL_TYPE,
              IsMisspec = IS_MISSPEC,
              n = n_train,
              p = P_SIM,
              K = K_FIXED,
              Replication = rep_idx,
              DataSeed = data_seed,
              FoldSeed = fold_seed,
              DataSignature = data_signature,
              FoldSignature = one_result$fold_signature,
              stringsAsFactors = FALSE
            )

            all_results[[result_counter]] <- cbind(
              scenario_columns[
                rep(1L, nrow(one_result$results)),
                ,
                drop = FALSE
              ],
              one_result$results
            )

            checkpoint <- do.call(rbind, all_results)
            utils::write.csv(
              checkpoint,
              OUTPUT_FILE,
              row.names = FALSE
            )
          }

          if (length(all_errors) > 0L) {
            utils::write.csv(
              do.call(rbind, all_errors),
              ERROR_FILE,
              row.names = FALSE
            )
          }
        }
      }
    }
  }
}

if (length(all_results) == 0L) {
  stop("没有任何成功的 Epanechnikov 核结果。")
}

final_results <- do.call(rbind, all_results)
utils::write.csv(final_results, OUTPUT_FILE, row.names = FALSE)

cat("\nEpanechnikov 核计算完成。\n")
cat("Result: ", OUTPUT_FILE, "\n", sep = "")
if (length(all_errors) > 0L) {
  cat("Errors: ", ERROR_FILE, "\n", sep = "")
}
