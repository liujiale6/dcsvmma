# ==============================================================================
# 03_SVMMA_RF_独立计算.R
#
# 只计算 SVMMA 和 RandomForest。
#
# SVMMA 和 RandomForest 不依赖 DCSVM 核或带宽 C，因此每套数据
# 只训练和计时一次。之后使用 Epanechnikov/Gaussian 两种损失
# 和 C = 0.2、0.5、1 分别评价 NSHL，不重复训练和计时。
#
# 计时原则：
# - 顺序运行，不使用 foreach/doParallel。
# - SVMMA 时间 = 完整训练集候选模型 + 5 折 OOF + 权重求解。
# - 数据生成、测试集评价和文件写入不计入 SVMMA 时间。
# ==============================================================================

required_packages <- c(
  "MASS",
  "caret",
  "LiblineaR",
  "Rsolnp",
  "randomForest"
)
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# ==============================================================================
# 0. 统一参数：三个计算脚本必须保持一致
# ==============================================================================

C_SCALE_VALUES <- c(0.2)
TARGET_DGP <- c("DGP1", "DGP2")
TARGET_MISSPEC <- c(FALSE, TRUE)

P_SIM <- 1000L
N_TRAIN_SEQ <- c(100L, 200L, 300L, 400L)
N_TEST <- 1000L
N_REPEATS <- 100L
JN <- 5L

IMBALANCE_RATIO <- 0.5
DATA_SEED_BASE <- 1000L

K_FIXED_SVM_BY_DGP <- c(
  DGP1 = 25L,
  DGP2 = 20L
)

EVALUATION_KERNELS <- c("Epanechnikov", "Gaussian")

RESULT_ROOT <- file.path(getwd(), "Split_Results")
OUTPUT_DIR <- file.path(RESULT_ROOT, "SVMMA_RF")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

OUTPUT_FILE <- file.path(
  OUTPUT_DIR,
  "SVMMA_RF_replications.csv"
)
ERROR_FILE <- file.path(
  OUTPUT_DIR,
  "SVMMA_RF_errors.csv"
)

# ==============================================================================
# 1. 数据生成：与两个 DCSVM 脚本完全一致
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

smoothed_hinge_values <- function(v, h, kernel_label) {
  v <- as.numeric(v)
  if (length(h) == 1L) {
    h <- rep(as.numeric(h), length(v))
  } else {
    h <- as.numeric(h)
  }

  if (length(h) != length(v) || any(!is.finite(h)) || any(h <= 0)) {
    stop("h 必须是正的标量或与 v 等长的正向量。")
  }

  if (identical(kernel_label, "Gaussian")) {
    z <- (1 - v) / h
    return((1 - v) * stats::pnorm(z) + h * stats::dnorm(z))
  }

  if (!identical(kernel_label, "Epanechnikov")) {
    stop("不支持的评价核：", kernel_label)
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

smoothed_loss_mean <- function(v, h, kernel_label) {
  mean(smoothed_hinge_values(v, h, kernel_label))
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

solve_simplex_weights <- function(
    Pred_mat,
    Y,
    loss_type,
    h_vec = NULL,
    kernel_label = NULL
) {
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
      smoothed_loss_mean(
        Y * as.numeric(Pred_mat %*% w),
        h_vec,
        kernel_label
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
# 3. SVM 特征筛选和安全拟合
# ==============================================================================

Indexs_fun_lasso <- function(X, Y, k, seed_base = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  p <- ncol(X)
  k <- min(as.integer(k), p)

  num_steps <- max(50L, floor(p / 5))
  lams <- seq(10, 0.001, length.out = num_steps)

  first_zero_step <- rep(NA_integer_, p)
  max_abs_coef <- numeric(p)
  successful_fits <- 0L

  for (step_index in seq_along(lams)) {
    lam <- lams[step_index]

    tryCatch({
      if (!is.null(seed_base)) {
        set.seed(as.integer(seed_base + step_index))
      }

      fit <- LiblineaR::LiblineaR(
        data = X,
        target = Y,
        type = 5,
        cost = lam,
        bias = 1
      )

      beta <- as.numeric(as.matrix(fit$W)[1, seq_len(p)])
      max_abs_coef <- pmax(max_abs_coef, abs(beta))

      newly_zero <- which(
        abs(beta) <= 1e-8 &
          is.na(first_zero_step)
      )
      first_zero_step[newly_zero] <- step_index
      successful_fits <- successful_fits + 1L
    }, error = function(e) NULL)
  }

  if (successful_fits == 0L) {
    stop("All L1-SVM fits failed during feature screening.")
  }

  stability_step <- ifelse(
    is.na(first_zero_step),
    num_steps + 1L,
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

train_safe_svm <- function(X, Y, C_val = 1, fit_seed = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))

  if (!is.null(fit_seed)) {
    set.seed(as.integer(fit_seed))
  }

  fit <- LiblineaR::LiblineaR(
    data = X,
    target = Y,
    type = 3,
    cost = C_val,
    bias = 1
  )

  p <- ncol(X)
  w_raw <- as.numeric(as.matrix(fit$W)[1, ])
  feature_weights <- w_raw[seq_len(p)]
  intercept <- if (length(w_raw) > p) {
    w_raw[p + 1L] * as.numeric(fit$Bias)
  } else {
    0
  }
  beta_full <- c(intercept, feature_weights)

  package_prediction <- as.numeric(as.character(
    predict(fit, X)$predictions
  ))
  raw_score <- as.numeric(cbind(1, X) %*% beta_full)
  direct_prediction <- ifelse(raw_score >= 0, 1, -1)
  reversed_prediction <- -direct_prediction
  direct_agreement <- mean(direct_prediction == package_prediction)
  reversed_agreement <- mean(reversed_prediction == package_prediction)

  if (reversed_agreement > direct_agreement) {
    beta_full <- -beta_full
    direct_agreement <- reversed_agreement
  }
  if (!is.finite(direct_agreement) || direct_agreement < 0.99) {
    stop(sprintf(
      "Unable to orient LiblineaR decision scores (agreement=%.4f).",
      direct_agreement
    ))
  }

  beta_full
}

# ==============================================================================
# 4. 单次 SVMMA + RandomForest 计算
# ==============================================================================

run_svm_rf_once <- function(
    X_train,
    Y_train,
    X_test,
    Y_test,
    samplesize,
    k_fixed,
    fold_seed,
    C_values,
    evaluation_kernels,
    Jn = 5L
) {
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)
  Y_train <- as.numeric(as.character(Y_train))
  Y_test <- as.numeric(as.character(Y_test))
  p <- ncol(X_train)

  # A. 完整训练集的 SVM 筛选与候选模型。
  t0_base <- start_method_timer()

  Indexs_svm <- Indexs_fun_lasso(
    X = X_train,
    Y = Y_train,
    k = k_fixed,
    seed_base = fold_seed + 100000L
  )

  Betas_svm <- matrix(0, nrow = p + 1L, ncol = nrow(Indexs_svm))
  for (i in seq_len(nrow(Indexs_svm))) {
    vars <- which(Indexs_svm[i, ])
    beta_i <- train_safe_svm(
      X = X_train[, vars, drop = FALSE],
      Y = Y_train,
      C_val = 1,
      fit_seed = fold_seed + 200000L + i
    )

    Betas_svm[1, i] <- beta_i[1]
    Betas_svm[vars + 1L, i] <- beta_i[-1]
  }

  timing_base <- stop_method_timer(t0_base)

  # B. 与 DCSVM 脚本使用相同种子的 5 折 OOF。
  set.seed(as.integer(fold_seed))
  folds <- caret::createFolds(
    as.factor(Y_train),
    k = Jn,
    list = FALSE
  )

  Pred_OOF <- matrix(
    0,
    nrow = samplesize,
    ncol = nrow(Indexs_svm)
  )

  t0_oof <- start_method_timer()

  for (j in seq_len(Jn)) {
    tr_idx <- folds != j
    val_idx <- folds == j

    Indexs_fold <- Indexs_fun_lasso(
      X = X_train[tr_idx, , drop = FALSE],
      Y = Y_train[tr_idx],
      k = ncol(Pred_OOF),
      seed_base = fold_seed + 300000L + j * 1000L
    )

    if (nrow(Indexs_fold) != ncol(Pred_OOF)) {
      stop("SVM OOF 候选模型数不一致。")
    }

    for (i in seq_len(ncol(Pred_OOF))) {
      vars <- which(Indexs_fold[i, ])
      beta_i <- train_safe_svm(
        X = X_train[tr_idx, vars, drop = FALSE],
        Y = Y_train[tr_idx],
        C_val = 1,
        fit_seed = fold_seed + 400000L + j * 1000L + i
      )

      Pred_OOF[val_idx, i] <-
        as.numeric(
          cbind(1, X_train[val_idx, vars, drop = FALSE]) %*%
            beta_i
        )
    }
  }

  timing_oof <- stop_method_timer(t0_oof)

  # C. SVMMA 权重求解。
  set.seed(as.integer(fold_seed + 500000L))
  t0_solver <- start_method_timer()
  w_svmma <- solve_simplex_weights(
    Pred_mat = Pred_OOF,
    Y = Y_train,
    loss_type = "hinge"
  )
  timing_solver <- stop_method_timer(t0_solver)

  elapsed_svmma <-
    timing_base[["elapsed"]] +
    timing_oof[["elapsed"]] +
    timing_solver[["elapsed"]]
  cpu_svmma <-
    timing_base[["cpu"]] +
    timing_oof[["cpu"]] +
    timing_solver[["cpu"]]

  Pred_Test_svm <- cbind(1, X_test) %*% Betas_svm
  score_svmma <- as.numeric(Pred_Test_svm %*% w_svmma)
  er_svmma <- get_error_rate(score_svmma, Y_test)

  # D. RandomForest 独立计时。
  set.seed(as.integer(fold_seed + 900000L))
  t0_rf <- start_method_timer()
  rf_model <- randomForest::randomForest(
    x = X_train,
    y = as.factor(Y_train),
    ntree = 200,
    mtry = floor(sqrt(p)),
    importance = FALSE,
    keep.forest = TRUE
  )
  pred_rf <- predict(rf_model, X_test)
  timing_rf <- stop_method_timer(t0_rf)

  pred_rf_num <- as.numeric(as.character(pred_rf))
  er_rf <- mean(pred_rf_num != Y_test)

  # E. 不重新拟合，只针对两种核和三个 C 评价 NSHL。
  result_rows <- list()
  row_counter <- 0L

  for (kernel_label in evaluation_kernels) {
    for (C_scale in C_values) {
      h_test <- get_dynamic_h(samplesize, C_scale)

      w_test_opt <- solve_simplex_weights(
        Pred_mat = Pred_Test_svm,
        Y = Y_test,
        loss_type = "smoothed",
        h_vec = h_test,
        kernel_label = kernel_label
      )
      min_loss_svm <- max(
        smoothed_loss_mean(
          Y_test * as.numeric(Pred_Test_svm %*% w_test_opt),
          h_test,
          kernel_label
        ),
        0.01
      )

      smoothed_loss_svm <- smoothed_loss_mean(
        Y_test * score_svmma,
        h_test,
        kernel_label
      )

      row_counter <- row_counter + 1L
      result_rows[[row_counter]] <- data.frame(
        C = C_scale,
        Kernel = kernel_label,
        MethodKey = "svmma",
        Method = "SVMMA",
        ER = er_svmma,
        SmoothedLoss = smoothed_loss_svm,
        MinLoss = min_loss_svm,
        NSHL = smoothed_loss_svm / min_loss_svm,
        Elapsed_Time = elapsed_svmma,
        CPU_Time = cpu_svmma,
        Elapsed_Base = timing_base[["elapsed"]],
        Elapsed_OOF = timing_oof[["elapsed"]],
        Elapsed_Solver = timing_solver[["elapsed"]],
        CPU_Base = timing_base[["cpu"]],
        CPU_OOF = timing_oof[["cpu"]],
        CPU_Solver = timing_solver[["cpu"]],
        Solver = as.character(attr(w_svmma, "solver")),
        Fallback = isTRUE(attr(w_svmma, "fallback")),
        stringsAsFactors = FALSE
      )

      row_counter <- row_counter + 1L
      result_rows[[row_counter]] <- data.frame(
        C = C_scale,
        Kernel = kernel_label,
        MethodKey = "rf",
        Method = "RandomForest",
        ER = er_rf,
        SmoothedLoss = NA_real_,
        MinLoss = NA_real_,
        NSHL = NA_real_,
        Elapsed_Time = timing_rf[["elapsed"]],
        CPU_Time = timing_rf[["cpu"]],
        Elapsed_Base = NA_real_,
        Elapsed_OOF = NA_real_,
        Elapsed_Solver = NA_real_,
        CPU_Base = NA_real_,
        CPU_OOF = NA_real_,
        CPU_Solver = NA_real_,
        Solver = NA_character_,
        Fallback = NA,
        stringsAsFactors = FALSE
      )
    }
  }

  list(
    results = do.call(rbind, result_rows),
    fold_signature = paste0("F", paste(folds, collapse = ""))
  )
}

# ==============================================================================
# 5. 顺序主循环
# ==============================================================================

cat("============================================================\n")
cat("SVMMA + RandomForest 独立计算\n")
cat("外层并行：关闭\n")
cat("SVMMA 每套数据只训练和计时一次\n")
cat("============================================================\n")

all_results <- list()
all_errors <- list()
result_counter <- 0L
error_counter <- 0L

for (DGP_TYPE in TARGET_DGP) {
  K_FIXED <- as.integer(K_FIXED_SVM_BY_DGP[[DGP_TYPE]])

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

        cat(sprintf(
          "[%s | %s | n=%d | rep=%d/%d]\n",
          DGP_TYPE,
          MODEL_TYPE,
          n_train,
          rep_idx,
          N_REPEATS
        ))

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

        one_result <- tryCatch(
          run_svm_rf_once(
            X_train = X_train,
            Y_train = Y_train,
            X_test = X_test,
            Y_test = Y_test,
            samplesize = n_train,
            k_fixed = K_FIXED,
            fold_seed = fold_seed,
            C_values = C_SCALE_VALUES,
            evaluation_kernels = EVALUATION_KERNELS,
            Jn = JN
          ),
          error = function(e) e
        )

        if (inherits(one_result, "error")) {
          error_counter <- error_counter + 1L
          all_errors[[error_counter]] <- data.frame(
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

if (length(all_results) == 0L) {
  stop("没有任何成功的 SVMMA/RandomForest 结果。")
}

final_results <- do.call(rbind, all_results)
utils::write.csv(final_results, OUTPUT_FILE, row.names = FALSE)

cat("\nSVMMA + RandomForest 计算完成。\n")
cat("Result: ", OUTPUT_FILE, "\n", sep = "")
if (length(all_errors) > 0L) {
  cat("Errors: ", ERROR_FILE, "\n", sep = "")
}
