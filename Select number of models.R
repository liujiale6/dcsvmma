# 模型个数选择(5)_每次独立生成数据.R
# ==============================================================================
# 模拟数据探究：SVMMA 与 DCSVMMA 的 训练/测试误差 随候选模型个数 (k) 的变化趋势
# 设定：每次重复生成 N_TRAIN 个训练样本和 N_TEST 个测试样本。
# Correct/Misspec 使用相同的6信号底层数据；Misspec 仅隐去 X6 并用噪声变量补齐维度。
# ==============================================================================

libs <- c("MASS", "e1071", "caret", "doParallel", "foreach", "LiblineaR", "Rsolnp", "dcsvm", "ggplot2", "tidyr")
for (l in libs) { if (!require(l, character.only = TRUE)) install.packages(l) }

library(MASS); library(e1071); library(caret); library(doParallel); library(foreach)
library(LiblineaR); library(Rsolnp); library(dcsvm); library(ggplot2); library(tidyr)

# ==========================================
# 0. 模拟数据生成函数
# ==========================================
# DGP1：均值漂移的高维正态分类模型。
generate_data_fun <- function(
    p,
    total_n,
    ismisspecification = FALSE,
    pos_ratio = 0.5
) {
  # Correct 和 Misspec 的真实底层模型完全一致：
  # p_full = p + 1 个原始变量，前 q_true = 5 个均为真实信号。
  # Correct 保留 X1,...,X5；Misspec 删除 X5，改用最后一个噪声变量，
  # 从而两者交给模型的最终维度都是 p。
  q_true <- 5L
  p_full <- as.integer(p) + 1L
  mu_val <- 0.8
  rho <- 0.5
  
  if (p < q_true) {
    stop("p 必须不小于真实信号变量数 q_true = 6。")
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

# DGP2：Logistic 分类模型
generate_data_fun2 <- function(
    p,
    total_n,
    ismisspecification = TRUE,
    pos_ratio = 0.5
) {
  # 有效变量数量
  q <- 5
  
  # 信号强度和变量相关性
  beta_value <- 0.5
  rho <- 0.3
  
  if (pos_ratio <= 0 || pos_ratio >= 1) {
    stop("pos_ratio 必须在 0 和 1 之间。")
  }
  
  if (ismisspecification) {
    q <- q + 1
    p <- p + 1
  }
  
  # 真实系数
  beta <- rep(0, p)
  beta[1:q] <- beta_value
  
  # AR(1) 协方差矩阵
  Sigma <- rho ^ abs(
    outer(
      1:p,
      1:p,
      "-"
    )
  )
  
  # 生成自变量
  X <- MASS::mvrnorm(
    n = total_n,
    mu = rep(0, p),
    Sigma = Sigma
  )
  
  X_beta <- as.numeric(X %*% beta)
  
  # 当 pos_ratio = 0.5 时，由于 X_beta 关于 0 对称，
  # Logistic 截距正好等于 0。
  if (abs(pos_ratio - 0.5) < 1e-12) {
    b0 <- 0
  } else {
    # 对其他类别比例，通过数值方法求解截距，
    # 使平均正类概率等于目标 pos_ratio。
    b0 <- stats::uniroot(
      function(intercept) {
        mean(stats::plogis(intercept + X_beta)) -
          pos_ratio
      },
      lower = -50,
      upper = 50
    )$root
  }
  
  # Logistic 正类概率
  prob <- stats::plogis(
    b0 + X_beta
  )
  
  # 生成类别标签
  Y <- stats::rbinom(
    n = total_n,
    size = 1,
    prob = prob
  )
  
  Y[Y == 0] <- -1
  
  # 错误设定时，删除新增的第 q 个真实变量
  inds <- if (ismisspecification) {
    c(
      1:(q - 1),
      (q + 1):p
    )
  } else {
    1:p
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

# ==========================================
# 1. 核心数学与工具函数
# ==========================================
get_dynamic_h <- function(n, C = 1.0) { return(C * n^(-1 / 5)) }

get_dynamic_lam2 <- function(n) {
  return(1 / (2 * n))
}

L_h_E <- function(v, h) {
  result <- numeric(length(v))
  case1 <- v <= 1 - h
  result[case1] <- 1 - v[case1]
  case2 <- (v > 1 - h) & (v <= 1 + h)
  if(any(case2)) {
    v_c2 <- v[case2]; h_c2 <- if(length(h) > 1) h[case2] else h
    result[case2] <- ((1 - v_c2 + h_c2)^3 * (3 * h_c2 - (1 - v_c2))) / (16 * h_c2^3)
  }
  return(result)
}
dcsvm_loss_mean <- function(v, h) { mean(L_h_E(v, h)) }

# DCSVMMA 特征筛选：使用 dcsvm 自带的标准整条 lambda 路径。
# 一次路径拟合后，按变量在整条路径上的稳定性和最大系数绝对值排序，
# 不再额外调用 cv.dcsvm() 进行内层 5 折交叉验证。
Indexs_fun_dcsvm <- function(
    X,
    Y,
    k = 30,
    h_val,
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
      kern = "epanechnikov",
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

  # lambda 越大，L1 压缩越强。从小 lambda 到大 lambda 遍历，
  # 越晚被压到 0 的变量，在正则化路径上越稳定。
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

# 对齐 Python L1-SVM 路径的统计含义，并用确定性规则处理同一步进入的变量。
Indexs_fun_lasso <- function(X, Y, k = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  p <- ncol(X)
  if (is.null(k)) k <- p
  k <- min(as.integer(k), p)

  num_steps <- max(50, floor(ncol(X) / 5))
  lams <- seq(10, 0.001, length.out = num_steps)

  # Python 的意图是：C 从大到小时，越晚被压到 0 的变量越稳定。
  # 原 R 写法在大量变量同一步为 0 时仅按列号倒序，容易把噪声变量排在前面。
  first_zero_step <- rep(NA_integer_, p)
  max_abs_coef <- numeric(p)
  successful_fits <- 0L

  for (step_index in seq_along(lams)) {
    lam <- lams[step_index]
    tryCatch({
      clf <- LiblineaR(
        data = X,
        target = Y,
        type = 5,
        cost = lam,
        bias = 1
      )
      beta <- as.numeric(as.matrix(clf$W)[1, seq_len(p)])
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
    stop("All L1-SVM fits failed during candidate-feature screening.")
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
  return(Indexs_mat)
}

# 提取与 predict.LiblineaR() 标签方向一致的连续判别分数。
# 不再假设 ClassNames[1] 一定对应 W 的固定正负方向。
train_safe_svm <- function(X, Y, C_val=1) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  clf <- LiblineaR(
    data = X,
    target = Y,
    type = 3,
    cost = C_val,
    bias = 1
  )

  p <- ncol(X)
  w_raw <- as.numeric(as.matrix(clf$W)[1, ])
  feature_weights <- w_raw[seq_len(p)]
  intercept <- if (length(w_raw) > p) {
    w_raw[p + 1L] * as.numeric(clf$Bias)
  } else {
    0
  }
  beta_full <- c(intercept, feature_weights)

  package_prediction <- as.numeric(as.character(
    predict(clf, X)$predictions
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
    stop(
      sprintf(
        "Unable to orient LiblineaR decision scores (agreement=%.4f).",
        direct_agreement
      )
    )
  }

  return(beta_full)
}

# ==========================================
# 2. 主执行代码
# ==========================================
if (sys.nframe() == 0) {
  
  WORK_DIR <- getwd()
  setwd(WORK_DIR)
  start_t <- Sys.time()
  
  cat("==== 启动模型个数选择：每次重复独立生成数据 ====\n")
  
  # =========================================================
  # >>> [核心实验参数配置区] <<<
  # =========================================================
  TARGET_DGP <- c("DGP1","DGP2")
  TARGET_MISSPEC <- c(FALSE, TRUE)
  
  P_SIM <- 1000
  N_TRAIN <- 300       # <--- 指定使用的训练样本数
  N_TEST <- 10000      # 测试样本数
  K_MAX <- 100         # 最大绘制到 100 个模型 
  N_REPEATS <- 10      # 独立数据重复次数
  C_SCALE <- 0.2       # 平滑带宽系数
  
  # 0.5 表示目标正类比例为 50%，即平衡数据。
  IMBALANCE_RATIO <- 0.5
  
  # 与主模拟文件采用相同的数据随机种子基数。
  # 种子不包含 IS_MISSPEC，因此 Correct/Misspec 共用相同底层完整数据，
  # 区别只是 Misspec 隐去了真实信号 X6。
  DATA_SEED_BASE <- 1000
  # =========================================================
  
  scenario_grid <- expand.grid(DGP_TYPE = TARGET_DGP, IS_MISSPEC = TARGET_MISSPEC, stringsAsFactors = FALSE)
  total_scenarios <- nrow(scenario_grid)
  
  cl <- makeCluster(min(parallel::detectCores() - 1, 10))
  registerDoParallel(cl)
  
  cat(sprintf("配置: N_Train=%d | N_Test=%d | K_max=%d | Repeats=%d | C=%.2f\n\n", 
              N_TRAIN, N_TEST, K_MAX, N_REPEATS, C_SCALE))
  
  for (row_idx in 1:nrow(scenario_grid)) {
    DGP_TYPE <- scenario_grid$DGP_TYPE[row_idx]
    IS_MISSPEC <- scenario_grid$IS_MISSPEC[row_idx]
    MODEL_TYPE <- ifelse(IS_MISSPEC, "Misspec", "Correct")
    
    cat(sprintf("▶ [%d/%d] 分析场景: %s | 架构: %s\n", row_idx, total_scenarios, DGP_TYPE, MODEL_TYPE))
    
    cat(sprintf(
      "  => 正在并行计算 %d 次独立数据重复...\n",
      N_REPEATS
    ))
    
    results_list <- foreach(
      rep_idx = 1:N_REPEATS,
      .packages = libs,
      .export = c(
        "C_SCALE", "N_TRAIN", "N_TEST", "K_MAX",
        "P_SIM", "DGP_TYPE", "IS_MISSPEC",
        "IMBALANCE_RATIO", "DATA_SEED_BASE",
        "generate_data_fun", "generate_data_fun2",
        "generate_simulation_data", "get_dynamic_h",
        "get_dynamic_lam2", "L_h_E", "dcsvm_loss_mean",
        "Indexs_fun_dcsvm", "Indexs_fun_lasso",
        "train_safe_svm"
      )
    ) %dopar% {
      
      # 每个 DGP 和重复次数使用独立种子；
      # Correct/Misspec 不改变数据种子，实现配对比较。
      dgp_code <- if (identical(DGP_TYPE, "DGP1")) 1 else 2
      data_seed <- DATA_SEED_BASE +
        dgp_code * 1000000 +
        N_TRAIN * 100 +
        rep_idx
      fold_seed <- data_seed + 50000000L
      set.seed(data_seed)
      
      total_n <- N_TRAIN + N_TEST
      sim_data <- generate_simulation_data(
        dgp_type = DGP_TYPE,
        p = P_SIM,
        total_n = total_n,
        is_misspec = IS_MISSPEC,
        pos_ratio = IMBALANCE_RATIO
      )
      
      # 先打乱本次新生成的数据，再划分训练集和测试集。
      split_order <- sample.int(total_n)
      train_idx <- split_order[seq_len(N_TRAIN)]
      test_idx <- split_order[N_TRAIN + seq_len(N_TEST)]
      
      X_train <- sim_data$X[train_idx, , drop = FALSE]
      Y_train_num <- sim_data$Y[train_idx]
      X_pre <- sim_data$X[test_idx, , drop = FALSE]
      Y_pre_num <- sim_data$Y[test_idx]
      
      if (length(unique(Y_train_num)) < 2 ||
          length(unique(Y_pre_num)) < 2) {
        stop(
          "生成的数据中训练集或测试集只包含一个类别；",
          "请更换 DATA_SEED_BASE 后重新运行。"
        )
      }
      
      df_rep <- data.frame()
      
      # ----------------------------------------------------
      # 阶段 A：分别获取特征排序
      # ----------------------------------------------------
      h_opt_global <- get_dynamic_h(n = N_TRAIN, C = C_SCALE)
      Indexs_svm <- Indexs_fun_lasso(X_train, Y_train_num, k = K_MAX)
      Indexs_dcsvm <- Indexs_fun_dcsvm(
        X = X_train,
        Y = Y_train_num,
        k = K_MAX,
        h_val = h_opt_global,
        screen_seed = fold_seed + 100L
      )
      
      # ----------------------------------------------------
      # 阶段 B：提前训练满维度的 Base Models (获得 Train/Test 预测边距)
      # ----------------------------------------------------
      Pred_Test_svm <- matrix(0, nrow = N_TEST, ncol = K_MAX)
      Pred_Train_svm <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)  
      
      Pred_Test_dc <- matrix(0, nrow = N_TEST, ncol = K_MAX)
      Pred_Train_dc <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)   
      
      for (i in 1:K_MAX) {
        # SVM
        vars_s <- which(Indexs_svm[i, ])
        b_res <- train_safe_svm(X_train[, vars_s, drop=F], Y_train_num, C_val=1)
        Pred_Test_svm[, i] <- cbind(1, X_pre[, vars_s, drop=F]) %*% c(b_res[1], b_res[-1])
        Pred_Train_svm[, i] <- cbind(1, X_train[, vars_s, drop=F]) %*% c(b_res[1], b_res[-1])
        
        # DCSVM
        vars_d <- which(Indexs_dcsvm[i, ])
        h_train <- get_dynamic_h(n = N_TRAIN, C = C_SCALE)
        
        lam2_train <- get_dynamic_lam2(N_TRAIN)
        
        fit_dc <- suppressWarnings(dcsvm::dcsvm(
          x = X_train[, vars_d, drop = FALSE],
          y = Y_train_num,
          lambda = 0,
          lam2 = lam2_train,
          hval = h_train,
          kern = "epanechnikov"
        ))
        Pred_Test_dc[, i] <- X_pre[, vars_d, drop=F] %*% as.numeric(fit_dc$beta) + fit_dc$b0
        Pred_Train_dc[, i] <- X_train[, vars_d, drop=F] %*% as.numeric(fit_dc$beta) + fit_dc$b0
      }
      
      # ----------------------------------------------------
      # 阶段 C：提前跑满交叉验证，生成 OOF 预测用于解权重
      # ----------------------------------------------------
      Jn <- 5
      set.seed(as.integer(fold_seed))
      folds <- createFolds(
        as.factor(Y_train_num),
        k = Jn,
        list = FALSE
      )
      Pred_OOF_svm <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)
      Pred_OOF_dc <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)
      h_oof <- numeric(N_TRAIN)
      
      for (j in 1:Jn) {
        tr_idx <- (folds != j); val_idx <- (folds == j)
        h_train_cv <- get_dynamic_h(n = sum(tr_idx), C = C_SCALE)
        lam2_cv <- get_dynamic_lam2(sum(tr_idx))
        h_oof[val_idx] <- h_train_cv
        
        # 当前外层训练折内重新进行特征筛选；验证折不参与筛选。
        Indexs_svm_fold <- Indexs_fun_lasso(
          X_train[tr_idx, , drop = FALSE],
          Y_train_num[tr_idx],
          k = K_MAX
        )
        Indexs_dcsvm_fold <- Indexs_fun_dcsvm(
          X_train[tr_idx, , drop = FALSE],
          Y_train_num[tr_idx],
          k = K_MAX,
          h_val = h_train_cv,
          screen_seed = fold_seed + 1000L + j
        )
        
        if (nrow(Indexs_svm_fold) != K_MAX) {
          stop("SVM 第 ", j, " 折筛选得到的候选模型数不一致。")
        }
        if (nrow(Indexs_dcsvm_fold) != K_MAX) {
          stop("DCSVM 第 ", j, " 折筛选得到的候选模型数不一致。")
        }
        
        for (i in 1:K_MAX) {
          vars_s <- which(Indexs_svm_fold[i, ])
          b_cv_svm <- train_safe_svm(X_train[tr_idx, vars_s, drop=F], Y_train_num[tr_idx], C_val=1)
          Pred_OOF_svm[val_idx, i] <- cbind(1, X_train[val_idx, vars_s, drop=F]) %*% c(b_cv_svm[1], b_cv_svm[-1])
          
          vars_d <- which(Indexs_dcsvm_fold[i, ])
          fit_cv_dc <- suppressWarnings(dcsvm::dcsvm(
            x = X_train[tr_idx, vars_d, drop = FALSE],
            y = Y_train_num[tr_idx],
            lambda = 0,
            lam2 = lam2_cv,
            hval = h_train_cv,
            kern = "epanechnikov"
          ))
          Pred_OOF_dc[val_idx, i] <- X_train[val_idx, vars_d, drop=F] %*% as.numeric(fit_cv_dc$beta) + fit_cv_dc$b0
        }
      }
      
      # 权重与误差计算辅助函数
      solve_weights <- function(Pred_mat, loss_type, h_vec = NULL) {
        if (ncol(Pred_mat) == 1) {
          result <- 1
          attr(result, "fallback") <- FALSE
          return(result)
        }
        
        w0 <- rep(1 / ncol(Pred_mat), ncol(Pred_mat))
        
        if (loss_type == "hinge") {
          obj_fn <- function(w) {
            mean(pmax(0, 1 - Y_train_num * as.numeric(Pred_mat %*% w)))
          }
        } else {
          obj_fn <- function(w) {
            dcsvm_loss_mean(Y_train_num * as.numeric(Pred_mat %*% w), h_vec)
          }
        }

        individual_losses <- vapply(
          seq_len(ncol(Pred_mat)),
          function(model_index) {
            one_hot <- rep(0, ncol(Pred_mat))
            one_hot[model_index] <- 1
            obj_fn(one_hot)
          },
          numeric(1)
        )
        best_single <- rep(0, ncol(Pred_mat))
        best_single[which.min(individual_losses)] <- 1
        
        solver_weights <- tryCatch({
          res <- suppressWarnings(solnp(
            pars = w0,
            fun = obj_fn,
            eqfun = function(w) sum(w) - 1,
            eqB = 0,
            LB = rep(0, ncol(Pred_mat)),
            UB = rep(1, ncol(Pred_mat)),
            control = list(trace = 0)
          ))
          candidate <- as.numeric(res$pars)
          if (
            length(candidate) != ncol(Pred_mat) ||
              any(!is.finite(candidate))
          ) {
            stop("The optimizer returned invalid weights.")
          }
          candidate <- pmin(1, pmax(0, candidate))
          if (sum(candidate) <= 0) {
            stop("The optimizer returned zero total weight.")
          }
          candidate / sum(candidate)
        }, error = function(e) {
          message("SVMMA/DCSVMMA weight solving failed: ", conditionMessage(e))
          NULL
        })
        
        candidates <- list(uniform = w0, best_single = best_single)
        if (!is.null(solver_weights)) {
          candidates$solnp <- solver_weights
        }
        candidate_losses <- vapply(candidates, obj_fn, numeric(1))
        selected_name <- names(which.min(candidate_losses))[1]
        w_opt <- candidates[[selected_name]]
        w_opt[abs(w_opt) < 1e-5] <- 0
        w_opt <- w_opt / sum(w_opt)
        fallback <- !identical(selected_name, "solnp")
        attr(w_opt, "fallback") <- fallback
        attr(w_opt, "solver") <- selected_name
        return(w_opt)
      }
      
      get_error <- function(Pred_mat, w, Y_truth) {
        preds <- as.numeric(Pred_mat %*% w)
        res_y <- sign(preds); res_y[res_y == 0] <- 1
        return(mean(res_y != Y_truth))
      }
      
      # ----------------------------------------------------
      # 阶段 D：从 1 递增到 K_MAX 截取计算
      # ----------------------------------------------------
      for (k in 1:K_MAX) {
        
        w_svm <- solve_weights(Pred_OOF_svm[, 1:k, drop=F], "hinge")
        er_svm_train <- get_error(Pred_Train_svm[, 1:k, drop=F], w_svm, Y_train_num)
        er_svm_test  <- get_error(Pred_Test_svm[, 1:k, drop=F], w_svm, Y_pre_num)
        
        w_dc <- solve_weights(Pred_OOF_dc[, 1:k, drop=F], "smoothed", h_oof)
        er_dc_train <- get_error(Pred_Train_dc[, 1:k, drop=F], w_dc, Y_train_num)
        er_dc_test  <- get_error(Pred_Test_dc[, 1:k, drop=F], w_dc, Y_pre_num)
        
        df_rep <- rbind(df_rep, data.frame(
          k = k,
          SVMMA_Train = er_svm_train, SVMMA_Test = er_svm_test,
          DCSVMMA_Train = er_dc_train, DCSVMMA_Test = er_dc_test,
          stringsAsFactors = FALSE
        ))
      }
      return(df_rep)
    }
    
    cat("  => 聚合完毕，正在绘制当前场景 4 线图...\n")
    
    df_all <- do.call(rbind, results_list)
    df_mean <- aggregate(. ~ k, data = df_all, FUN = mean)
    
    df_long <- pivot_longer(df_mean, cols = -k, names_to = "Condition", values_to = "ER")
    
    # 将 Method 与 Train/Test 合并成一个图例变量，避免出现两个图例
    df_long$Condition <- factor(
      df_long$Condition,
      levels = c("DCSVMMA_Train", "DCSVMMA_Test", "SVMMA_Train", "SVMMA_Test"),
      labels = c("dcsvmma_train", "dcsvmma_test", "svmma_train", "svmma_test")
    )
    
    color_palette <- c(
      "dcsvmma_train" = "#E41A1C", "dcsvmma_test" = "#E41A1C",
      "svmma_train"   = "#984EA3", "svmma_test"   = "#984EA3"
    )
    linetype_palette <- c(
      "dcsvmma_train" = "dashed", "dcsvmma_test" = "solid",
      "svmma_train"   = "dashed", "svmma_test"   = "solid"
    )
    shape_palette <- c(
      "dcsvmma_train" = 16, "dcsvmma_test" = 16,
      "svmma_train"   = 17, "svmma_test"   = 17
    )
    
    p_trend <- ggplot(df_long, aes(x = k, y = ER, color = Condition, linetype = Condition, shape = Condition)) +
      geom_line(linewidth = 1.2) + geom_point(size = 3) +
      scale_color_manual(name = "Method", values = color_palette) +
      scale_linetype_manual(name = "Method", values = linetype_palette) +
      scale_shape_manual(name = "Method", values = shape_palette) +
      labs(
        x = expression(paste("Number of Candidate Models (", k, ")")), 
        y = "Classification Error Rate (ER)"
      ) +
      theme_minimal() +
      theme(
        axis.title = element_text(size = 22),
        axis.text = element_text(size = 22),
        legend.position = c(0.98, 0.98),
        legend.justification = c(1, 1),
        legend.background = element_rect(fill = alpha("white", 0.85), color = NA),
        legend.title = element_text(size = 24, face = "bold"),
        legend.text = element_text(size = 18),
        legend.key.width = unit(3.5, "line"),
        legend.key.height = unit(1.2, "line"),
        legend.margin = ggplot2::margin(4, 6, 4, 6) # <--- 这里修正了由于 margin 冲突导致的画图报错
      ) +
      guides(
        color = guide_legend(order = 1, override.aes = list(linewidth = 1.6, size = 4)),
        linetype = guide_legend(order = 1, override.aes = list(linewidth = 1.6, size = 4)),
        shape = guide_legend(order = 1, override.aes = list(linewidth = 1.6, size = 4))
      )
    
    if (!dir.exists("../figures/Method_Compare/")) dir.create("../figures/Method_Compare/", recursive = TRUE)
    
    img_name <- sprintf("../figures/Method_Compare/Trend_ER_Sim_%s_%s_n%d.png", DGP_TYPE, MODEL_TYPE, N_TRAIN)
    data_name <- sprintf("../figures/Method_Compare/Trend_ER_Sim_%s_%s_n%d_Data.csv", DGP_TYPE, MODEL_TYPE, N_TRAIN)
    
    ggsave(img_name, p_trend, width = 12, height = 7, dpi = 300)
    write.csv(df_mean, data_name, row.names = FALSE)
    
    cat(sprintf("  => 图表已保存: %s\n", img_name))
  }
  
  stopCluster(cl)
  cat(sprintf("\n完美竣工！总耗时: %.2f 分钟\n", as.numeric(difftime(Sys.time(), start_t, units = "mins"))))
}
