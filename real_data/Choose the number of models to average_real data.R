# 真实数据_模型个数选择
# ==============================================================================
# 真实数据探究：SVMMA 与 DCSVMMA 的训练/测试误差随候选模型个数 (k) 的变化趋势
# 设定：固定 C_SCALE = 1，固定使用总样本的 50% 训练，横轴为候选模型个数 k。
# ==============================================================================

libs <- c("caret", "doParallel", "foreach", "LiblineaR", "Rsolnp", "dcsvm", "ggplot2", "tidyr")
for (l in libs) { if (!require(l, character.only = TRUE)) install.packages(l) }

library(caret); library(doParallel); library(foreach)
library(LiblineaR); library(Rsolnp); library(dcsvm); library(ggplot2); library(tidyr)

if (!requireNamespace("datamicroarray", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("ramhiser/datamicroarray", upgrade = "never")
}

# ==========================================
# 0. 真实数据读取与训练/测试划分
# ==========================================
# 新增符号 data_name：用于在 "chowdary" 和 "gravier" 中选择数据。
load_real_data <- function(data_name) {
  # data_env 和 obj 分别表示临时数据环境及从包中读出的原始数据对象。
  data_env <- new.env(parent = baseenv())
  utils::data(list = data_name, package = "datamicroarray", envir = data_env)
  obj <- get(data_name, envir = data_env, inherits = FALSE)
  X <- as.matrix(obj$x)
  storage.mode(X) <- "double"
  # 与完整分析一致：把第一个因子水平编码为1，第二个水平编码为-1。
  Y_factor <- droplevels(as.factor(obj$y))
  if (nlevels(Y_factor) != 2L) stop(data_name, " 不是二分类数据。")
  Y <- ifelse(Y_factor == levels(Y_factor)[1L], 1, -1)
  list(X = X, Y = as.numeric(Y))
}

# 新增函数 variance_prescreen_and_scale：
# 只根据当前训练数据的原始方差保留方差最大的 variance_top_n 个特征，
# 再用该训练数据的均值和标准差处理训练数据及对应的待预测数据。
variance_prescreen_and_scale <- function(
    X_train_raw,
    X_apply_raw,
    variance_top_n = 1000L
) {
  X_train_raw <- as.matrix(X_train_raw)
  X_apply_raw <- as.matrix(X_apply_raw)
  storage.mode(X_train_raw) <- "double"
  storage.mode(X_apply_raw) <- "double"

  if (ncol(X_train_raw) != ncol(X_apply_raw)) {
    stop("训练数据和待预测数据的原始特征数不一致。")
  }
  if (!is.finite(variance_top_n) || variance_top_n < 1L) {
    stop("variance_top_n 必须是正整数。")
  }

  variance_top_n <- min(
    as.integer(variance_top_n),
    ncol(X_train_raw)
  )
  feature_variances <- apply(X_train_raw, 2, stats::var)
  feature_variances[!is.finite(feature_variances)] <- -Inf

  # 方差相同时按原始列号排序，保证结果完全可复现。
  variance_order <- order(
    -feature_variances,
    seq_along(feature_variances)
  )
  selected_features <- variance_order[seq_len(variance_top_n)]

  X_train_selected <- X_train_raw[
    , selected_features, drop = FALSE
  ]
  X_apply_selected <- X_apply_raw[
    , selected_features, drop = FALSE
  ]

  center_train <- colMeans(X_train_selected)
  scale_train <- apply(X_train_selected, 2, stats::sd)
  scale_train[!is.finite(scale_train) | scale_train < 1e-8] <- 1

  X_train_selected <- sweep(
    X_train_selected, 2, center_train, "-"
  )
  X_train_selected <- sweep(
    X_train_selected, 2, scale_train, "/"
  )
  X_apply_selected <- sweep(
    X_apply_selected, 2, center_train, "-"
  )
  X_apply_selected <- sweep(
    X_apply_selected, 2, scale_train, "/"
  )

  list(
    X_train = X_train_selected,
    X_apply = X_apply_selected,
    selected_features = selected_features
  )
}

# 按完整分析相同的createDataPartition方式抽取固定数量的训练样本。
# 方差预筛选、中心化和标准化参数全部只由训练集计算。
prepare_real_data_split <- function(
    real_data,
    n_train,
    data_seed,
    variance_top_n = 1000L
) {
  set.seed(data_seed)
  Y <- real_data$Y
  n_train <- as.integer(n_train)
  train_idx <- as.vector(
    caret::createDataPartition(
      as.factor(Y),
      p = n_train / length(Y),
      list = FALSE
    )
  )[seq_len(n_train)]
  test_idx <- setdiff(seq_along(Y), train_idx)

  X_train_raw <- real_data$X[train_idx, , drop = FALSE]
  X_pre_raw <- real_data$X[test_idx, , drop = FALSE]

  processed <- variance_prescreen_and_scale(
    X_train_raw = X_train_raw,
    X_apply_raw = X_pre_raw,
    variance_top_n = variance_top_n
  )

  list(
    X_train = processed$X_train,
    Y_train = Y[train_idx],
    X_pre = processed$X_apply,
    Y_pre = Y[test_idx],
    X_train_raw = X_train_raw,
    X_pre_raw = X_pre_raw,
    variance_features = processed$selected_features
  )
}

# ==========================================
# 1. 核心数学与工具函数
# ==========================================
get_dynamic_h <- function(n, C = 1.0) { return(C * n^(-1 / 5)) }

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

# 与完整分析一致：使用cv.dcsvm的50点lambda路径和5折内部交叉验证排序变量。
Indexs_fun_dcsvm <- function(X, Y, k = 30, h_val) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  p <- ncol(X)
  k <- min(as.integer(k), p)

  fit1 <- suppressWarnings(
    dcsvm::cv.dcsvm(
      X, Y,
      nlambda = 50,
      lam2 = 0,
      kern = "epanechnikov",
      hval = h_val,
      nfolds = 5
    )
  )

  features_ind <- integer(0)
  lambdas <- sort(fit1$lambda, decreasing = TRUE)
  for (lam in lambdas) {
    tryCatch({
      coef_1 <- as.numeric(
        coef(fit1, lam, type = "coefficients")
      )[-1]
      features_ind <- c(
        features_ind,
        setdiff(which(coef_1 != 0), features_ind)
      )
    }, error = function(e) NULL)
  }

  features_ind <- c(
    features_ind,
    setdiff(seq_len(p), features_ind)
  )[seq_len(k)]

  Indexs_mat <- matrix(FALSE, nrow = k, ncol = p)
  for (i in seq_len(k)) {
    Indexs_mat[i, features_ind[seq_len(i)]] <- TRUE
  }
  unique(Indexs_mat)
}

# 对齐 Python L1-SVM 路径的统计含义，并用确定性规则处理同一步进入的变量。
Indexs_fun_lasso <- function(X, Y, k = NULL, seed_base = NULL) {
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
      if (!is.null(seed_base)) {
        set.seed(as.integer(seed_base + step_index))
      }
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
train_safe_svm <- function(X, Y, C_val=1, fit_seed = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  if (!is.null(fit_seed)) {
    set.seed(as.integer(fit_seed))
  }
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
  
  cat("==== 启动真实数据模型个数选择 ====\n")
  
  # =========================================================
  # >>> [核心实验参数配置区] <<<
  # =========================================================
  # TARGET_DATA可保留一个或同时保留两个数据集。
  TARGET_DATA <- c("chowdary", "gravier")
  # 模型个数选择固定使用总样本的50%作为训练样本。
  TRAIN_RATIO <- 0.5
  # 与完整分析一致：每次只用训练集方差最大的3000个特征。
  VARIANCE_TOP_N <- 3000L
  K_MAX <- 50         # 最大绘制到 50 个模型 
  N_REPEATS <- 100L     # 独立重复100次
  C_SCALE <- 1.0       # 平滑带宽系数C=1
  N_FOLDS <- 10L       # 模型平均OOF使用10折

  detected_cores <- parallel::detectCores()
  if (!is.finite(detected_cores) || detected_cores < 2L) detected_cores <- 2L
  N_WORKERS <- max(1L, min(as.integer(detected_cores) - 1L, 10L))
  # =========================================================

  total_scenarios <- length(TARGET_DATA)
  
  cl <- makeCluster(N_WORKERS)
  registerDoParallel(cl)
  
  cat(sprintf(
    paste0(
      "配置: Train_Ratio=%.0f%% | Variance_Top=%d | K_max=%d",
      " | Repeats=%d | OOF_Folds=%d | C=%.2f | Workers=%d\n\n"
    ),
    100 * TRAIN_RATIO, VARIANCE_TOP_N, K_MAX,
    N_REPEATS, N_FOLDS, C_SCALE, N_WORKERS
  ))
  
  for (row_idx in seq_along(TARGET_DATA)) {
    DGP_TYPE <- TARGET_DATA[row_idx]
    MODEL_TYPE <- "RealData"
    # 新增符号 REAL_DATA：保存当前由 TARGET_DATA 选中的完整真实数据对象。
    REAL_DATA <- load_real_data(DGP_TYPE)
    N_TRAIN <- round(nrow(REAL_DATA$X) * TRAIN_RATIO)

    cat(sprintf("▶ [%d/%d] 分析场景: %s | 架构: %s\n", row_idx, total_scenarios, DGP_TYPE, MODEL_TYPE))
    
    cat(sprintf(
      "  => 正在并行计算 %d 次独立数据重复...\n",
      N_REPEATS
    ))
    
    results_list <- foreach(
      rep_idx = 0:(N_REPEATS - 1L),
      .packages = libs,
      .export = c(
        "C_SCALE", "N_TRAIN", "K_MAX", "VARIANCE_TOP_N", "N_FOLDS",
        "DGP_TYPE", "TARGET_DATA",
        "TRAIN_RATIO", "REAL_DATA",
        "variance_prescreen_and_scale", "prepare_real_data_split",
        "get_dynamic_h",
        "L_h_E", "dcsvm_loss_mean",
        "Indexs_fun_dcsvm", "Indexs_fun_lasso",
        "train_safe_svm"
      )
    ) %dopar% {
      
      # 与完整分析一致：相同训练样本量和重复编号使用相同种子。
      data_seed <- N_TRAIN * 1000L + rep_idx
      fold_seed <- data_seed
      set.seed(data_seed)
      
      # 唯一替换的数据步骤：每次重复重新抽取 50% 训练样本，其余作为测试样本。
      # 新增符号 real_split：保存本次重复的训练集、测试集及对应类别标签。
      real_split <- prepare_real_data_split(
        real_data = REAL_DATA,
        n_train = N_TRAIN,
        data_seed = data_seed,
        variance_top_n = VARIANCE_TOP_N
      )
      X_train <- real_split$X_train
      Y_train_num <- real_split$Y_train
      X_pre <- real_split$X_pre
      Y_pre_num <- real_split$Y_pre
      X_train_raw <- real_split$X_train_raw
      N_TRAIN <- nrow(X_train)
      
      if (length(unique(Y_train_num)) < 2 ||
          length(unique(Y_pre_num)) < 2) {
        stop(
          "真实数据的训练集或测试集只包含一个类别；",
          "请调整重复编号或随机种子公式后重新运行。"
        )
      }
      
      df_rep <- data.frame()
      
      # ----------------------------------------------------
      # 阶段 A：分别获取特征排序
      # ----------------------------------------------------
      h_opt_global <- get_dynamic_h(n = N_TRAIN, C = C_SCALE)
      Indexs_svm <- Indexs_fun_lasso(
        X_train,
        Y_train_num,
        k = K_MAX,
        seed_base = fold_seed + 100000L
      )
      Indexs_dcsvm <- Indexs_fun_dcsvm(
        X = X_train,
        Y = Y_train_num,
        k = K_MAX,
        h_val = h_opt_global
      )
      
      # ----------------------------------------------------
      # 阶段 B：提前训练满维度的 Base Models (获得 Train/Test 预测边距)
      # ----------------------------------------------------
      Pred_Test_svm <- matrix(0, nrow = nrow(X_pre), ncol = K_MAX)
      Pred_Train_svm <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)  
      
      Pred_Test_dc <- matrix(0, nrow = nrow(X_pre), ncol = K_MAX)
      Pred_Train_dc <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)   
      
      for (i in 1:K_MAX) {
        # SVM
        vars_s <- which(Indexs_svm[i, ])
        b_res <- train_safe_svm(
          X_train[, vars_s, drop = FALSE],
          Y_train_num,
          C_val = 1,
          fit_seed = fold_seed + 200000L + i
        )
        Pred_Test_svm[, i] <- cbind(1, X_pre[, vars_s, drop=F]) %*% c(b_res[1], b_res[-1])
        Pred_Train_svm[, i] <- cbind(1, X_train[, vars_s, drop=F]) %*% c(b_res[1], b_res[-1])
        
        # DCSVM
        vars_d <- which(Indexs_dcsvm[i, ])
        h_train <- get_dynamic_h(n = N_TRAIN, C = C_SCALE)
        
        fit_dc <- suppressWarnings(dcsvm::dcsvm(
          x = X_train[, vars_d, drop = FALSE],
          y = Y_train_num,
          lambda = 0,
          lam2 = 1,
          hval = h_train,
          kern = "epanechnikov"
        ))
        Pred_Test_dc[, i] <- X_pre[, vars_d, drop=F] %*% as.numeric(fit_dc$beta) + fit_dc$b0
        Pred_Train_dc[, i] <- X_train[, vars_d, drop=F] %*% as.numeric(fit_dc$beta) + fit_dc$b0
      }
      
      # ----------------------------------------------------
      # 阶段 C：提前跑满交叉验证，生成 OOF 预测用于解权重
      # ----------------------------------------------------
      Jn <- N_FOLDS
      set.seed(as.integer(fold_seed))
      folds <- createFolds(
        as.factor(Y_train_num),
        k = Jn,
        list = FALSE
      )
      Pred_OOF_svm <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)
      Pred_OOF_dc <- matrix(0, nrow = N_TRAIN, ncol = K_MAX)
      h_oof <- numeric(N_TRAIN)

      # 与完整分析的SVMMA一致：每个OOF训练折重新进行方差筛选和标准化。
      fold_processed_data <- lapply(seq_len(Jn), function(j) {
        tr_idx <- folds != j
        val_idx <- folds == j
        variance_prescreen_and_scale(
          X_train_raw = X_train_raw[tr_idx, , drop = FALSE],
          X_apply_raw = X_train_raw[val_idx, , drop = FALSE],
          variance_top_n = VARIANCE_TOP_N
        )
      })
      
      for (j in 1:Jn) {
        tr_idx <- (folds != j); val_idx <- (folds == j)
        X_fold_train <- fold_processed_data[[j]]$X_train
        X_fold_val <- fold_processed_data[[j]]$X_apply
        h_train_cv <- get_dynamic_h(n = sum(tr_idx), C = C_SCALE)
        h_oof[val_idx] <- h_train_cv
        
        # SVMMA与完整分析一致：当前OOF训练折内重新筛选特征。
        Indexs_svm_fold <- Indexs_fun_lasso(
          X_fold_train,
          Y_train_num[tr_idx],
          k = K_MAX,
          seed_base = fold_seed + 300000L + j * 1000L
        )
        
        if (nrow(Indexs_svm_fold) != K_MAX) {
          stop("SVM 第 ", j, " 折筛选得到的候选模型数不一致。")
        }
        
        for (i in 1:K_MAX) {
          vars_s <- which(Indexs_svm_fold[i, ])
          b_cv_svm <- train_safe_svm(
            X_fold_train[, vars_s, drop = FALSE],
            Y_train_num[tr_idx],
            C_val = 1,
            fit_seed = fold_seed + 400000L + j * 1000L + i
          )
          Pred_OOF_svm[val_idx, i] <-
            cbind(1, X_fold_val[, vars_s, drop = FALSE]) %*%
            c(b_cv_svm[1], b_cv_svm[-1])
          
          # DCSVMMA与完整分析一致：OOF中复用外层训练集得到的候选特征集合。
          vars_d <- which(Indexs_dcsvm[i, ])
          fit_cv_dc <- suppressWarnings(dcsvm::dcsvm(
            x = X_train[tr_idx, vars_d, drop = FALSE],
            y = Y_train_num[tr_idx],
            lambda = 0,
            lam2 = 1,
            hval = h_train_cv,
            kern = "epanechnikov"
          ))
          Pred_OOF_dc[val_idx, i] <-
            X_train[val_idx, vars_d, drop = FALSE] %*%
            as.numeric(fit_cv_dc$beta) + fit_cv_dc$b0
        }
      }
      
      # 权重与误差计算辅助函数
      solve_weights <- function(Pred_mat, loss_type, h_vec = NULL) {
        # 单个候选模型的唯一可行权重就是1；无需调用优化器。
        if (ncol(Pred_mat) == 1L) return(1)

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
        res <- suppressWarnings(
          solnp(
            pars = w0,
            fun = obj_fn,
            eqfun = function(w) sum(w) - 1,
            eqB = 0,
            LB = rep(0, ncol(Pred_mat)),
            UB = rep(1, ncol(Pred_mat)),
            control = list(trace = 0)
          )
        )
        w_opt <- ifelse(abs(res$pars) < 1e-5, 0, res$pars)
        if (sum(w_opt) > 0) w_opt / sum(w_opt) else w0
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
      labels = c("DCSVMMA_Train", "DCSVMMA_Test", "SVMMA_Train", "SVMMA_Test")
    )
    
    color_palette <- c(
      "DCSVMMA_Train" = "#E41A1C", "DCSVMMA_Test" = "#E41A1C",
      "SVMMA_Train"   = "#984EA3", "SVMMA_Test"   = "#984EA3"
    )
    linetype_palette <- c(
      "DCSVMMA_Train" = "dashed", "DCSVMMA_Test" = "solid",
      "SVMMA_Train"   = "dashed", "SVMMA_Test"   = "solid"
    )
    shape_palette <- c(
      "DCSVMMA_Train" = 16, "DCSVMMA_Test" = 16,
      "SVMMA_Train"   = 17, "SVMMA_Test"   = 17
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
