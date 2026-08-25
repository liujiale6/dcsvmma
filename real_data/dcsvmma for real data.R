# main_realdata.R
# ==============================================================================
# 真实数据: 每次重抽样独立特征排序 + 8种核心方法对决
# 修改：每次划分按训练集方差预筛选特征；修正SVMMA；删除单模型DCSVM；补充CPU/elapsed计时
# ==============================================================================

libs <- c("e1071", "caret", "doParallel", "foreach", "LiblineaR", "Rsolnp", "dcsvm", "ggplot2", "randomForest", "datamicroarray")
for (l in libs) { if (!require(l, character.only = TRUE)) install.packages(l) }

library(e1071); library(caret); library(doParallel); library(foreach); 
library(LiblineaR); library(Rsolnp); library(dcsvm); library(ggplot2)
library(randomForest); library(datamicroarray)

# ==========================================
# 0. 训练集方差预筛选与标准化
# ==========================================
# 只使用当前训练集计算方差、均值和标准差；测试集不参与任何筛选。
variance_prescreen_and_scale <- function(X_train_raw, X_apply_raw, variance_top_n = 1000L) {
  X_train_raw <- as.matrix(X_train_raw)
  X_apply_raw <- as.matrix(X_apply_raw)
  storage.mode(X_train_raw) <- "double"
  storage.mode(X_apply_raw) <- "double"

  if (ncol(X_train_raw) != ncol(X_apply_raw)) {
    stop("训练集和测试集的原始特征数不一致。")
  }

  variance_top_n <- min(as.integer(variance_top_n), ncol(X_train_raw))
  feature_variances <- apply(X_train_raw, 2, stats::var)
  feature_variances[!is.finite(feature_variances)] <- -Inf
  variance_order <- order(-feature_variances, seq_along(feature_variances))
  selected_features <- variance_order[seq_len(variance_top_n)]

  X_train <- X_train_raw[, selected_features, drop = FALSE]
  X_apply <- X_apply_raw[, selected_features, drop = FALSE]
  center_train <- colMeans(X_train)
  scale_train <- apply(X_train, 2, stats::sd)
  scale_train[!is.finite(scale_train) | scale_train < 1e-8] <- 1

  X_train <- sweep(sweep(X_train, 2, center_train, "-"), 2, scale_train, "/")
  X_apply <- sweep(sweep(X_apply, 2, center_train, "-"), 2, scale_train, "/")

  list(
    X_train = X_train,
    X_apply = X_apply,
    selected_features = selected_features
  )
}

# ==========================================
# 1. 纯正理论带宽与平滑损失
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

dcsvm_loss_sum <- function(v, h) { sum(L_h_E(v, h)) }
dcsvm_loss_mean <- function(v, h) { mean(L_h_E(v, h)) }

# 在并行 worker 内同时记录两种时间：
# 1. elapsed：实际经过的墙钟时间，会受到其他 worker 抢占 CPU 的影响；
# 2. cpu：当前 worker 消耗的 user + system CPU 时间，更适合比较方法计算量。
# 每段方法计时开始前先执行垃圾回收，减少前一个方法残留对象对当前计时的影响。
start_method_timer <- function() {
  invisible(gc(verbose = FALSE))
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

# ==========================================
# 2. 局部特征筛选与安全基模型
# ==========================================
Indexs_fun_dcsvm <- function(X, Y, k = 30, h_val) {
  p <- ncol(X)
  fit1 <- suppressWarnings(cv.dcsvm(X, Y, nlambda = 50, lam2=0, kern = "epanechnikov", hval = h_val, nfolds = 5))
  features_ind <- c(); lambdas <- sort(fit1$lambda, decreasing = TRUE)
  for (lam in lambdas) {
    tryCatch({
      coef_1 <- as.numeric(coef(fit1, lam, type = "coefficients"))[-1]
      features_ind <- c(features_ind, setdiff(which(coef_1 != 0), features_ind))
    }, error = function(e) NULL)
  }
  features_ind <- c(features_ind, setdiff(1:p, features_ind))[1:min(k, p)]
  Indexs_mat <- matrix(FALSE, nrow = length(features_ind), ncol = p)
  for (i in 1:nrow(Indexs_mat)) Indexs_mat[i, features_ind[1:i]] <- TRUE 
  return(unique(Indexs_mat)) 
}

# SVMMA 的候选特征路径：沿一整条 L1-SVM cost 路径逐次拟合，
# 再按变量在路径上的稳定性和最大系数绝对值排序，构造1,...,k个嵌套模型。
Indexs_fun_lasso <- function(X, Y, k = NULL, seed_base = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  p <- ncol(X)
  if (is.null(k)) k <- p
  k <- min(as.integer(k), p)

  num_steps <- max(50L, floor(p / 5))
  lams <- seq(10, 0.001, length.out = num_steps)
  first_zero_step <- rep(NA_integer_, p)
  max_abs_coef <- numeric(p)
  successful_fits <- 0L

  for (step_index in seq_along(lams)) {
    tryCatch({
      if (!is.null(seed_base)) set.seed(as.integer(seed_base + step_index))
      fit <- LiblineaR::LiblineaR(
        data = X, target = Y, type = 5,
        cost = lams[step_index], bias = 1
      )
      beta <- as.numeric(as.matrix(fit$W)[1, seq_len(p)])
      max_abs_coef <- pmax(max_abs_coef, abs(beta))
      newly_zero <- which(abs(beta) <= 1e-8 & is.na(first_zero_step))
      first_zero_step[newly_zero] <- step_index
      successful_fits <- successful_fits + 1L
    }, error = function(e) NULL)
  }

  if (successful_fits == 0L) {
    stop("SVMMA特征筛选阶段的所有L1-SVM拟合均失败。")
  }

  stability_step <- ifelse(is.na(first_zero_step), num_steps + 1L, first_zero_step)
  features_ind <- order(-stability_step, -max_abs_coef, seq_len(p))[seq_len(k)]
  Indexs_mat <- matrix(FALSE, nrow = k, ncol = p)
  for (i in seq_len(k)) Indexs_mat[i, features_ind[seq_len(i)]] <- TRUE
  Indexs_mat
}

# 根据LiblineaR自身的预测结果校正连续判别分数方向，
# 避免旧版只依赖ClassNames顺序而把部分SVM分数方向取反。
train_safe_svm <- function(X, Y, C_val = 1, fit_seed = NULL) {
  X <- as.matrix(X)
  Y <- as.numeric(as.character(Y))
  if (!is.null(fit_seed)) set.seed(as.integer(fit_seed))

  fit <- LiblineaR::LiblineaR(
    data = X, target = Y, type = 3,
    cost = C_val, bias = 1
  )

  p <- ncol(X)
  w_raw <- as.numeric(as.matrix(fit$W)[1, ])
  feature_weights <- w_raw[seq_len(p)]
  intercept <- if (length(w_raw) > p) w_raw[p + 1L] * as.numeric(fit$Bias) else 0
  beta_full <- c(intercept, feature_weights)

  package_prediction <- as.numeric(as.character(predict(fit, X)$predictions))
  raw_score <- as.numeric(cbind(1, X) %*% beta_full)
  direct_prediction <- ifelse(raw_score >= 0, 1, -1)
  direct_agreement <- mean(direct_prediction == package_prediction)
  reversed_agreement <- mean(-direct_prediction == package_prediction)

  if (reversed_agreement > direct_agreement) {
    beta_full <- -beta_full
    direct_agreement <- reversed_agreement
  }
  if (!is.finite(direct_agreement) || direct_agreement < 0.99) {
    stop(sprintf("无法可靠确定LiblineaR判别分数方向（agreement=%.4f）。", direct_agreement))
  }

  beta_full
}

# ==========================================
# 3. 核心计算逻辑 (8 种方法对决)
# ==========================================
run_main <- function(
    rep_idx, X_train, Y_train, X_pre, Y_pre, samplesize,
    datatype, modeltype, C_scale, X_train_raw_full,
    k_fixed_svm, k_fixed_dcsvm, n_folds,
    variance_top_n = 1000L, fold_seed = 1L
) {
  p <- ncol(X_train); C_val <- 1 
  Y_train_num <- as.numeric(as.character(Y_train))
  Y_pre_num <- as.numeric(as.character(Y_pre))
  X_train_raw_full <- as.matrix(X_train_raw_full)
  if (nrow(X_train_raw_full) != samplesize || nrow(X_train) != samplesize) {
    stop("SVMMA内部交叉验证所用的原始训练集样本数不一致。")
  }
  
  h_opt_train <- get_dynamic_h(n = samplesize, C = C_scale)
  
  # === SVM 家族特征排序与基模型训练 ===
  t0_svm_base <- start_method_timer()
  Indexs_svm <- Indexs_fun_lasso(
    X_train, Y_train_num, k = k_fixed_svm,
    seed_base = fold_seed + 100000L
  )
  
  Betas_svm <- matrix(0, nrow = p + 1, ncol = nrow(Indexs_svm))
  for (i in 1:nrow(Indexs_svm)) {
    idx <- Indexs_svm[i, ]
    X_s <- X_train[, idx, drop = FALSE]
    b_vec <- train_safe_svm(
      X_s, Y_train_num, C_val = C_val,
      fit_seed = fold_seed + 200000L + i
    )
    Betas_svm[1, i] <- b_vec[1]; Betas_svm[which(idx) + 1, i] <- b_vec[-1]
  }
  timing_svm_base <- stop_method_timer(t0_svm_base)
  time_svm_base <- timing_svm_base[["elapsed"]]
  time_svm_base_cpu <- timing_svm_base[["cpu"]]
  
  # === DCSVM 家族特征排序与基模型训练 ===
  t0_dc_base <- start_method_timer()
  Indexs_dcsvm <- Indexs_fun_dcsvm(
    X_train, Y_train_num,
    k = k_fixed_dcsvm,
    h_val = h_opt_train
  )
  
  Betas_dcsvm <- matrix(0, nrow = p + 1, ncol = nrow(Indexs_dcsvm))
  loss_train_dcsvm <- numeric(nrow(Indexs_dcsvm))
  for (i in 1:nrow(Indexs_dcsvm)) {
    vars <- which(Indexs_dcsvm[i, ])
    fit_new <- dcsvm::dcsvm(x = X_train[, vars, drop=F], y = Y_train_num, lambda = 0, lam2 = 1, hval = h_opt_train, kern = "epanechnikov")
    Betas_dcsvm[1, i] <- fit_new$b0; Betas_dcsvm[vars + 1, i] <- as.numeric(fit_new$beta)
    train_pred <- as.numeric(X_train[, vars, drop=F] %*% fit_new$beta) + fit_new$b0
    loss_train_dcsvm[i] <- dcsvm_loss_sum(Y_train_num * train_pred, h_opt_train)
  }
  idx_sums <- rowSums(Indexs_dcsvm)
  
  # 1. DCSVMICL & SDCL
  score_dcl <- loss_train_dcsvm + idx_sums * log(samplesize)
  w_dcl_dc <- rep(0, nrow(Indexs_dcsvm)); w_dcl_dc[which.min(score_dcl)] <- 1
  sdcl_scaled <- -score_dcl / 2; w_sdcl_dc <- exp(sdcl_scaled - max(sdcl_scaled)); w_sdcl_dc <- w_sdcl_dc / sum(w_sdcl_dc)
  
  # 2. DCSVMICH & SDCH (高维理论修正)
  score_dch <- loss_train_dcsvm + idx_sums * (log(samplesize)^(3/2))
  w_dch_dc <- rep(0, nrow(Indexs_dcsvm)); w_dch_dc[which.min(score_dch)] <- 1
  sdch_scaled <- -score_dch / samplesize; w_sdch_dc <- exp(sdch_scaled - max(sdch_scaled)); w_sdch_dc <- w_sdch_dc / sum(w_sdch_dc)
  
  w_unif_dc <- rep(1 / nrow(Indexs_dcsvm), nrow(Indexs_dcsvm))
  timing_dc_base <- stop_method_timer(t0_dc_base)
  time_dc_base <- timing_dc_base[["elapsed"]]
  time_dc_base_cpu <- timing_dc_base[["cpu"]]
  
  # === 可编辑折数的交叉验证提取 OOF 预测矩阵 (物理隔离时间计算) ===
  set.seed(as.integer(fold_seed))
  Jn <- as.integer(n_folds)
  folds <- createFolds(as.factor(Y_train), k = Jn, list = FALSE)
  Pred_OOF_svm <- matrix(0, nrow = samplesize, ncol = nrow(Indexs_svm))
  Pred_OOF_dc <- matrix(0, nrow = samplesize, ncol = nrow(Indexs_dcsvm))
  h_oof <- numeric(samplesize)
  
  time_cv_svm <- 0; time_cv_dc <- 0
  time_cv_svm_cpu <- 0; time_cv_dc_cpu <- 0
  
  # 第一阶段：SVMMA每一折都只用该折训练样本重新完成
  # 方差筛选、标准化、L1-SVM特征排序和基模型拟合。
  t0_cv_svm <- start_method_timer()
  for (j in 1:Jn) {
    tr_idx <- (folds != j); val_idx <- (folds == j)

    fold_processed <- variance_prescreen_and_scale(
      X_train_raw = X_train_raw_full[tr_idx, , drop = FALSE],
      X_apply_raw = X_train_raw_full[val_idx, , drop = FALSE],
      variance_top_n = variance_top_n
    )
    X_fold_train <- fold_processed$X_train
    X_fold_val <- fold_processed$X_apply
    Indexs_svm_fold <- Indexs_fun_lasso(
      X_fold_train, Y_train_num[tr_idx],
      k = ncol(Pred_OOF_svm),
      seed_base = fold_seed + 300000L + j * 1000L
    )

    for (i in 1:nrow(Indexs_svm)) {
      vars <- which(Indexs_svm_fold[i, ])
      b_res <- train_safe_svm(
        X_fold_train[, vars, drop = FALSE],
        Y_train_num[tr_idx],
        C_val = C_val,
        fit_seed = fold_seed + 400000L + j * 1000L + i
      )
      Pred_OOF_svm[val_idx, i] <- as.numeric(
        cbind(1, X_fold_val[, vars, drop = FALSE]) %*% b_res
      )
    }
  }
  timing_cv_svm <- stop_method_timer(t0_cv_svm)
  time_cv_svm <- timing_cv_svm[["elapsed"]]
  time_cv_svm_cpu <- timing_cv_svm[["cpu"]]
  
  # 强制垃圾回收，切断内存污染
  gc(verbose = FALSE)
  
  # 第二阶段：一口气跑完纯净的 DCSVM CV
  t0_cv_dc <- start_method_timer()
  for (j in 1:Jn) {
    tr_idx <- (folds != j); val_idx <- (folds == j)
    h_oof[val_idx] <- get_dynamic_h(n = sum(tr_idx), C = C_scale)

    for (i in 1:nrow(Indexs_dcsvm)) {
      vars <- which(Indexs_dcsvm[i, ])
      h_train_cv <- get_dynamic_h(n = sum(tr_idx), C = C_scale)
      fit_cv <- dcsvm::dcsvm(x = X_train[tr_idx, vars, drop=F], y = Y_train_num[tr_idx], lambda = 0, lam2 = 1, hval = h_train_cv, kern = "epanechnikov")
      Pred_OOF_dc[val_idx, i] <- X_train[val_idx, vars, drop=F] %*% as.numeric(fit_cv$beta) + fit_cv$b0
    }
  }
  timing_cv_dc <- stop_method_timer(t0_cv_dc)
  time_cv_dc <- timing_cv_dc[["elapsed"]]
  time_cv_dc_cpu <- timing_cv_dc[["cpu"]]
  
  # === MA 权重求解优化器 ===
  solve_weights <- function(Pred_mat, loss_type, h_vec=NULL) {
    w0 <- rep(1/ncol(Pred_mat), ncol(Pred_mat))
    if (loss_type == "hinge") {
      obj_fn <- function(w) { mean(pmax(0, 1 - Y_train_num * as.numeric(Pred_mat %*% w))) }
    } else {
      obj_fn <- function(w) { dcsvm_loss_mean(Y_train_num * as.numeric(Pred_mat %*% w), h_vec) }
    }
    res <- suppressWarnings(solnp(pars = w0, fun = obj_fn, eqfun = function(w) sum(w) - 1, eqB = 0, LB = rep(0, ncol(Pred_mat)), UB = rep(1, ncol(Pred_mat)), control = list(trace = 0)))
    w_opt <- ifelse(abs(res$pars) < 1e-5, 0, res$pars); return(if(sum(w_opt) > 0) w_opt / sum(w_opt) else w0)
  }
  
  t0_w_svm <- start_method_timer()
  w_ma_svm <- solve_weights(Pred_OOF_svm, "hinge")
  timing_solnp_svm <- stop_method_timer(t0_w_svm)
  time_solnp_svm <- timing_solnp_svm[["elapsed"]]
  time_solnp_svm_cpu <- timing_solnp_svm[["cpu"]]

  t0_w_dc <- start_method_timer()
  w_ma_dc <- solve_weights(Pred_OOF_dc, "smoothed", h_oof)
  timing_solnp_dc <- stop_method_timer(t0_w_dc)
  time_solnp_dc <- timing_solnp_dc[["elapsed"]]
  time_solnp_dc_cpu <- timing_solnp_dc[["cpu"]]
  
  mse_rf <- NA; time_rf <- NA; time_rf_cpu <- NA
  tryCatch({
    t0 <- start_method_timer()
    # 随机森林 (RF): 使用 caret 在训练集上进行 5 折交叉验证寻优 mtry
    mtry_grid <- expand.grid(.mtry = c(max(1, floor(sqrt(p)/2)), floor(sqrt(p)), min(p, floor(sqrt(p)*2))))
    ctrl <- trainControl(method = "cv", number = 5)
    rf_cv <- train(x = X_train, y = as.factor(Y_train_num), method = "rf", 
                   tuneGrid = mtry_grid, trControl = ctrl, ntree = 100)
    # 使用 CV 选出的最佳模型去预测测试集
    pred_rf <- as.numeric(as.character(predict(rf_cv, X_pre)))
    mse_rf <- mean(pred_rf != Y_pre_num)
    timing_rf <- stop_method_timer(t0)
    time_rf <- timing_rf[["elapsed"]]
    time_rf_cpu <- timing_rf[["cpu"]]
  }, error = function(e) {
    mse_rf <<- NA
    time_rf <<- NA
    time_rf_cpu <<- NA
  })
  
  # === 统一评估标尺 (Smoothed Loss) ===
  h_test <- get_dynamic_h(n = samplesize, C = C_scale)
  Eval_Matrix_Smoothed <- function(w, Betas_mat) {
    preds <- as.numeric(cbind(1, X_pre) %*% (Betas_mat %*% w)); res_y <- sign(preds); res_y[res_y == 0] <- 1
    sm_loss <- dcsvm_loss_mean(Y_pre_num * preds, h_test) 
    return(list(MSE = mean(res_y != Y_pre_num), SmoothedLoss = sm_loss))
  }
  get_min_smoothed <- function(Betas_mat) {
    Pred_Test <- cbind(1, X_pre) %*% Betas_mat
    obj_fn <- function(w) { dcsvm_loss_mean(Y_pre_num * as.numeric(Pred_Test %*% w), h_test) }
    res <- suppressWarnings(solnp(pars = rep(1/ncol(Betas_mat), ncol(Betas_mat)), fun = obj_fn, eqfun = function(w) sum(w) - 1, eqB = 0, LB = rep(0, ncol(Betas_mat)), UB = rep(1, ncol(Betas_mat)), control = list(trace = 0)))
    return(max(obj_fn(res$pars), 0.01))
  }
  
  minloss_svm <- max(get_min_smoothed(Betas_svm),0.01)
  minloss_dcsvm <- max(get_min_smoothed(Betas_dcsvm),0.01)
  
  # === 结果记录写入 ===
  df_res <- data.frame(
    repeat_idx = rep_idx, n = samplesize, p = p,
    
    ratio_rf = NA,
    ratio_dch = Eval_Matrix_Smoothed(w_dch_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_dcl = Eval_Matrix_Smoothed(w_dcl_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_sdcl = Eval_Matrix_Smoothed(w_sdcl_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_sdch = Eval_Matrix_Smoothed(w_sdch_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_unif_dc = Eval_Matrix_Smoothed(w_unif_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_ma_dc = Eval_Matrix_Smoothed(w_ma_dc, Betas_dcsvm)$SmoothedLoss / minloss_dcsvm,
    ratio_ma = Eval_Matrix_Smoothed(w_ma_svm, Betas_svm)$SmoothedLoss / minloss_svm,
    
    MSE_rf = mse_rf,
    MSE_dch = Eval_Matrix_Smoothed(w_dch_dc, Betas_dcsvm)$MSE,
    MSE_dcl = Eval_Matrix_Smoothed(w_dcl_dc, Betas_dcsvm)$MSE,
    MSE_sdcl = Eval_Matrix_Smoothed(w_sdcl_dc, Betas_dcsvm)$MSE,
    MSE_sdch = Eval_Matrix_Smoothed(w_sdch_dc, Betas_dcsvm)$MSE,
    MSE_unif_dc = Eval_Matrix_Smoothed(w_unif_dc, Betas_dcsvm)$MSE,
    MSE_ma_dc = Eval_Matrix_Smoothed(w_ma_dc, Betas_dcsvm)$MSE,
    MSE_ma = Eval_Matrix_Smoothed(w_ma_svm, Betas_svm)$MSE,
    
    # Time_*：CPU秒数；Elapsed_*：真实墙钟秒数。
    Time_rf = time_rf_cpu,
    Time_dch = time_dc_base_cpu, Time_dcl = time_dc_base_cpu,
    Time_sdcl = time_dc_base_cpu, Time_sdch = time_dc_base_cpu,
    Time_unif_dc = time_dc_base_cpu,
    Time_ma_dc = time_dc_base_cpu + time_cv_dc_cpu + time_solnp_dc_cpu,
    Time_ma = time_svm_base_cpu + time_cv_svm_cpu + time_solnp_svm_cpu,

    Elapsed_rf = time_rf,
    Elapsed_dch = time_dc_base, Elapsed_dcl = time_dc_base,
    Elapsed_sdcl = time_dc_base, Elapsed_sdch = time_dc_base,
    Elapsed_unif_dc = time_dc_base,
    Elapsed_ma_dc = time_dc_base + time_cv_dc + time_solnp_dc,
    Elapsed_ma = time_svm_base + time_cv_svm + time_solnp_svm,

    # 诊断列：分别保存候选模型拟合、OOF交叉验证和权重求解时间。
    TimePart_svm_base_cpu = time_svm_base_cpu,
    TimePart_svm_cv_cpu = time_cv_svm_cpu,
    TimePart_svm_solver_cpu = time_solnp_svm_cpu,
    TimePart_dc_base_cpu = time_dc_base_cpu,
    TimePart_dc_cv_cpu = time_cv_dc_cpu,
    TimePart_dc_solver_cpu = time_solnp_dc_cpu,
    ElapsedPart_svm_base = time_svm_base,
    ElapsedPart_svm_cv = time_cv_svm,
    ElapsedPart_svm_solver = time_solnp_svm,
    ElapsedPart_dc_base = time_dc_base,
    ElapsedPart_dc_cv = time_cv_dc,
    ElapsedPart_dc_solver = time_solnp_dc,
    
    stringsAsFactors = FALSE
  )
  
  path_rep <- sprintf("../Results/datatype=%s/modeltype=%s/n=%d_p=%d/replication/", datatype, modeltype, samplesize, p)
  if (!dir.exists(path_rep)) dir.create(path_rep, recursive = TRUE)
  write.csv(df_res, sprintf("%srepeat=%d.csv", path_rep, rep_idx), row.names = FALSE)
  
  return(df_res)
}

# ==========================================
# 4. 主程序入口
# ==========================================
if (sys.nframe() == 0) {
  WORK_DIR <- getwd()
  setwd(WORK_DIR)
  
  start_t <- Sys.time()
  set.seed(2026)

  # ==========================================================
  # 可编辑参数区：只需在这里修改实验配置
  # ==========================================================
  DATASET_NAME <- "chowdary"
  C_SCALE <- 1.0
  VARIANCE_TOP_N <- 3000L
  K_FIXED_svm <- 10L
  K_FIXED_dcsvm <- 10L
  N_REPEATS <- 100L
  TRAIN_RATIOS <- c(0.3, 0.4, 0.5, 0.6, 0.7)
  N_FOLDS <- 10L
  detected_cores <- parallel::detectCores()
  if (!is.finite(detected_cores) || detected_cores < 2L) detected_cores <- 2L
  N_WORKERS <- max(1L, min(as.integer(detected_cores) - 1L, 10L))

  if (K_FIXED_svm < 1L || K_FIXED_svm > VARIANCE_TOP_N) {
    stop("K_FIXED_svm必须在1和VARIANCE_TOP_N之间。")
  }
  if (K_FIXED_dcsvm < 1L || K_FIXED_dcsvm > VARIANCE_TOP_N) {
    stop("K_FIXED_dcsvm必须在1和VARIANCE_TOP_N之间。")
  }
  if (N_REPEATS < 1L || N_FOLDS < 2L || N_WORKERS < 1L) {
    stop("N_REPEATS、N_FOLDS或N_WORKERS设置不合法。")
  }

  cat("正在加载真实数据集...\n")
  data(list = DATASET_NAME, package = 'datamicroarray')
  current_data <- get(DATASET_NAME)
  
  X_raw <- as.matrix(current_data$x)
  storage.mode(X_raw) <- "numeric"
  y_vec <- ifelse(current_data$y == levels(as.factor(current_data$y))[1], 1, -1)

  # 每次重复独立划分后，只根据该次训练集筛选方差最大的特征。
  total_n <- nrow(X_raw)
  total_p <- ncol(X_raw)
  analysis_p <- min(VARIANCE_TOP_N, total_p)

  if (K_FIXED_svm > analysis_p || K_FIXED_dcsvm > analysis_p) {
    stop("候选模型个数不能超过当前数据实际保留的特征数。")
  }
  
  cat(sprintf(
    "=> 进入 %d 次重抽样验证... (每次仅用训练集筛选方差前%d特征)\n",
    N_REPEATS, analysis_p
  ))
  
  model_configs <- list()
  for (ratio in TRAIN_RATIOS) {
    n_train <- round(total_n * ratio) 
    model_configs[[length(model_configs) + 1]] <- c(n_train, analysis_p)
  }
  
  cat(sprintf(
    paste0(
      "实验配置完成。总样本: %d, 原始维度: %d, 每次分析维度: %d",
      " | K_svm: %d, K_dcsvm: %d | OOF折数: %d | 带宽系数 C: %.2f\n"
    ),
    total_n, total_p, analysis_p,
    K_FIXED_svm, K_FIXED_dcsvm, N_FOLDS, C_SCALE
  ))
  
  tasks <- list()
  for (conf in model_configs) {
    for (r in 0:(N_REPEATS - 1L)) {
      tasks[[length(tasks) + 1]] <- list(
        samplesize = conf[1], p = conf[2], rep_idx = r,
        datatype = 'realdata', modeltype = 1
      )
    }
  }
  
  cl <- makeCluster(N_WORKERS); registerDoParallel(cl)
  cat(sprintf(
    "启动%d个并行worker！Time_*为CPU秒，Elapsed_*为墙钟秒。工作路径: %s\n",
    N_WORKERS, WORK_DIR
  ))
  
  foreach(
    t = tasks,
    .packages = libs,
    .export = c(
      "WORK_DIR", "C_SCALE", "VARIANCE_TOP_N",
      "K_FIXED_svm", "K_FIXED_dcsvm", "N_FOLDS",
      "variance_prescreen_and_scale", "get_dynamic_h", "L_h_E",
      "dcsvm_loss_sum", "dcsvm_loss_mean", "Indexs_fun_dcsvm",
      "Indexs_fun_lasso", "train_safe_svm", "start_method_timer",
      "stop_method_timer", "run_main"
    )
  ) %dopar% {
    setwd(WORK_DIR)
    set.seed(t$samplesize * 1000 + t$rep_idx)
    
    train_idx <- as.vector(
      createDataPartition(
        as.factor(y_vec),
        p = t$samplesize / nrow(X_raw),
        list = FALSE
      )
    )[seq_len(t$samplesize)]
    test_idx <- setdiff(seq_len(nrow(X_raw)), train_idx)

    X_train_raw <- X_raw[train_idx, , drop = FALSE]
    X_pre_raw <- X_raw[test_idx, , drop = FALSE]
    processed <- variance_prescreen_and_scale(
      X_train_raw = X_train_raw,
      X_apply_raw = X_pre_raw,
      variance_top_n = VARIANCE_TOP_N
    )
    
    run_main(
      t$rep_idx,
      processed$X_train,
      y_vec[train_idx],
      processed$X_apply,
      y_vec[test_idx],
      t$samplesize,
      t$datatype,
      t$modeltype,
      C_scale = C_SCALE,
      X_train_raw_full = X_train_raw,
      k_fixed_svm = K_FIXED_svm,
      k_fixed_dcsvm = K_FIXED_dcsvm,
      n_folds = N_FOLDS,
      variance_top_n = VARIANCE_TOP_N,
      fold_seed = t$samplesize * 1000 + t$rep_idx
    )
  }
  stopCluster(cl)
  
  cat("并行结束，开始聚合数据与绘制核心折线图...\n")
  
  all_files <- list()
  for (conf in model_configs) {
    folder <- sprintf("../Results/datatype=realdata/modeltype=1/n=%d_p=%d/replication/", conf[1], conf[2])
    expected_files <- file.path(
      folder,
      sprintf("repeat=%d.csv", 0:(N_REPEATS - 1L))
    )
    all_files <- c(all_files, expected_files[file.exists(expected_files)])
  }
  if (length(all_files) == 0L) stop("没有找到本次运行生成的重复结果文件。")
  df_all <- do.call(rbind, lapply(all_files, read.csv))
  df_mean <- aggregate(df_all[, !names(df_all) %in% c("repeat_idx", "n")], by = list(n = df_all$n), FUN = mean, na.rm = TRUE)
  
  if (!dir.exists("../figures/Method_Compare/")) dir.create("../figures/Method_Compare/", recursive = TRUE)
  
  # =========================================================
  # 图例排序与后续版本一致：DCSVMICL位于DCSVMICH之前。
  # =========================================================
  methods_all <- c("rf", "dcl", "dch", "sdcl", "sdch", "unif_dc", "ma_dc", "ma")
  method_names <- c("RandomForest", "DCSVMICL", "DCSVMICH", "SDCL", "SDCH", "DCSVM-UNIF", "DCSVMMA", "SVMMA")
  methods_nshl <- methods_all[methods_all != "rf"]
  method_names_nshl <- method_names[methods_all != "rf"]
  
  summary_table <- data.frame(Method = method_names, stringsAsFactors = FALSE)
  sd_table <- data.frame(Method = method_names, stringsAsFactors = FALSE)
  time_table <- data.frame(Method = method_names, stringsAsFactors = FALSE)
  
  for (n_val in sort(unique(df_all$n))) {
    df_n <- df_all[df_all$n == n_val, ]
    er_col <- c(); nshl_col <- c(); time_col <- c(); elapsed_col <- c(); er_sd_col <- c()
    for (m in methods_all) {
      er_mean <- mean(df_n[[paste0("MSE_", m)]], na.rm = TRUE); er_sd <- sd(df_n[[paste0("MSE_", m)]], na.rm = TRUE)
      er_col <- c(er_col, sprintf("%.4f +/- %.4f", er_mean, er_sd))
      er_sd_col <- c(er_sd_col, sprintf("%.4f", er_sd))
      
      if (m != "rf") {
        nshl_mean <- mean(df_n[[paste0("ratio_", m)]], na.rm = TRUE); nshl_sd <- sd(df_n[[paste0("ratio_", m)]], na.rm = TRUE)
        nshl_col <- c(nshl_col, sprintf("%.4f +/- %.4f", nshl_mean, nshl_sd))
      } else { nshl_col <- c(nshl_col, "-") }
      
      t_mean <- mean(df_n[[paste0("Time_", m)]], na.rm = TRUE); t_sd <- sd(df_n[[paste0("Time_", m)]], na.rm = TRUE)
      time_col <- c(time_col, sprintf("%.4f +/- %.4f", t_mean, t_sd))

      elapsed_mean <- mean(df_n[[paste0("Elapsed_", m)]], na.rm = TRUE)
      elapsed_sd <- sd(df_n[[paste0("Elapsed_", m)]], na.rm = TRUE)
      elapsed_col <- c(elapsed_col, sprintf("%.4f +/- %.4f", elapsed_mean, elapsed_sd))
    }
    summary_table[[sprintf("n=%d_ER", n_val)]] <- er_col
    summary_table[[sprintf("n=%d_NSHL", n_val)]] <- nshl_col
    summary_table[[sprintf("n=%d_Time_CPU(S)", n_val)]] <- time_col
    summary_table[[sprintf("n=%d_Elapsed(S)", n_val)]] <- elapsed_col
    sd_table[[sprintf("n=%d_ER_SD", n_val)]] <- er_sd_col
    time_table[[sprintf("n=%d_Time_CPU(S)", n_val)]] <- time_col
    time_table[[sprintf("n=%d_Elapsed(S)", n_val)]] <- elapsed_col
  }
  write.csv(summary_table, "../figures/Method_Compare/Summary_Table_8_Methods_Ordered.csv", row.names = FALSE)
  write.csv(sd_table, "../figures/Method_Compare/ER_SD_Table_8_Methods_Ordered.csv", row.names = FALSE)
  write.csv(time_table, "../figures/Method_Compare/Time_Table_8_Methods_Ordered.csv", row.names = FALSE)
  
  # === 绘图模块 ===
  mse_cols <- paste0("MSE_", methods_all)
  ratio_cols_nshl <- paste0("ratio_", methods_nshl)
  
  df_er_long <- data.frame(
    n = rep(df_mean$n, length(method_names)), ER = unlist(df_mean[mse_cols]),
    Method = factor(rep(method_names, each = nrow(df_mean)), levels = method_names)
  )
  
  df_nhl_long <- data.frame(
    n = rep(df_mean$n, length(method_names_nshl)),
    NHL = unlist(df_mean[ratio_cols_nshl]),
    Method = factor(
      rep(method_names_nshl, each = nrow(df_mean)),
      levels = method_names_nshl
    )
  )
  df_nhl_long <- df_nhl_long[!is.na(df_nhl_long$NHL), ]
  df_nhl_long$Method <- droplevels(df_nhl_long$Method)
  
  # 定义 8 种颜色，严格匹配排列顺序
  color_palette <- c("RandomForest"="#FFD700", 
                     "DCSVMICH"="#FF7F00", "DCSVMICL"="#377EB8", "SDCL"="#4DAF4A", 
                     "SDCH"="#F781BF", "DCSVM-UNIF"="#A65628", "DCSVMMA"="#E41A1C", 
                     "SVMMA"="#984EA3")

  shape_values <- c(
    "RandomForest" = 1, "DCSVMICH" = 3, "DCSVMICL" = 4,
    "SDCL" = 5, "SDCH" = 6, "DCSVM-UNIF" = 7,
    "DCSVMMA" = 8, "SVMMA" = 9
  )
  method_linetypes <- c(
    "RandomForest" = "dotdash",
    "DCSVMICH" = "solid", "DCSVMICL" = "solid",
    "SDCL" = "solid", "SDCH" = "solid",
    "DCSVM-UNIF" = "solid", "DCSVMMA" = "solid",
    "SVMMA" = "dashed"
  )

  method_breaks_er <- levels(droplevels(df_er_long$Method))
  method_breaks_nhl <- levels(droplevels(df_nhl_long$Method))

  custom_theme <- theme_minimal() + theme(
    axis.title = element_text(size = 22),
    axis.text = element_text(size = 22),
    legend.position = c(0.85, 0.75),
    legend.background = element_rect(fill = alpha("white", 0.8), color = NA),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 22, face = "bold"),
    legend.key.width = unit(3, "line")
  )
  
  # === 绘制 ER 对比图 ===
  p_er <- ggplot(df_er_long, aes(x = n, y = ER, color = Method, linetype = Method, shape = Method)) +
    geom_line(linewidth = 1.2) + geom_point(size = 3) +
    scale_color_manual(name = "Method", values = color_palette, breaks = method_breaks_er) +
    scale_linetype_manual(name = "Method", values = method_linetypes, breaks = method_breaks_er) +
    scale_shape_manual(name = "Method", values = shape_values, breaks = method_breaks_er) +
    labs(x = "Training Sample Size (n)", y = "Error Rate (ER)") +
    custom_theme +
    guides(
      color = guide_legend(order = 1),
      linetype = guide_legend(order = 1),
      shape = guide_legend(order = 1)
    )
  ggsave("../figures/Method_Compare/All_8_Weights_ER_Ordered.png", p_er, width = 12, height = 7, dpi=300)
  
  # === 绘制 NSHL 对比图 (剔除单体模型) ===
  p_nhl <- ggplot(df_nhl_long, aes(x = n, y = NHL, color = Method, linetype = Method, shape = Method)) +
    geom_line(linewidth = 1.2) + geom_point(size = 3) +
    scale_color_manual(name = "Method", values = color_palette, breaks = method_breaks_nhl) +
    scale_linetype_manual(name = "Method", values = method_linetypes, breaks = method_breaks_nhl) +
    scale_shape_manual(name = "Method", values = shape_values, breaks = method_breaks_nhl) +
    labs(x = "Training Sample Size (n)", y = "Normalized Smoothed Hinge Loss (NSHL)") +
    custom_theme +
    guides(
      color = guide_legend(order = 1),
      linetype = guide_legend(order = 1),
      shape = guide_legend(order = 1)
    )
  ggsave("../figures/Method_Compare/All_7_Weights_NSHL_Ordered.png", p_nhl, width = 12, height = 7, dpi=300)
  
  cat(sprintf("\n任务完成！总耗时: %.2f 分\n", as.numeric(difftime(Sys.time(), start_t, units = "mins"))))
}

save.image()
