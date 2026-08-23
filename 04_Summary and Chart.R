# ==============================================================================
# 04_汇总表格与ER_NSHL绘图.R
#
# 请先依次完成：
#   01_DCSVMMA_EP_独立计算.R
#   02_DCSVMMA_Gaussian_独立计算.R
#   03_SVMMA_RF_独立计算.R
#
# 本脚本：
# 1. 按核检查共有重复使用了完全相同的数据、种子和 OOF 折。
# 2. 输出 ER/NSHL 性能汇总表。
# 3. 单独输出仅含 SVMMA、DCSVMMA-EP 和 DCSVMMA-Gaussian 的时间表。
# 4. 按核、C、DGP 和 Correct/Misspec 分别绘制 ER 和 NSHL 图。
# ==============================================================================

required_packages <- c("ggplot2")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

RESULT_ROOT <- file.path(getwd(), "Split_Results")

EP_FILE <- file.path(
  RESULT_ROOT,
  "DCSVMMA_EP",
  "DCSVMMA_EP_replications.csv"
)
GAUSSIAN_FILE <- file.path(
  RESULT_ROOT,
  "DCSVMMA_Gaussian",
  "DCSVMMA_Gaussian_replications.csv"
)
SVM_FILE <- file.path(
  RESULT_ROOT,
  "SVMMA_RF",
  "SVMMA_RF_replications.csv"
)

SUMMARY_DIR <- file.path(RESULT_ROOT, "Summary")
TABLE_DIR <- file.path(SUMMARY_DIR, "Tables")
FIGURE_ROOT <- file.path(SUMMARY_DIR, "Figures")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_ROOT, recursive = TRUE, showWarnings = FALSE)

required_files <- c(EP_FILE, GAUSSIAN_FILE, SVM_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "以下结果文件不存在，请先运行三个计算脚本：\n",
    paste(missing_files, collapse = "\n")
  )
}

read_result <- function(path) {
  utils::read.csv(
    path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

ep_results <- read_result(EP_FILE)
gaussian_results <- read_result(GAUSSIAN_FILE)
svm_results <- read_result(SVM_FILE)

required_columns <- c(
  "C", "DGP", "ModelType", "IsMisspec", "n", "p", "K",
  "Replication", "DataSeed", "FoldSeed", "DataSignature",
  "FoldSignature", "Kernel", "MethodKey", "Method", "ER",
  "SmoothedLoss", "MinLoss", "NSHL", "Elapsed_Time", "CPU_Time",
  "Elapsed_Base", "Elapsed_OOF", "Elapsed_Solver",
  "CPU_Base", "CPU_OOF", "CPU_Solver", "Solver", "Fallback"
)

check_columns <- function(df, source_name) {
  missing <- setdiff(required_columns, names(df))
  if (length(missing) > 0L) {
    stop(
      source_name,
      " 缺少以下列：",
      paste(missing, collapse = ", ")
    )
  }
}

check_columns(ep_results, "Epanechnikov result")
check_columns(gaussian_results, "Gaussian result")
check_columns(svm_results, "SVMMA/RF result")

# ==============================================================================
# 1. 按核分别检查并保留可配对的重复
# ==============================================================================

pair_keys <- c(
  "C",
  "DGP",
  "ModelType",
  "n",
  "Replication",
  "DataSeed",
  "FoldSeed"
)

signature_table <- function(df, source_suffix) {
  output <- unique(
    df[, c(pair_keys, "DataSignature", "FoldSignature"), drop = FALSE]
  )
  names(output)[names(output) == "DataSignature"] <-
    paste0("DataSignature_", source_suffix)
  names(output)[names(output) == "FoldSignature"] <-
    paste0("FoldSignature_", source_suffix)
  output
}

sig_ep <- signature_table(ep_results, "EP")
sig_gaussian <- signature_table(gaussian_results, "Gaussian")
sig_svm <- signature_table(svm_results, "SVM")

get_kernel_pairs <- function(
    dc_signatures,
    dc_suffix,
    kernel_label
) {
  paired <- merge(
    dc_signatures,
    sig_svm,
    by = pair_keys,
    all = FALSE
  )

  if (nrow(paired) == 0L) {
    stop(
      kernel_label,
      " 与 SVMMA/RF 没有可以配对的重复。"
    )
  }

  dc_data_column <- paste0("DataSignature_", dc_suffix)
  dc_fold_column <- paste0("FoldSignature_", dc_suffix)

  data_match <-
    paired[[dc_data_column]] == paired$DataSignature_SVM
  fold_match <-
    paired[[dc_fold_column]] == paired$FoldSignature_SVM

  if (any(!data_match)) {
    stop(
      "检测到 ",
      kernel_label,
      " 与 SVMMA/RF 使用的训练数据不一致。"
    )
  }
  if (any(!fold_match)) {
    stop(
      "检测到 ",
      kernel_label,
      " 与 SVMMA/RF 使用的 OOF 折划分不一致。"
    )
  }

  unique(paired[, pair_keys, drop = FALSE])
}

ep_pairs <- get_kernel_pairs(
  sig_ep,
  dc_suffix = "EP",
  kernel_label = "Epanechnikov"
)
gaussian_pairs <- get_kernel_pairs(
  sig_gaussian,
  dc_suffix = "Gaussian",
  kernel_label = "Gaussian"
)

filter_to_pairs <- function(df, paired_keys) {
  filtered <- merge(
    df,
    paired_keys,
    by = pair_keys,
    all = FALSE,
    sort = FALSE
  )
  filtered[, names(df), drop = FALSE]
}

# 每个核只与 SVMMA/RF 的相同重复比较。
# 因此 Gaussian 即使只运行较少次数，也能单独计算均值和标准差。
ep_performance <- filter_to_pairs(ep_results, ep_pairs)
gaussian_performance <- filter_to_pairs(
  gaussian_results,
  gaussian_pairs
)
svm_ep_performance <- filter_to_pairs(
  svm_results[svm_results$Kernel == "Epanechnikov", , drop = FALSE],
  ep_pairs
)
svm_gaussian_performance <- filter_to_pairs(
  svm_results[svm_results$Kernel == "Gaussian", , drop = FALSE],
  gaussian_pairs
)
svm_performance <- rbind(
  svm_ep_performance,
  svm_gaussian_performance
)

cat(
  "配对检查通过：每个核内部的数据、种子和 OOF 折完全一致。\n"
)
cat(
  "Epanechnikov 配对记录数：", nrow(ep_pairs),
  "；Gaussian 配对记录数：", nrow(gaussian_pairs),
  "。\n",
  sep = ""
)

# ==============================================================================
# 2. 合并结果，构造 SVMMA1
# ==============================================================================

common_columns <- required_columns

ep_common <- ep_performance[, common_columns, drop = FALSE]
gaussian_common <- gaussian_performance[
  , common_columns, drop = FALSE
]
svm_common <- svm_performance[, common_columns, drop = FALSE]

all_base_results <- rbind(
  ep_common,
  gaussian_common,
  svm_common
)

# SVMMA1 使用 SVMMA 的预测损失，但以对应核的 DCSVM 候选模型
# 测试最优损失为分母，与原模拟代码的 ratio_ma1 一致。
join_keys <- c(
  "C",
  "DGP",
  "ModelType",
  "IsMisspec",
  "n",
  "p",
  "Replication",
  "DataSeed",
  "FoldSeed",
  "Kernel"
)

svmma_for_join <- svm_performance[
  svm_performance$MethodKey == "svmma",
  ,
  drop = FALSE
]
dc_denom <- rbind(
  ep_performance[
    ep_performance$MethodKey == "dcsvmma",
    ,
    drop = FALSE
  ],
  gaussian_performance[
    gaussian_performance$MethodKey == "dcsvmma",
    ,
    drop = FALSE
  ]
)

dc_denom <- unique(
  dc_denom[, c(join_keys, "MinLoss"), drop = FALSE]
)
names(dc_denom)[names(dc_denom) == "MinLoss"] <- "DCSVM_MinLoss"

svmma1_joined <- merge(
  svmma_for_join,
  dc_denom,
  by = join_keys,
  all = FALSE
)

if (nrow(svmma1_joined) != nrow(svmma_for_join)) {
  stop("SVMMA1 分母无法与所有 SVMMA 重复一一匹配。")
}

svmma1_rows <- svmma1_joined[, common_columns, drop = FALSE]
svmma1_rows$MethodKey <- "svmma1"
svmma1_rows$Method <- "SVMMA1"
svmma1_rows$ER <- NA_real_
svmma1_rows$MinLoss <- svmma1_joined$DCSVM_MinLoss
svmma1_rows$NSHL <-
  svmma1_rows$SmoothedLoss /
  svmma1_rows$MinLoss
svmma1_rows$Elapsed_Time <- NA_real_
svmma1_rows$CPU_Time <- NA_real_
svmma1_rows$Elapsed_Base <- NA_real_
svmma1_rows$Elapsed_OOF <- NA_real_
svmma1_rows$Elapsed_Solver <- NA_real_
svmma1_rows$CPU_Base <- NA_real_
svmma1_rows$CPU_OOF <- NA_real_
svmma1_rows$CPU_Solver <- NA_real_
svmma1_rows$Solver <- NA_character_
svmma1_rows$Fallback <- NA

all_results <- rbind(all_base_results, svmma1_rows)

utils::write.csv(
  all_results,
  file.path(TABLE_DIR, "All_Paired_Replication_Results.csv"),
  row.names = FALSE
)

# ==============================================================================
# 3. ER/NSHL 综合汇总表
# ==============================================================================

mean_safe <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

sd_safe <- function(x) {
  valid <- x[is.finite(x)]
  if (length(valid) <= 1L) NA_real_ else stats::sd(valid)
}

summary_group_columns <- c(
  "Kernel",
  "C",
  "DGP",
  "ModelType",
  "n",
  "MethodKey",
  "Method"
)

summary_key <- do.call(
  interaction,
  c(
    all_results[, summary_group_columns, drop = FALSE],
    list(drop = TRUE, lex.order = TRUE)
  )
)

performance_summary_list <- lapply(
  split(all_results, summary_key),
  function(group_df) {
    data.frame(
      Kernel = group_df$Kernel[1],
      C = group_df$C[1],
      DGP = group_df$DGP[1],
      ModelType = group_df$ModelType[1],
      n = group_df$n[1],
      MethodKey = group_df$MethodKey[1],
      Method = group_df$Method[1],
      Successful_Repeats = length(unique(group_df$Replication)),
      Mean_ER = mean_safe(group_df$ER),
      SD_ER = sd_safe(group_df$ER),
      Mean_NSHL = mean_safe(group_df$NSHL),
      SD_NSHL = sd_safe(group_df$NSHL),
      stringsAsFactors = FALSE
    )
  }
)

performance_summary <- do.call(rbind, performance_summary_list)
row.names(performance_summary) <- NULL

method_order <- c(
  "rf",
  "dcl",
  "dch",
  "sdcl",
  "sdch",
  "unif",
  "dcsvmma",
  "svmma",
  "svmma1"
)

performance_summary$MethodOrder <- match(
  performance_summary$MethodKey,
  method_order
)
performance_summary$KernelOrder <- match(
  performance_summary$Kernel,
  c("Epanechnikov", "Gaussian")
)

performance_summary <- performance_summary[
  order(
    performance_summary$KernelOrder,
    performance_summary$C,
    performance_summary$DGP,
    performance_summary$ModelType,
    performance_summary$n,
    performance_summary$MethodOrder
  ),
  ,
  drop = FALSE
]

utils::write.csv(
  performance_summary,
  file.path(TABLE_DIR, "Performance_Summary_ER_NSHL_Long.csv"),
  row.names = FALSE
)

# ==============================================================================
# 4. 只汇总三个目标方法的时间表
# ==============================================================================

runtime_ep <- ep_results[
  ep_results$MethodKey == "dcsvmma",
  ,
  drop = FALSE
]
runtime_ep$RuntimeMethod <- "DCSVMMA-EP"

runtime_gaussian <- gaussian_results[
  gaussian_results$MethodKey == "dcsvmma",
  ,
  drop = FALSE
]
runtime_gaussian$RuntimeMethod <- "DCSVMMA-Gaussian"

# SVMMA 时间与评价核无关。只取 Epanechnikov 评价行，
# 保留三个 C 的配对行，避免重复计数 Gaussian 评价行。
runtime_svm <- svm_results[
  svm_results$MethodKey == "svmma" &
    svm_results$Kernel == "Epanechnikov",
  ,
  drop = FALSE
]
runtime_svm$RuntimeMethod <- "SVMMA"

runtime_columns <- c(required_columns, "RuntimeMethod")
runtime_rows <- rbind(
  runtime_ep[, runtime_columns, drop = FALSE],
  runtime_gaussian[, runtime_columns, drop = FALSE],
  runtime_svm[, runtime_columns, drop = FALSE]
)

runtime_group_columns <- c(
    "C",
    "DGP",
    "ModelType",
    "n",
    "RuntimeMethod"
  )
runtime_key <- do.call(
  interaction,
  c(
    runtime_rows[, runtime_group_columns, drop = FALSE],
    list(drop = TRUE, lex.order = TRUE)
  )
)

runtime_summary_list <- lapply(
  split(runtime_rows, runtime_key),
  function(group_df) {
    data.frame(
      C = group_df$C[1],
      DGP = group_df$DGP[1],
      ModelType = group_df$ModelType[1],
      n = group_df$n[1],
      Method = group_df$RuntimeMethod[1],
      Successful_Repeats = length(unique(group_df$Replication)),
      Mean_Elapsed_Sec = mean_safe(group_df$Elapsed_Time),
      SD_Elapsed_Sec = sd_safe(group_df$Elapsed_Time),
      Median_Elapsed_Sec = stats::median(
        group_df$Elapsed_Time,
        na.rm = TRUE
      ),
      Mean_CPU_Sec = mean_safe(group_df$CPU_Time),
      SD_CPU_Sec = sd_safe(group_df$CPU_Time),
      Mean_Elapsed_Base = mean_safe(group_df$Elapsed_Base),
      Mean_Elapsed_OOF = mean_safe(group_df$Elapsed_OOF),
      Mean_Elapsed_Solver = mean_safe(group_df$Elapsed_Solver),
      stringsAsFactors = FALSE
    )
  }
)

runtime_summary <- do.call(rbind, runtime_summary_list)
row.names(runtime_summary) <- NULL
runtime_summary$MethodOrder <- match(
  runtime_summary$Method,
  c("DCSVMMA-EP", "DCSVMMA-Gaussian", "SVMMA")
)
runtime_summary <- runtime_summary[
  order(
    runtime_summary$C,
    runtime_summary$DGP,
    runtime_summary$ModelType,
    runtime_summary$n,
    runtime_summary$MethodOrder
  ),
  ,
  drop = FALSE
]

utils::write.csv(
  runtime_summary,
  file.path(TABLE_DIR, "Runtime_Summary_Only_3_Methods.csv"),
  row.names = FALSE
)

# ==============================================================================
# 5. 生成与原模拟风格一致的分场景表格和 ER/NSHL 图
# ==============================================================================

plot_method_order_er <- c(
  "RandomForest",
  "DCSVMICL",
  "DCSVMICH",
  "SDCL",
  "SDCH",
  "DCSVM-UNIF",
  "DCSVMMA",
  "SVMMA"
)
plot_method_order_nshl <- c(
  "DCSVMICL",
  "DCSVMICH",
  "SDCL",
  "SDCH",
  "DCSVM-UNIF",
  "DCSVMMA",
  "SVMMA",
  "SVMMA1"
)

color_palette <- c(
  "RandomForest" = "#FFD700",
  "DCSVMICH" = "#FF7F00",
  "DCSVMICL" = "#377EB8",
  "SDCL" = "#4DAF4A",
  "SDCH" = "#F781BF",
  "DCSVM-UNIF" = "#A65628",
  "DCSVMMA" = "#E41A1C",
  "SVMMA" = "#984EA3",
  "SVMMA1" = "#00BFC4"
)

shape_values <- c(
  "RandomForest" = 1,
  "DCSVMICH" = 3,
  "DCSVMICL" = 4,
  "SDCL" = 5,
  "SDCH" = 6,
  "DCSVM-UNIF" = 7,
  "DCSVMMA" = 8,
  "SVMMA" = 9,
  "SVMMA1" = 0
)

method_linetypes <- c(
  "RandomForest" = "dotdash",
  "DCSVMICH" = "solid",
  "DCSVMICL" = "solid",
  "SDCL" = "solid",
  "SDCH" = "solid",
  "DCSVM-UNIF" = "solid",
  "DCSVMMA" = "solid",
  "SVMMA" = "dashed",
  "SVMMA1" = "dotted"
)

custom_theme <- ggplot2::theme_minimal() +
  ggplot2::theme(
    axis.title = ggplot2::element_text(size = 22),
    axis.text = ggplot2::element_text(size = 22),
    legend.position = c(0.85, 0.75),
    legend.background = ggplot2::element_rect(
      fill = ggplot2::alpha("white", 0.8),
      color = NA
    ),
    legend.text = ggplot2::element_text(size = 16),
    legend.title = ggplot2::element_text(size = 22, face = "bold"),
    legend.key.width = grid::unit(3, "line")
  )

kernel_values <- c("Epanechnikov", "Gaussian")
C_values <- sort(unique(performance_summary$C))
DGP_values <- unique(performance_summary$DGP)
model_type_values <- unique(performance_summary$ModelType)

format_mean_sd <- function(mean_value, sd_value) {
  if (!is.finite(mean_value)) {
    return("-")
  }
  if (!is.finite(sd_value)) {
    return(sprintf("%.4f", mean_value))
  }
  sprintf("%.4f +/- %.4f", mean_value, sd_value)
}

for (kernel_label in kernel_values) {
  for (C_value in C_values) {
    C_folder <- paste0(
      "C=",
      format(C_value, trim = TRUE, scientific = FALSE)
    )
    figure_dir <- file.path(
      FIGURE_ROOT,
      kernel_label,
      C_folder
    )
    dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

    for (DGP_value in DGP_values) {
      for (model_type_value in model_type_values) {
        scenario_summary <- performance_summary[
          performance_summary$Kernel == kernel_label &
            performance_summary$C == C_value &
            performance_summary$DGP == DGP_value &
            performance_summary$ModelType == model_type_value,
          ,
          drop = FALSE
        ]

        if (nrow(scenario_summary) == 0L) {
          next
        }

        # 分场景宽表：每个单元格为 mean +/- sd。
        table_methods <- unique(c(
          plot_method_order_er,
          plot_method_order_nshl
        ))
        scenario_table <- data.frame(
          Method = table_methods,
          stringsAsFactors = FALSE
        )

        for (n_value in sort(unique(scenario_summary$n))) {
          er_values <- vapply(
            table_methods,
            function(method_name) {
              row <- scenario_summary[
                scenario_summary$n == n_value &
                  scenario_summary$Method == method_name,
                ,
                drop = FALSE
              ]
              if (nrow(row) == 0L) return("-")
              format_mean_sd(row$Mean_ER[1], row$SD_ER[1])
            },
            character(1)
          )

          nshl_values <- vapply(
            table_methods,
            function(method_name) {
              row <- scenario_summary[
                scenario_summary$n == n_value &
                  scenario_summary$Method == method_name,
                ,
                drop = FALSE
              ]
              if (nrow(row) == 0L) return("-")
              format_mean_sd(row$Mean_NSHL[1], row$SD_NSHL[1])
            },
            character(1)
          )

          scenario_table[[paste0("n=", n_value, "_ER")]] <- er_values
          scenario_table[[paste0("n=", n_value, "_NSHL")]] <- nshl_values
        }

        utils::write.csv(
          scenario_table,
          file.path(
            figure_dir,
            paste0(
              "Summary_",
              DGP_value,
              "_",
              model_type_value,
              ".csv"
            )
          ),
          row.names = FALSE
        )

        # ER 图。
        er_data <- scenario_summary[
          scenario_summary$Method %in% plot_method_order_er &
            is.finite(scenario_summary$Mean_ER),
          ,
          drop = FALSE
        ]
        er_data$Method <- factor(
          er_data$Method,
          levels = plot_method_order_er
        )

        p_er <- ggplot2::ggplot(
          er_data,
          ggplot2::aes(
            x = n,
            y = Mean_ER,
            color = Method,
            linetype = Method,
            shape = Method
          )
        ) +
          ggplot2::geom_line(linewidth = 1.2) +
          ggplot2::geom_point(size = 3) +
          ggplot2::scale_color_manual(
            name = "Method",
            values = color_palette,
            breaks = plot_method_order_er
          ) +
          ggplot2::scale_linetype_manual(
            name = "Method",
            values = method_linetypes,
            breaks = plot_method_order_er
          ) +
          ggplot2::scale_shape_manual(
            name = "Method",
            values = shape_values,
            breaks = plot_method_order_er
          ) +
          ggplot2::labs(
            x = "Training Sample Size (n)",
            y = "Error Rate (ER)"
          ) +
          custom_theme +
          ggplot2::guides(
            color = ggplot2::guide_legend(order = 1),
            linetype = ggplot2::guide_legend(order = 1),
            shape = ggplot2::guide_legend(order = 1)
          )

        ggplot2::ggsave(
          filename = file.path(
            figure_dir,
            paste0(
              "Weights_ER_",
              DGP_value,
              "_",
              model_type_value,
              "_8_Methods.png"
            )
          ),
          plot = p_er,
          width = 12,
          height = 7,
          dpi = 300
        )

        # NSHL 图。
        nshl_data <- scenario_summary[
          scenario_summary$Method %in% plot_method_order_nshl &
            is.finite(scenario_summary$Mean_NSHL),
          ,
          drop = FALSE
        ]
        nshl_data$Method <- factor(
          nshl_data$Method,
          levels = plot_method_order_nshl
        )

        p_nshl <- ggplot2::ggplot(
          nshl_data,
          ggplot2::aes(
            x = n,
            y = Mean_NSHL,
            color = Method,
            linetype = Method,
            shape = Method
          )
        ) +
          ggplot2::geom_line(linewidth = 1.2) +
          ggplot2::geom_point(size = 3) +
          ggplot2::scale_color_manual(
            name = "Method",
            values = color_palette,
            breaks = plot_method_order_nshl
          ) +
          ggplot2::scale_linetype_manual(
            name = "Method",
            values = method_linetypes,
            breaks = plot_method_order_nshl
          ) +
          ggplot2::scale_shape_manual(
            name = "Method",
            values = shape_values,
            breaks = plot_method_order_nshl
          ) +
          ggplot2::labs(
            x = "Training Sample Size (n)",
            y = "Normalized Smoothed Hinge Loss (NSHL)"
          ) +
          custom_theme +
          ggplot2::guides(
            color = ggplot2::guide_legend(order = 1),
            linetype = ggplot2::guide_legend(order = 1),
            shape = ggplot2::guide_legend(order = 1)
          )

        ggplot2::ggsave(
          filename = file.path(
            figure_dir,
            paste0(
              "Weights_NSHL_",
              DGP_value,
              "_",
              model_type_value,
              "_8_Methods.png"
            )
          ),
          plot = p_nshl,
          width = 12,
          height = 7,
          dpi = 300
        )
      }
    }
  }
}

cat("\n汇总与绘图完成。\n")
cat(
  "Performance table: ",
  file.path(TABLE_DIR, "Performance_Summary_ER_NSHL_Long.csv"),
  "\n",
  sep = ""
)
cat(
  "Runtime table: ",
  file.path(TABLE_DIR, "Runtime_Summary_Only_3_Methods.csv"),
  "\n",
  sep = ""
)
cat("Figures: ", FIGURE_ROOT, "\n", sep = "")
