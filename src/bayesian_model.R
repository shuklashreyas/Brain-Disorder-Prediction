library(tidyverse)
library(caret)
library(brms)
library(pROC)

options(mc.cores = parallel::detectCores())

set.seed(123)

data_path <- "/Users/shreyas/Desktop/BrainScanning/bayesiandata/cnn_probs_for_bayes.csv"

df <- readr::read_csv(data_path)

df <- df %>%
  mutate(
    label_ord = factor(
      true_label,
      levels = c(
        "No Impairment",
        "Very Mild Impairment",
        "Mild Impairment",
        "Moderate Impairment"
      ),
      ordered = TRUE
    ),
    ad_binary = if_else(
      true_label %in% c("Mild Impairment", "Moderate Impairment"),
      1L, 0L
    )
  )



train_idx <- createDataPartition(df$ad_binary, p = 0.8, list = FALSE)

train <- df[train_idx, ]
test  <- df[-train_idx, ]

cat("Train rows:", nrow(train), " Test rows:", nrow(test), "\n")

num_predictors <- c(
  "p_mild_impairment",
  "p_moderate_impairment",
  "p_no_impairment",
  "p_very_mild_impairment"
)

cat("Numeric predictors used:\n")
print(num_predictors)

cat("\n==== Fitting Binary Logistic Model ====\n")

binary_formula <- bf(
  ad_binary ~ p_mild_impairment +
    p_moderate_impairment +
    p_no_impairment +
    p_very_mild_impairment
)

binary_fit <- brm(
  formula = binary_formula,
  data = train,
  family = bernoulli(link = "logit"),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 123,
  control = list(adapt_delta = 0.95)
)

print(summary(binary_fit))


pp_check(binary_fit, type = "dens_overlay")


bin_epred <- posterior_epred(binary_fit, newdata = test)
bin_prob_ad <- colMeans(bin_epred)


test$binary_true_label <- factor(
  if_else(test$ad_binary == 1L, "AD", "NoAD"),
  levels = c("NoAD", "AD")
)

test$binary_pred_label <- factor(
  if_else(bin_prob_ad > 0.5, "AD", "NoAD"),
  levels = c("NoAD", "AD")
)

binary_cm <- confusionMatrix(
  test$binary_pred_label,
  test$binary_true_label
)

cat("\nBinary Bayesian model test accuracy:",
    round(binary_cm$overall["Accuracy"], 3), "\n\n")

cat("==== Additional Metrics ====\n")
cat("Binary Model Confusion Matrix:\n")
print(binary_cm)

# AUC
bin_roc <- roc(response = test$ad_binary, predictor = bin_prob_ad)
cat("\nBinary Model AUC:", round(as.numeric(auc(bin_roc)), 3), "\n\n")



cat("==== Fitting Ordinal Model ====\n")

ordinal_formula <- bf(
  label_ord ~ p_mild_impairment +
    p_moderate_impairment +
    p_no_impairment +
    p_very_mild_impairment
)

ordinal_fit <- brm(
  formula = ordinal_formula,
  data = train,
  family = cumulative(link = "logit"),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 123,
  control = list(adapt_delta = 0.95)
)

print(summary(ordinal_fit))


pp_check(ordinal_fit, type = "dens_overlay")


ord_pp <- posterior_predict(ordinal_fit, newdata = test)

majority_vote <- function(x) {
  tab <- table(x)
  as.integer(names(which.max(tab)))
}

ord_idx <- apply(ord_pp, 2, majority_vote)

ord_levels <- levels(train$label_ord)

test$ord_pred_label <- factor(
  ord_levels[ord_idx],
  levels = ord_levels,
  ordered = TRUE
)

ordinal_cm <- confusionMatrix(
  factor(test$ord_pred_label, ordered = FALSE),
  factor(test$label_ord, ordered = FALSE)
)

cat("Ordinal Bayesian model test accuracy:",
    round(ordinal_cm$overall["Accuracy"], 3), "\n\n")

cat("Ordinal Model Confusion Matrix:\n")
print(ordinal_cm)

cat("\n==== Analysis Complete ====\n")
