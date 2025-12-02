# ==============================
# Bayesian AD model in R (brms)
# ==============================

# ---- 0. Packages ----
# install.packages(c("tidyverse", "brms"))  # run once
library(tidyverse)
library(brms)

options(mc.cores = parallel::detectCores())
set.seed(42)

# ---- 1. Load data ----
# CHANGE THIS PATH to your actual CSV
data_path <- "data/brain_features.csv"

df <- read_csv(data_path)

# Expect a column "label" with 4 classes
df <- df %>%
  mutate(
    label = factor(
      label,
      levels = c("No Impairment",
                 "Very Mild Impairment",
                 "Mild Impairment",
                 "Moderate Impairment")
    ),
    # Binary version: AD vs No AD
    ad_binary = if_else(
      label %in% c("Mild Impairment", "Moderate Impairment"),
      1L, 0L
    )
  )

# ---- 2. Train / Test split ----
set.seed(42)
N <- nrow(df)
train_idx <- sample(seq_len(N), size = floor(0.8 * N))

train <- df[train_idx, ]
test  <- df[-train_idx, ]

cat("Train rows:", nrow(train), " Test rows:", nrow(test), "\n")

# ---- 3. Define predictors automatically ----
# Use all numeric columns except the targets
num_cols <- train %>%
  select(where(is.numeric)) %>%
  colnames()

num_cols <- setdiff(num_cols, c("ad_binary"))  # drop target

cat("Numeric predictors:\n")
print(num_cols)

# build formula: ad_binary ~ x1 + x2 + ...
formula_str_bin <- paste("ad_binary ~", paste(num_cols, collapse = " + "))
formula_bin <- bf(as.formula(formula_str_bin))

# For ordinal 4-class outcome, treat label as ordered factor
train <- train %>%
  mutate(label_ord = ordered(label, levels = c(
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment"
  )))

test <- test %>%
  mutate(label_ord = ordered(label, levels = levels(train$label_ord)))

formula_str_ord <- paste("label_ord ~", paste(num_cols, collapse = " + "))
formula_ord <- bf(as.formula(formula_str_ord))

# ---- 4. Priors ----
priors_bin <- c(
  prior(student_t(3, 0, 2.5), class = "Intercept"),
  prior(normal(0, 1), class = "b")
)

priors_ord <- c(
  prior(student_t(3, 0, 5), class = "Intercept"),
  prior(normal(0, 1), class = "b")
)

# ---- 5. Fit Bayesian logistic (binary AD vs non-AD) ----
fit_bin <- brm(
  formula = formula_bin,
  data    = train,
  family  = bernoulli(link = "logit"),
  prior   = priors_bin,
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  seed    = 42
)

print(summary(fit_bin))
plot(fit_bin)
pp_check(fit_bin)

# ---- 6. Evaluate binary model on test set ----
bin_pred <- fitted(fit_bin, newdata = test, summary = TRUE)[, "Estimate"]
test$bin_prob  <- bin_pred
test$bin_pred  <- if_else(test$bin_prob > 0.5, 1L, 0L)

bin_acc <- mean(test$bin_pred == test$ad_binary)
cat("Binary Bayesian model test accuracy:", round(bin_acc, 3), "\n")

# ---- 7. Fit Bayesian ordinal model (4-stage AD) ----
fit_ord <- brm(
  formula = formula_ord,
  data    = train,
  family  = cumulative("logit"),
  prior   = priors_ord,
  chains  = 4,
  iter    = 2000,
  warmup  = 1000,
  seed    = 42
)

print(summary(fit_ord))
plot(fit_ord)
pp_check(fit_ord)

# ---- 8. Evaluate ordinal model on test set ----
# Get predicted class (MAP) on test
ord_pred <- posterior_predict(fit_ord, newdata = test, summary = TRUE)
# posterior_predict (summary=TRUE) returns most likely category index per row
# map that back to factor levels:
pred_idx <- as.integer(ord_pred)
test$ord_pred_label <- factor(
  levels(train$label_ord)[pred_idx],
  levels = levels(train$label_ord)
)

ord_acc <- mean(test$ord_pred_label == test$label_ord)
cat("Ordinal Bayesian model test accuracy:", round(ord_acc, 3), "\n")

# ---- 9. Save models ----
dir.create("MLModels", showWarnings = FALSE)
saveRDS(fit_bin, file = "MLModels/bayes_logistic_AD_vs_control.rds")
saveRDS(fit_ord, file = "MLModels/bayes_ordinal_4class.rds")

cat("Saved models to MLModels/.\n")
