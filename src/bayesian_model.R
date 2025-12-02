# Bayesian Logistic Regression for Alzheimer's MRI Classification
# Uses MCMC (Metropolis-Hastings) for posterior inference
# Provides interpretability through posterior distributions and feature importance

<<<<<<< HEAD
library(MASS)
library(ggplot2)
library(jsonlite)
library(dplyr)

`%+%` <- function(a, b) paste0(a, b)

log_posterior <- function(beta, X, y, sigma2 = 10) {
  Xbeta <- X %*% beta
  Xbeta <- pmax(pmin(Xbeta, 50), -50)
  
  p <- 1 / (1 + exp(-Xbeta))
  
  p <- pmax(p, 1e-10)
  p <- pmin(p, 1 - 1e-10)
  log_likelihood <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  if (is.na(log_likelihood) || is.infinite(log_likelihood)) {
    return(-Inf)
  }
  
  log_prior <- -sum(beta^2) / (2 * sigma2)
  
  result <- log_likelihood + log_prior
  
  if (is.na(result) || is.infinite(result)) {
    return(-Inf)
  }
  
  return(result)
}

mcmc_sampler <- function(X, y, n_iter = 10000, burn_in = 2000, 
                         proposal_sd = NULL, sigma2 = 10) {
  p <- ncol(X)
  beta <- rep(0, p)
  samples <- matrix(0, n_iter, p)
  acceptances <- 0
  
  if (is.null(proposal_sd)) {
    proposal_sd <- 0.05 / sqrt(p)
    proposal_sd <- max(0.01, min(0.1, proposal_sd))
  }
  
  cat("Running MCMC sampler...\n")
  cat(sprintf("Iterations: %d (burn-in: %d)\n", n_iter, burn_in))
  cat(sprintf("Proposal SD: %.4f\n", proposal_sd))
  
  for (i in 1:n_iter) {
    proposal <- beta + rnorm(p, 0, proposal_sd)
    
    log_alpha <- log_posterior(proposal, X, y, sigma2) - 
                 log_posterior(beta, X, y, sigma2)
    
    if (is.na(log_alpha) || is.infinite(log_alpha)) {
      if (is.infinite(log_alpha) && log_alpha > 0) {
        beta <- proposal
        acceptances <- acceptances + 1
      }
    } else {
      if (log(runif(1)) < log_alpha) {
        beta <- proposal
        acceptances <- acceptances + 1
      }
    }
    
    samples[i, ] <- beta
    
    if (i %% 1000 == 0) {
      cat(sprintf("Iteration %d/%d (Acceptance rate: %.2f%%)\n", 
                  i, n_iter, 100 * acceptances / i))
    }
  }
  
  if (burn_in >= n_iter) {
    warning("Burn-in period is >= total iterations. Using all samples.")
    post_samples <- samples
  } else {
    post_samples <- samples[(burn_in + 1):n_iter, ]
  }
  
  posterior_mean <- colMeans(post_samples)
  posterior_sd <- apply(post_samples, 2, sd)
  posterior_ci_lower <- apply(post_samples, 2, quantile, 0.025)
  posterior_ci_upper <- apply(post_samples, 2, quantile, 0.975)
  
  acceptance_rate <- acceptances / n_iter
  
  cat(sprintf("\nMCMC Complete!\n"))
  cat(sprintf("Final acceptance rate: %.2f%%\n", 100 * acceptance_rate))
  
  return(list(
    samples = post_samples,
    posterior_mean = posterior_mean,
    posterior_sd = posterior_sd,
    posterior_ci_lower = posterior_ci_lower,
    posterior_ci_upper = posterior_ci_upper,
    acceptance_rate = acceptance_rate
  ))
}

predict_bayesian <- function(X_test, posterior_mean) {
  Xbeta <- X_test %*% posterior_mean
  Xbeta <- pmax(pmin(Xbeta, 50), -50)
  return(1 / (1 + exp(-Xbeta)))
}
=======
options(mc.cores = parallel::detectCores())
set.seed(123)

data_path  <- "/Users/shreyas/Desktop/BrainScanning/bayesiandata/cnn_probs_for_bayes.csv"
model_dir  <- "/Users/shreyas/Desktop/BrainScanning/MLModels/bayesian"

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

bin_epred   <- posterior_epred(binary_fit, newdata = test)
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

ord_idx    <- apply(ord_pp, 2, majority_vote)
ord_levels <- levels(train$label_ord)
>>>>>>> bc00809 (feat : bayesian model trained suneetpathangay@gmail.com)

load_roi_features <- function(json_path, binary = TRUE) {
  cat("Loading ROI features from", json_path, "\n")
  
  data <- fromJSON(json_path)
  df <- as.data.frame(data)
  
  if (binary) {
    y <- df$binary_label
    cat("Using binary classification: NonDemented (0) vs Demented (1)\n")
  } else {
    y <- df$label
    cat("Using multi-class classification\n")
  }
  
  exclude_cols <- c("label", "binary_label", "image_path", "class_name")
  feature_cols <- setdiff(names(df), exclude_cols)
  
  X <- as.matrix(df[, feature_cols])
  
  if (any(is.na(X)) || any(is.infinite(X))) {
    cat("Warning: Found NA or Inf values in features. Replacing with 0.\n")
    X[is.na(X) | is.infinite(X)] <- 0
  }
  
  X <- cbind(1, X)
  
  X_scaled <- scale(X[, -1])
  if (any(is.na(X_scaled))) {
    X_scaled[is.na(X_scaled)] <- 0
    cat("Warning: Some features have zero variance. Keeping them unstandardized.\n")
  }
  X[, -1] <- X_scaled
  
  cat(sprintf("Loaded %d samples with %d features\n", nrow(X), length(feature_cols)))
  cat(sprintf("Class distribution: %s\n", 
              paste(table(y), collapse = ", ")))
  
  return(list(
    X = X,
    y = y,
    feature_names = c("intercept", feature_cols),
    metadata = df[, exclude_cols]
  ))
}

<<<<<<< HEAD
plot_posterior_distributions <- function(mcmc_result, feature_names, 
                                         output_path = "results/bayesian_posteriors.png") {
  samples <- mcmc_result$samples
  n_features <- min(12, ncol(samples))
  
  feature_vars <- apply(samples, 2, var)
  top_indices <- order(feature_vars, decreasing = TRUE)[1:n_features]
  
  plot_data <- data.frame()
  for (i in top_indices) {
    df_temp <- data.frame(
      coefficient = samples[, i],
      feature = feature_names[i]
    )
    plot_data <- rbind(plot_data, df_temp)
  }
  
  p <- ggplot(plot_data, aes(x = coefficient)) +
    geom_density(fill = "steelblue", alpha = 0.6) +
    facet_wrap(~ feature, scales = "free", ncol = 3) +
    labs(title = "Posterior Distributions of Coefficients",
         subtitle = "Top features by posterior variance",
         x = "Coefficient Value",
         y = "Density") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 10))
  
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 12, height = 8, dpi = 300)
  cat("Saved posterior distribution plot to", output_path, "\n")
  
  return(p)
}
=======

ordinal_cm <- confusionMatrix(
  factor(test$ord_pred_label, ordered = FALSE),
  factor(test$label_ord, ordered = FALSE)
)
>>>>>>> bc00809 (feat : bayesian model trained suneetpathangay@gmail.com)

plot_feature_importance <- function(mcmc_result, feature_names, 
                                    output_path = "results/bayesian_feature_importance.png") {
  exclude_idx <- which(feature_names == "intercept")
  if (length(exclude_idx) > 0) {
    idx <- setdiff(1:length(feature_names), exclude_idx)
  } else {
    idx <- 1:length(feature_names)
  }
  
  means <- abs(mcmc_result$posterior_mean[idx])
  top_indices <- order(means, decreasing = TRUE)[1:min(15, length(means))]
  top_indices <- idx[top_indices]
  
  plot_data <- data.frame(
    feature = feature_names[top_indices],
    mean = mcmc_result$posterior_mean[top_indices],
    lower = mcmc_result$posterior_ci_lower[top_indices],
    upper = mcmc_result$posterior_ci_upper[top_indices],
    significant = (mcmc_result$posterior_ci_lower[top_indices] > 0 | 
                   mcmc_result$posterior_ci_upper[top_indices] < 0)
  )
  
  plot_data$feature <- factor(plot_data$feature, 
                              levels = plot_data$feature[order(plot_data$mean)])
  
  p <- ggplot(plot_data, aes(x = feature, y = mean, color = significant)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    coord_flip() +
    labs(title = "Feature Importance (Posterior Means with 95% Credible Intervals)",
         x = "Feature",
         y = "Coefficient Value",
         color = "Significant\n(CI excludes 0)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "right")
  
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 10, height = 8, dpi = 300)
  cat("Saved feature importance plot to", output_path, "\n")
  
  return(p)
}

plot_trace_plots <- function(mcmc_result, feature_names, 
                             output_path = "results/bayesian_traces.png") {
  samples <- mcmc_result$samples
  n_features <- min(6, ncol(samples))
  
  feature_vars <- apply(samples, 2, var)
  top_indices <- order(feature_vars, decreasing = TRUE)[1:n_features]
  
  plot_data <- data.frame()
  for (i in top_indices) {
    df_temp <- data.frame(
      iteration = 1:nrow(samples),
      value = samples[, i],
      feature = feature_names[i]
    )
    plot_data <- rbind(plot_data, df_temp)
  }
  
  p <- ggplot(plot_data, aes(x = iteration, y = value)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~ feature, scales = "free_y", ncol = 2) +
    labs(title = "MCMC Trace Plots",
         x = "Iteration",
         y = "Coefficient Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 10, height = 8, dpi = 300)
  cat("Saved trace plots to", output_path, "\n")
  
  return(p)
}

<<<<<<< HEAD
evaluate_model <- function(X_test, y_test, posterior_mean) {
  probs <- predict_bayesian(X_test, posterior_mean)
  predictions <- ifelse(probs > 0.5, 1, 0)
  
  accuracy <- mean(predictions == y_test)
  
  cm <- table(Predicted = predictions, Actual = y_test)
  cm_matrix <- matrix(as.numeric(cm), nrow = nrow(cm), ncol = ncol(cm))
  rownames(cm_matrix) <- rownames(cm)
  colnames(cm_matrix) <- colnames(cm)
  
  if (nrow(cm_matrix) == 2 && ncol(cm_matrix) == 2) {
    tp <- cm_matrix[2, 2]
    tn <- cm_matrix[1, 1]
    fp <- cm_matrix[2, 1]
    fn <- cm_matrix[1, 2]
    
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    f1 <- 2 * (precision * recall) / (precision + recall)
    
    metrics <- list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1 = f1,
      confusion_matrix = cm_matrix
    )
  } else {
    metrics <- list(
      accuracy = accuracy,
      confusion_matrix = cm_matrix
    )
  }
  
  return(list(
    metrics = metrics,
    predictions = predictions,
    probabilities = probs
  ))
}

run_bayesian_analysis <- function(features_path = "data/roi_features.json",
                                  test_split = 0.2,
                                  n_iter = 10000,
                                  binary = TRUE) {
  cat("=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "\n")
  cat("Bayesian Logistic Regression for Alzheimer's MRI Classification\n")
  cat("=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "\n\n")
  
  data <- load_roi_features(features_path, binary = binary)
  X <- data$X
  y <- data$y
  feature_names <- data$feature_names
  
  set.seed(42)
  n <- nrow(X)
  test_indices <- sample(n, floor(n * test_split))
  train_indices <- setdiff(1:n, test_indices)
  
  X_train <- X[train_indices, ]
  y_train <- y[train_indices]
  X_test <- X[test_indices, ]
  y_test <- y[test_indices]
  
  cat(sprintf("Train set: %d samples\n", length(y_train)))
  cat(sprintf("Test set: %d samples\n\n", length(y_test)))
  
  mcmc_result <- mcmc_sampler(X_train, y_train, n_iter = n_iter)
  
  cat("\nEvaluating on test set...\n")
  eval_result <- evaluate_model(X_test, y_test, mcmc_result$posterior_mean)
  
  cat("\nTest Set Performance:\n")
  cat(sprintf("Accuracy: %.4f\n", eval_result$metrics$accuracy))
  if ("precision" %in% names(eval_result$metrics)) {
    cat(sprintf("Precision: %.4f\n", eval_result$metrics$precision))
    cat(sprintf("Recall: %.4f\n", eval_result$metrics$recall))
    cat(sprintf("F1 Score: %.4f\n", eval_result$metrics$f1))
  }
  cat("\nConfusion Matrix:\n")
  print(eval_result$metrics$confusion_matrix)
  
  cat("\nGenerating visualizations...\n")
  plot_posterior_distributions(mcmc_result, feature_names)
  plot_feature_importance(mcmc_result, feature_names)
  plot_trace_plots(mcmc_result, feature_names)
  
  test_metrics <- eval_result$metrics
  if ("confusion_matrix" %in% names(test_metrics)) {
    test_metrics$confusion_matrix <- as.matrix(test_metrics$confusion_matrix)
  }
  
  results <- list(
    posterior_mean = as.numeric(mcmc_result$posterior_mean),
    posterior_sd = as.numeric(mcmc_result$posterior_sd),
    posterior_ci_lower = as.numeric(mcmc_result$posterior_ci_lower),
    posterior_ci_upper = as.numeric(mcmc_result$posterior_ci_upper),
    feature_names = as.character(feature_names),
    test_metrics = test_metrics,
    acceptance_rate = as.numeric(mcmc_result$acceptance_rate)
  )
  
  results_path <- "results/bayesian_results.json"
  dir.create(dirname(results_path), showWarnings = FALSE, recursive = TRUE)
  write_json(results, results_path, pretty = TRUE)
  cat("\nSaved results to", results_path, "\n")
  
  cat("\nTop 10 Most Important Features:\n")
  feature_importance <- data.frame(
    feature = feature_names[-1],
    mean = abs(mcmc_result$posterior_mean[-1]),
    ci_lower = mcmc_result$posterior_ci_lower[-1],
    ci_upper = mcmc_result$posterior_ci_upper[-1]
  )
  feature_importance <- feature_importance[order(feature_importance$mean, decreasing = TRUE), ]
  print(head(feature_importance, 10))
  
  cat("\nAnalysis complete!\n")
  
  return(list(
    mcmc_result = mcmc_result,
    eval_result = eval_result,
    results = results
  ))
}

if (!interactive()) {
  features_path <- "data/roi_features.json"
  if (!file.exists(features_path)) {
    cat("ERROR: ROI features file not found at", features_path, "\n")
    cat("Please run extract_roi_features.py first to extract features.\n")
    cat("Example: python src/extract_roi_features.py <path_to_dataset>\n")
  } else {
    result <- run_bayesian_analysis(features_path = features_path)
  }
}
=======

if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

saveRDS(
  binary_fit,
  file = file.path(model_dir, "binary_bayes_cnn_probs.rds")
)

saveRDS(
  ordinal_fit,
  file = file.path(model_dir, "ordinal_bayes_cnn_probs.rds")
)

cat("\nSaved models to", model_dir, "\n")
cat("\n==== Analysis Complete ====\n")
>>>>>>> bc00809 (feat : bayesian model trained suneetpathangay@gmail.com)
