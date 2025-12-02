# Bayesian Logistic Regression for Alzheimer's MRI Classification
# Uses MCMC (Metropolis-Hastings) for posterior inference
# Provides interpretability through posterior distributions and feature importance

library(MASS)
library(ggplot2)
library(jsonlite)
library(dplyr)

# String concatenation helper
`%+%` <- function(a, b) paste0(a, b)

# ============================================================================
# Core Bayesian Functions
# ============================================================================

#' Log posterior probability for Bayesian logistic regression
#' 
#' @param beta Coefficient vector
#' @param X Design matrix (with intercept column)
#' @param y Binary response vector (0/1)
#' @param sigma2 Prior variance for coefficients (default: 10)
#' @return Log posterior probability
log_posterior <- function(beta, X, y, sigma2 = 10) {
  # Avoid numerical overflow
  Xbeta <- X %*% beta
  Xbeta <- pmax(pmin(Xbeta, 50), -50)  # Clip to prevent overflow
  
  # Logistic probability
  p <- 1 / (1 + exp(-Xbeta))
  
  # Log-likelihood (with numerical stability)
  p <- pmax(p, 1e-10)
  p <- pmin(p, 1 - 1e-10)
  log_likelihood <- sum(y * log(p) + (1 - y) * log(1 - p))
  
  # Check for invalid values
  if (is.na(log_likelihood) || is.infinite(log_likelihood)) {
    return(-Inf)  # Reject invalid proposals
  }
  
  # Log-prior (Gaussian with variance sigma2)
  log_prior <- -sum(beta^2) / (2 * sigma2)
  
  result <- log_likelihood + log_prior
  
  # Return -Inf if result is invalid (will cause rejection)
  if (is.na(result) || is.infinite(result)) {
    return(-Inf)
  }
  
  return(result)
}

#' Metropolis-Hastings sampler for Bayesian logistic regression
#' 
#' @param X Design matrix
#' @param y Binary response vector
#' @param n_iter Number of MCMC iterations
#' @param burn_in Number of burn-in iterations
#' @param proposal_sd Standard deviation for proposal distribution
#' @param sigma2 Prior variance for coefficients
#' @return List containing samples, acceptance rate, and posterior means
mcmc_sampler <- function(X, y, n_iter = 10000, burn_in = 2000, 
                         proposal_sd = NULL, sigma2 = 10) {
  p <- ncol(X)
  beta <- rep(0, p)  # Initialize at zero
  samples <- matrix(0, n_iter, p)
  acceptances <- 0
  
  # Adaptive proposal_sd based on number of features
  # Smaller proposal for high-dimensional problems
  if (is.null(proposal_sd)) {
    proposal_sd <- 0.05 / sqrt(p)  # Scale with dimension
    proposal_sd <- max(0.01, min(0.1, proposal_sd))  # Clamp between 0.01 and 0.1
  }
  
  cat("Running MCMC sampler...\n")
  cat(sprintf("Iterations: %d (burn-in: %d)\n", n_iter, burn_in))
  cat(sprintf("Proposal SD: %.4f\n", proposal_sd))
  
  for (i in 1:n_iter) {
    # Propose new beta
    proposal <- beta + rnorm(p, 0, proposal_sd)
    
    # Calculate log acceptance probability
    log_alpha <- log_posterior(proposal, X, y, sigma2) - 
                 log_posterior(beta, X, y, sigma2)
    
    # Handle NaN/Inf values - reject if invalid
    if (is.na(log_alpha) || is.infinite(log_alpha)) {
      if (is.infinite(log_alpha) && log_alpha > 0) {
        # Accept if proposal is infinitely better
        beta <- proposal
        acceptances <- acceptances + 1
      }
      # Otherwise reject (keep current beta)
    } else {
      # Accept or reject based on Metropolis-Hastings rule
      if (log(runif(1)) < log_alpha) {
        beta <- proposal
        acceptances <- acceptances + 1
      }
    }
    
    samples[i, ] <- beta
    
    # Progress indicator
    if (i %% 1000 == 0) {
      cat(sprintf("Iteration %d/%d (Acceptance rate: %.2f%%)\n", 
                  i, n_iter, 100 * acceptances / i))
    }
  }
  
  # Discard burn-in
  if (burn_in >= n_iter) {
    warning("Burn-in period is >= total iterations. Using all samples.")
    post_samples <- samples
  } else {
    post_samples <- samples[(burn_in + 1):n_iter, ]
  }
  
  # Calculate posterior statistics
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

#' Predict probabilities using posterior mean coefficients
#' 
#' @param X_test Test design matrix
#' @param posterior_mean Posterior mean coefficients
#' @return Predicted probabilities
predict_bayesian <- function(X_test, posterior_mean) {
  Xbeta <- X_test %*% posterior_mean
  Xbeta <- pmax(pmin(Xbeta, 50), -50)  # Clip to prevent overflow
  return(1 / (1 + exp(-Xbeta)))
}

# ============================================================================
# Feature Extraction and Data Loading
# ============================================================================

#' Load ROI features from JSON file
#' 
#' @param json_path Path to JSON file with extracted features
#' @param binary If TRUE, use binary classification (NonDemented vs Demented)
#' @return List with X (design matrix), y (labels), feature_names, and metadata
load_roi_features <- function(json_path, binary = TRUE) {
  cat("Loading ROI features from", json_path, "\n")
  
  data <- fromJSON(json_path)
  df <- as.data.frame(data)
  
  # Extract labels
  if (binary) {
    y <- df$binary_label
    cat("Using binary classification: NonDemented (0) vs Demented (1)\n")
  } else {
    y <- df$label
    cat("Using multi-class classification\n")
  }
  
  # Extract feature columns (exclude metadata)
  exclude_cols <- c("label", "binary_label", "image_path", "class_name")
  feature_cols <- setdiff(names(df), exclude_cols)
  
  # Create design matrix
  X <- as.matrix(df[, feature_cols])
  
  # Check for and handle any missing or infinite values
  if (any(is.na(X)) || any(is.infinite(X))) {
    cat("Warning: Found NA or Inf values in features. Replacing with 0.\n")
    X[is.na(X) | is.infinite(X)] <- 0
  }
  
  # Add intercept column
  X <- cbind(1, X)  # Intercept in first column
  
  # Standardize features (except intercept)
  # Handle zero variance features
  X_scaled <- scale(X[, -1])
  if (any(is.na(X_scaled))) {
    # If any features have zero variance, keep them as-is (centered at 0)
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

# ============================================================================
# Visualization Functions
# ============================================================================

#' Plot posterior distributions for coefficients
#' 
#' @param mcmc_result Result from mcmc_sampler
#' @param feature_names Vector of feature names
#' @param output_path Path to save plot
plot_posterior_distributions <- function(mcmc_result, feature_names, 
                                         output_path = "results/bayesian_posteriors.png") {
  samples <- mcmc_result$samples
  n_features <- min(12, ncol(samples))  # Plot top 12 features
  
  # Select features with highest variance (most informative)
  feature_vars <- apply(samples, 2, var)
  top_indices <- order(feature_vars, decreasing = TRUE)[1:n_features]
  
  # Prepare data for plotting
  plot_data <- data.frame()
  for (i in top_indices) {
    df_temp <- data.frame(
      coefficient = samples[, i],
      feature = feature_names[i]
    )
    plot_data <- rbind(plot_data, df_temp)
  }
  
  # Create plot
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
  
  # Save plot
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 12, height = 8, dpi = 300)
  cat("Saved posterior distribution plot to", output_path, "\n")
  
  return(p)
}

#' Plot feature importance (posterior means with credible intervals)
#' 
#' @param mcmc_result Result from mcmc_sampler
#' @param feature_names Vector of feature names
#' @param output_path Path to save plot
plot_feature_importance <- function(mcmc_result, feature_names, 
                                    output_path = "results/bayesian_feature_importance.png") {
  # Exclude intercept
  exclude_idx <- which(feature_names == "intercept")
  if (length(exclude_idx) > 0) {
    idx <- setdiff(1:length(feature_names), exclude_idx)
  } else {
    idx <- 1:length(feature_names)
  }
  
  # Get top features by absolute posterior mean
  means <- abs(mcmc_result$posterior_mean[idx])
  top_indices <- order(means, decreasing = TRUE)[1:min(15, length(means))]
  top_indices <- idx[top_indices]
  
  # Prepare data
  plot_data <- data.frame(
    feature = feature_names[top_indices],
    mean = mcmc_result$posterior_mean[top_indices],
    lower = mcmc_result$posterior_ci_lower[top_indices],
    upper = mcmc_result$posterior_ci_upper[top_indices],
    significant = (mcmc_result$posterior_ci_lower[top_indices] > 0 | 
                   mcmc_result$posterior_ci_upper[top_indices] < 0)
  )
  
  # Order by mean
  plot_data$feature <- factor(plot_data$feature, 
                              levels = plot_data$feature[order(plot_data$mean)])
  
  # Create plot
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
  
  # Save plot
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 10, height = 8, dpi = 300)
  cat("Saved feature importance plot to", output_path, "\n")
  
  return(p)
}

#' Plot MCMC trace plots
#' 
#' @param mcmc_result Result from mcmc_sampler
#' @param feature_names Vector of feature names
#' @param output_path Path to save plot
plot_trace_plots <- function(mcmc_result, feature_names, 
                             output_path = "results/bayesian_traces.png") {
  samples <- mcmc_result$samples
  n_features <- min(6, ncol(samples))
  
  # Select features with highest variance
  feature_vars <- apply(samples, 2, var)
  top_indices <- order(feature_vars, decreasing = TRUE)[1:n_features]
  
  # Prepare data
  plot_data <- data.frame()
  for (i in top_indices) {
    df_temp <- data.frame(
      iteration = 1:nrow(samples),
      value = samples[, i],
      feature = feature_names[i]
    )
    plot_data <- rbind(plot_data, df_temp)
  }
  
  # Create plot
  p <- ggplot(plot_data, aes(x = iteration, y = value)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~ feature, scales = "free_y", ncol = 2) +
    labs(title = "MCMC Trace Plots",
         x = "Iteration",
         y = "Coefficient Value") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
  
  # Save plot
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  ggsave(output_path, p, width = 10, height = 8, dpi = 300)
  cat("Saved trace plots to", output_path, "\n")
  
  return(p)
}

# ============================================================================
# Evaluation Functions
# ============================================================================

#' Evaluate Bayesian model performance
#' 
#' @param X_test Test design matrix
#' @param y_test Test labels
#' @param posterior_mean Posterior mean coefficients
#' @return List with accuracy, predictions, and probabilities
evaluate_model <- function(X_test, y_test, posterior_mean) {
  # Predict probabilities
  probs <- predict_bayesian(X_test, posterior_mean)
  predictions <- ifelse(probs > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(predictions == y_test)
  
  # Confusion matrix (convert to plain numeric matrix for JSON serialization)
  cm <- table(Predicted = predictions, Actual = y_test)
  # Convert to plain matrix, removing any table attributes
  cm_matrix <- matrix(as.numeric(cm), nrow = nrow(cm), ncol = ncol(cm))
  rownames(cm_matrix) <- rownames(cm)
  colnames(cm_matrix) <- colnames(cm)
  
  # Calculate metrics
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

# ============================================================================
# Main Execution Function
# ============================================================================

#' Main function to run Bayesian analysis
#' 
#' @param features_path Path to JSON file with ROI features
#' @param test_split Proportion of data to use for testing (default: 0.2)
#' @param n_iter Number of MCMC iterations
#' @param binary Whether to use binary classification
run_bayesian_analysis <- function(features_path = "data/roi_features.json",
                                  test_split = 0.2,
                                  n_iter = 10000,
                                  binary = TRUE) {
  cat("=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "\n")
  cat("Bayesian Logistic Regression for Alzheimer's MRI Classification\n")
  cat("=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "\n\n")
  
  # Load data
  data <- load_roi_features(features_path, binary = binary)
  X <- data$X
  y <- data$y
  feature_names <- data$feature_names
  
  # Split into train and test
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
  
  # Run MCMC
  mcmc_result <- mcmc_sampler(X_train, y_train, n_iter = n_iter)
  
  # Evaluate on test set
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
  
  # Create visualizations
  cat("\nGenerating visualizations...\n")
  plot_posterior_distributions(mcmc_result, feature_names)
  plot_feature_importance(mcmc_result, feature_names)
  plot_trace_plots(mcmc_result, feature_names)
  
  # Save results
  # Ensure confusion matrix is a proper matrix for JSON serialization
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
  
  # Print top features
  cat("\nTop 10 Most Important Features:\n")
  feature_importance <- data.frame(
    feature = feature_names[-1],  # Exclude intercept
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

# ============================================================================
# Run if executed directly
# ============================================================================

if (!interactive()) {
  # Check if features file exists
  features_path <- "data/roi_features.json"
  if (!file.exists(features_path)) {
    cat("ERROR: ROI features file not found at", features_path, "\n")
    cat("Please run extract_roi_features.py first to extract features.\n")
    cat("Example: python src/extract_roi_features.py <path_to_dataset>\n")
  } else {
    # Run analysis
    result <- run_bayesian_analysis(features_path = features_path)
  }
}
