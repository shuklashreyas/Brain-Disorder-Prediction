# Compare CNN and Bayesian Model Predictions
# Analyzes agreement between CNN and Bayesian model predictions

library(jsonlite)
library(ggplot2)
library(dplyr)

# String concatenation helper
`%+%` <- function(a, b) paste0(a, b)

#' Compare predictions from CNN and Bayesian models
#' 
#' @param cnn_results_path Path to CNN results JSON
#' @param bayesian_results_path Path to Bayesian results JSON
#' @param features_path Path to ROI features JSON (for metadata)
compare_predictions <- function(cnn_results_path = "results/cnn_metrics.json",
                               bayesian_results_path = "results/bayesian_results.json",
                               features_path = "data/roi_features.json") {
  
  cat("Comparing CNN and Bayesian Model Predictions\n")
  cat("=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "=" %+% "\n\n")
  
  # Load results
  if (file.exists(bayesian_results_path)) {
    bayesian_results <- fromJSON(bayesian_results_path)
    cat("Loaded Bayesian results\n")
  } else {
    cat("Warning: Bayesian results not found\n")
    return(NULL)
  }
  
  if (file.exists(cnn_results_path)) {
    cnn_results <- fromJSON(cnn_results_path)
    cat("Loaded CNN results\n")
  } else {
    cat("Warning: CNN results not found\n")
    return(NULL)
  }
  
  # Load features for metadata
  if (file.exists(features_path)) {
    features_data <- fromJSON(features_path)
    cat("Loaded feature metadata\n\n")
  } else {
    cat("Warning: Features file not found\n\n")
    features_data <- NULL
  }
  
  # Extract key metrics
  cat("Model Performance Comparison:\n")
  cat("-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "\n")
  
  if ("test_metrics" %in% names(bayesian_results)) {
    bayesian_acc <- bayesian_results$test_metrics$accuracy
    cat(sprintf("Bayesian Model Accuracy: %.4f\n", bayesian_acc))
    
    if ("precision" %in% names(bayesian_results$test_metrics)) {
      cat(sprintf("Bayesian Precision: %.4f\n", bayesian_results$test_metrics$precision))
      cat(sprintf("Bayesian Recall: %.4f\n", bayesian_results$test_metrics$recall))
      cat(sprintf("Bayesian F1: %.4f\n", bayesian_results$test_metrics$f1))
    }
  }
  
  if ("accuracy" %in% names(cnn_results)) {
    cnn_acc <- cnn_results$accuracy
    cat(sprintf("CNN Model Accuracy: %.4f\n", cnn_acc))
  }
  
  # Feature importance analysis
  cat("\nTop Features from Bayesian Model:\n")
  cat("-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "-" %+% "\n")
  
  if ("feature_names" %in% names(bayesian_results) && 
      "posterior_mean" %in% names(bayesian_results)) {
    # Exclude intercept
    idx <- which(bayesian_results$feature_names != "intercept")
    feature_df <- data.frame(
      feature = bayesian_results$feature_names[idx],
      coefficient = abs(bayesian_results$posterior_mean[idx]),
      ci_lower = bayesian_results$posterior_ci_lower[idx],
      ci_upper = bayesian_results$posterior_ci_upper[idx]
    )
    feature_df <- feature_df[order(feature_df$coefficient, decreasing = TRUE), ]
    
    cat("Top 10 most important ROI features:\n")
    print(head(feature_df, 10))
    
    # Save feature importance
    write_json(feature_df, "results/bayesian_feature_importance.json", pretty = TRUE)
    cat("\nSaved feature importance to results/bayesian_feature_importance.json\n")
  }
  
  # Create comparison summary
  comparison <- list(
    bayesian_accuracy = ifelse("test_metrics" %in% names(bayesian_results), 
                               bayesian_results$test_metrics$accuracy, NA),
    cnn_accuracy = ifelse("accuracy" %in% names(cnn_results), 
                          cnn_results$accuracy, NA),
    top_features = ifelse(exists("feature_df"), 
                         head(feature_df, 10), NULL)
  )
  
  write_json(comparison, "results/model_comparison.json", pretty = TRUE)
  cat("\nSaved comparison to results/model_comparison.json\n")
  
  return(comparison)
}

# Run if executed directly
if (!interactive()) {
  compare_predictions()
}

