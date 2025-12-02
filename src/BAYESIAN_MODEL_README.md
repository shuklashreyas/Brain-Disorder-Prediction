# Bayesian Logistic Regression Model for Alzheimer's MRI Classification

## Overview

This Bayesian model provides interpretable classification of Alzheimer's disease from MRI images by analyzing extracted Region-of-Interest (ROI) features. Unlike the CNN which operates on raw pixels, this model focuses on specific brain regions and their statistical properties, offering insights into which anatomical features are most predictive of Alzheimer's.

## Key Features

- **ROI Feature Extraction**: Extracts intensity, texture, and histogram features from key brain regions (hippocampus, ventricles, cortex)
- **Bayesian Inference**: Uses Metropolis-Hastings MCMC sampling for posterior inference
- **Interpretability**: Provides posterior distributions, credible intervals, and feature importance rankings
- **Uncertainty Quantification**: Full posterior distributions allow for uncertainty estimates in predictions

## Files

- `extract_roi_features.py`: Python script to extract ROI features from MRI images
- `bayesian_model.R`: Main R script with Bayesian logistic regression implementation
- `compare_models.R`: Script to compare CNN and Bayesian model predictions

## Usage

### Step 1: Extract ROI Features

First, extract features from your MRI dataset:

```bash
python src/extract_roi_features.py <path_to_dataset>
```

The dataset should have the following structure:
```
dataset/
  NonDemented/
    *.jpg
  VeryMildDemented/
    *.jpg
  MildDemented/
    *.jpg
  ModerateDemented/
    *.jpg
```

This will create `data/roi_features.json` with extracted features.

### Step 2: Run Bayesian Analysis

In R or RStudio:

```r
source("src/bayesian_model.R")
result <- run_bayesian_analysis(
  features_path = "data/roi_features.json",
  test_split = 0.2,
  n_iter = 10000,
  binary = TRUE
)
```

Or from command line:

```bash
Rscript src/bayesian_model.R
```

### Step 3: Compare with CNN

After running both models:

```r
source("src/compare_models.R")
comparison <- compare_predictions()
```

## Model Details

### Feature Extraction

The model extracts the following features from each MRI image:

1. **Global Features**:
   - Mean, standard deviation, skewness, kurtosis
   - Texture measures
   - Histogram entropy

2. **ROI Features** (for each of 4 brain regions):
   - Mean, std, min, max, median intensity
   - Texture measures (local variance)
   - Histogram entropy

### Bayesian Framework

- **Prior**: Gaussian prior on coefficients with variance σ² = 10
- **Likelihood**: Logistic regression likelihood
- **Inference**: Metropolis-Hastings MCMC with adaptive proposal distribution
- **Default**: 10,000 iterations with 2,000 burn-in

### Output

The model generates:

1. **Results JSON** (`results/bayesian_results.json`):
   - Posterior means and standard deviations
   - 95% credible intervals
   - Test set performance metrics

2. **Visualizations**:
   - `results/bayesian_posteriors.png`: Posterior distributions for top features
   - `results/bayesian_feature_importance.png`: Feature importance with credible intervals
   - `results/bayesian_traces.png`: MCMC trace plots

3. **Feature Importance** (`results/bayesian_feature_importance.json`):
   - Ranked list of most important ROI features

## Interpretation

### Feature Importance

Features with posterior credible intervals that exclude zero are considered "significant". Positive coefficients indicate features associated with Alzheimer's (demented), while negative coefficients indicate features associated with healthy brains.

### Key Brain Regions

The model typically identifies:
- **Hippocampus regions**: Critical for memory, often atrophied in Alzheimer's
- **Ventricle regions**: Enlarged ventricles are a sign of brain atrophy
- **Cortex regions**: Thinning cortex indicates disease progression

### Comparison with CNN

- **CNN**: Best performance, operates on raw pixels, less interpretable
- **Bayesian**: Good performance, operates on extracted features, highly interpretable
- **Combined**: Use CNN for predictions, Bayesian model for understanding which regions matter

## R Dependencies

Required R packages:
- `MASS`: Statistical functions
- `ggplot2`: Visualization
- `jsonlite`: JSON I/O
- `dplyr`: Data manipulation

Install with:
```r
install.packages(c("MASS", "ggplot2", "jsonlite", "dplyr"))
```

## Parameters

### `run_bayesian_analysis()`

- `features_path`: Path to JSON file with extracted features
- `test_split`: Proportion of data for testing (default: 0.2)
- `n_iter`: Number of MCMC iterations (default: 10000)
- `binary`: Use binary classification? (default: TRUE)
  - `TRUE`: NonDemented (0) vs Demented (1, 2, 3 → 1)
  - `FALSE`: 4-class classification (0, 1, 2, 3)

### MCMC Tuning

If acceptance rate is too low (< 20%) or too high (> 50%), adjust `proposal_sd` in `mcmc_sampler()`:
- Lower acceptance rate → decrease `proposal_sd`
- Higher acceptance rate → increase `proposal_sd`

## Example Output

```
Bayesian Logistic Regression for Alzheimer's MRI Classification
================================================================

Loading ROI features from data/roi_features.json
Using binary classification: NonDemented (0) vs Demented (1)
Loaded 5124 samples with 25 features
Class distribution: 2562, 2562

Train set: 4099 samples
Test set: 1025 samples

Running MCMC sampler...
Iterations: 10000 (burn-in: 2000)
Iteration 1000/10000 (Acceptance rate: 32.50%)
...
MCMC Complete!
Final acceptance rate: 31.25%

Evaluating on test set...

Test Set Performance:
Accuracy: 0.8244
Precision: 0.8123
Recall: 0.8456
F1 Score: 0.8287

Top 10 Most Important Features:
                    feature      mean
1            roi_0_texture 0.4523
2            roi_1_std     0.3891
3            global_mean   0.3210
...
```

