# Brain-Disorder-Prediction

## Project Description

This project aims to detect Alzheimer's disease from brain MRI scans using two complementary machine-learning methods:

### 1. Manual 2D Convolutional Neural Network (Python, NumPy)
* Classifies MRI images into Non-Demented, Very Mild Demented, Mild Demented, or Moderate Demented.
* Implemented from scratch (convolution, ReLU, pooling, fully connected layers, backprop).
* Uses data augmentation, normalization, and stratified evaluation to align with insights from recent literature.

### 2. Bayesian Logistic Regression (R)
* Operates on extracted ROI-level or global image features (intensity, texture, histogram metrics).
* Uses a manually coded Gibbs sampler.
* Provides interpretability and uncertainty estimates by analyzing which features correlate with AD severity.

The goal is to compare deep learning performance with Bayesian interpretability, following the direction suggested by recent research showing that lightweight CNNs outperform classical ML for MRI-based Alzheimer's detection.

## Project Structure
```
BrainScanning/
│
├── data/
│   └── raw/
│       └── alzheimers_dataset/
│           └── Combined Dataset/        # MRI data (train/val/test folders inside)
│
├── notebooks/
│   └── data_cleaning.ipynb              # Preprocessing, EDA, data verification
│
├── src/
│   ├── cnn_model_manual.py              # Manual NumPy CNN implementation (Model 1)
│   └── bayesian_model.R                 # Bayesian model with Gibbs sampling (Model 2)
│
├── streamlit_app/
│   └── app.py                           # Streamlit app for visualization + demo (Extra Credit)
│
├── results/                             # Confusion matrices, ROC curves, posterior plots
│
├── report/                              # Literature review, final report, poster
│
├── README.md
└── requirements.txt
```

## How to Run + Requirements

### 1. Install Python Dependencies

Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Typical packages included:
```
numpy
matplotlib
opencv-python
pillow
streamlit
tensorflow (optional for verification only)
```

### 2. Run the Manual CNN Model (Python)
```bash
python src/cnn_model_manual.py
```

This will:
* Load MRI images
* Preprocess and normalize them
* Train the NumPy CNN
* Save metrics to the `results/` folder

### 3. Run the Bayesian Model (R)

Open R or RStudio:
```r
source("src/bayesian_model.R")
```

This script will:
* Load extracted image features
* Run a Gibbs sampler for Bayesian logistic regression
* Output posterior results to `results/`

### 4. Launch the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

This app displays:
* Project overview
* CNN predictions
* Probability outputs
* Visualizations (ROC curve, confusion matrix)