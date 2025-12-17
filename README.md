# Accident Analysis Project

This project performs machine learning analysis on traffic accident data to predict the severity of accidents (injury vs no-injury).

## Overview

The project includes comprehensive data preprocessing, exploratory data analysis, and implementation of multiple machine learning models to classify accident severity. The models evaluated include Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).

## Files

- `Accident.py` - Main Python script containing all analysis code
- `accidents.csv` - Dataset containing traffic accident records with various features
- `Accident_analysis.pdf` - Analysis report documentation

## Features

The dataset includes the following features:
- RushHour: Indicator for rush hour
- WRK_ZONE: Work zone indicator
- WKDY: Weekday indicator
- INT_HWY: Intersection/Highway indicator
- LGTCON_day: Light condition during day
- LEVEL: Road level
- SPD_LIM: Speed limit
- SUR_COND_dry: Surface condition (dry/not dry)
- TRAF_two_way: Two-way traffic indicator
- WEATHER_adverse: Adverse weather condition indicator
- MAX_SEV: Maximum severity (target variable: injury/no-injury)

## Methodology

### Data Preprocessing

1. **Descriptive Statistics**: Comprehensive statistical summary including mean, median, mode, and other descriptive metrics
2. **Missing Value Treatment**: 
   - Mode imputation for categorical variables
   - Median imputation for `SPD_LIM` (speed limit)
3. **Outlier Detection**: Boxplot analysis for numerical variables
4. **Data Splitting**: 70% training, 15% validation, 15% test sets

### Models Implemented

1. **Logistic Regression**
   - L2 regularization with high C value
   - Coefficient analysis and statistical significance testing using statsmodels

2. **Decision Tree**
   - Grid search cross-validation for hyperparameter tuning
   - Parameters optimized: max_depth, min_samples_split, min_impurity_decrease
   - Decision tree visualization

3. **Random Forest**
   - Grid search for hyperparameter optimization
   - Feature importance analysis
   - Parameters optimized: n_estimators, max_depth, min_samples_split

4. **K-Nearest Neighbors (KNN)**
   - Standard scaling for feature normalization
   - K value optimization (1-14 neighbors)
   - Best k value: 7

### Model Evaluation

Each model is evaluated using:
- Confusion Matrix
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- Error Rate

### Additional Analysis

- **Variance Inflation Factor (VIF)**: Multicollinearity analysis
- **Feature Effect Simulation**: Analysis of how each feature affects prediction probabilities across different models

## Requirements

Required Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels

Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

## Usage

1. Ensure the CSV file path in `Accident.py` matches your file location:
   ```python
   df = pd.read_csv('/Users/yizhen/Desktop/6105hw/assignment3/accidents-Assignment_3.csv')
   ```

2. Run the script:
   ```bash
   python Accident.py
   ```

## Results

The script outputs:
- Descriptive statistics and data summary
- Missing value counts
- Model coefficients and parameters
- Confusion matrices and performance metrics for each model
- Feature importance rankings (Random Forest)
- Visualization plots (boxplots, decision tree, accuracy curves)

## Author

yizhen

