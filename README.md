# Comparison of Various Machine Learning Models within the MICE Framework

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Data](#data)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Future Work](#future-work)
7. [Dependencies](#dependencies)
8. [Usage](#usage)
9. [Contact](#contact)

## Introduction
This project focuses on the comparison of various machine learning models for imputation within the Multiple Imputation by Chained Equations (MICE) framework. It explores the challenge of handling both categorical and numerical missing data, using real-world activity data from Strava.

## Objectives
- Investigate different imputation methods for handling missing data.
- Compare the performance of various machine learning models.
- Analyze the impact of imputation on model prediction accuracy.
- Select and fine-tune the best performing model for predicting 'Elapsed Time.'

## Data
The dataset consists of Strava activity data, including features like distance, moving time, activity type, and others. Artificial missingness was introduced in the 'Activity Type' column to create a more challenging imputation scenario.

## Methodology
1. **Data Exploration**: Visualization and analysis of relationships between features.
2. **Imputation Comparison**: Evaluation of different imputation methods (e.g., Logistic Regression, Linear Regression, Random Forest).
3. **Modeling and Hyperparameter Tuning**: Comparison of various models (e.g., Lasso, Elastic Net, SVM) with hyperparameter tuning.
4. **Final Model Selection**: Selection and training of the best-performing model.

## Results
- A comprehensive comparison of imputation methods and prediction models.
- Selection of the Random Forest model with Gradient Boosting for imputation.
- Detailed analysis of residuals and correlations in the imputed dataset.
- Achieved meaningful prediction accuracy using RMSE as the evaluation metric.

## Future Work
- Exploration of additional imputation methods.
- Further refinement of modeling techniques.
- Investigation into the handling of outliers and high leverage points.

## Dependencies
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

## Usage
To run the project, execute the Jupyter Notebook `MICE_Benchmark.ipynb`. Custom functions used in the project are located in the `Scripts/imputation.py` file.

## Contact
Micahel Rowlands - mrowlands314@gmail.com
