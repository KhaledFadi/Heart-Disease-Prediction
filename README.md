# Heart Disease Prediction Project

## Overview
This project involves analyzing and predicting heart disease using various machine learning algorithms. The project includes data cleaning, preprocessing, visualization, and model training to predict the presence of heart disease.

## Table of Contents
- [Installation](#installation)
- [Data Description](#data-description)
- [Data Cleaning](#data-cleaning)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [License](#license)

## Installation
To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install the required packages using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
## Data Description
The dataset used in this project is a CSV file containing medical information related to heart disease. The dataset includes features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and a target variable indicating the presence of heart disease.

## Data Cleaning
- Handling Missing Values: Removing rows with null values.

- Duplicate Records: Identifying and removing duplicate records.
## Data Preprocessing
1-Viewing Data: Displaying the first and last 10 rows of the dataset.

2-Descriptive Statistics: Calculating statistical measures for the dataset.

3-Grouping and Filtering: Grouping data by various attributes and calculating relevant statistics.

4-Scaling: Normalizing the feature columns using MinMaxScaler.

## Data Visualization
1-Histograms: Plotting histograms to show the distribution of key features.

2-Scatter Plots: Visualizing the relationship between different features and the target variable.
## Model Training and Evaluation
The following machine learning models are trained and evaluated:

1-K-Nearest Neighbors (KNN):

- Defining the model and training on the dataset.
  
- Evaluating the model using accuracy score and confusion matrix.
  
2-Naive Bayes:
  
- Defining the Gaussian Naive Bayes model.
  
- Training and evaluating the model using accuracy score and confusion matrix.
  
3-Support Vector Machine (SVM):
- Defining the SVM model with a linear kernel.
  
- Training and evaluating the model using accuracy score and confusion matrix.
  
4-Random Forest:
- Defining the Random Forest model.
  
- Training and evaluating the model using accuracy score and confusion matrix.
  
5-XGBoost:
- Defining the XGBoost model.
  
- Training and evaluating the model using accuracy score and confusion matrix.
  
## Results
The results of the models' performance are compared, and the accuracy of each model is reported. The confusion matrix is used to visualize the classification results.
## License
This project is licensed under the Apache License 2.0.
