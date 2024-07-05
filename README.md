# Heart Disease Prediction
This repository contains a project aimed at predicting heart disease using machine learning techniques. The project involves data preprocessing, model training, and evaluation to determine the accuracy of the predictions.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Description](#data-description)
- [Model-Training](#model-training)
- [Results](#results)
- [License](#license)
  
## Introduction
Heart disease is a major health concern worldwide, and early detection can significantly improve treatment outcomes. This project leverages machine learning to predict the presence of heart disease in patients based on various health metrics.

## Installation
To run this project locally, you need to have Python and the following libraries installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## Data Description
The dataset used in this project is a CSV file containing medical information related to heart disease. The dataset includes features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and a target variable indicating the presence of heart disease.

## Model Training
The model training process involves the following steps:

1-Data Preprocessing: Cleaning the data, handling missing values, and encoding categorical variables.

2-Feature Selection: Selecting the most relevant features for the model.

3-Splitting the Data: Dividing the dataset into training and testing sets.

4-Model Selection: Choosing the machine learning algorithm. In this case, we used Logistic Regression.

5-Training the Model: Fitting the model to the training data.

## Results
The results of the models' performance are compared, and the accuracy of each model is reported. The confusion matrix is used to visualize the classification results.
## License
This project is licensed under the Apache License 2.0.
