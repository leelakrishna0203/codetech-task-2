# codetech-task-2
Name:M LEELAKRISHNA

Company:CODETECH IT SOLUTIONS

ID:CT6DS708

Date:June25th to August10th,2024

Mentor:Muzammil Ahmed

## Overview of the Project

## Project:Preditive Modeling with Linear Regression on E-commerce Customer Analysis

This project analyzes customer data from an e-commerce platform to understand key factors influencing yearly spending. The analysis includes data visualization, feature selection, model training, and evaluation using linear regression.

## Objective

The objective of this project is to identify and analyze factors that influence the yearly amount spent by customers, using linear regression to predict customer spending based on various features.

## Key Activities

### 1. Data Loading and Exploration
- **Loading Data:** The dataset was loaded using Pandas.
- **Initial Exploration:** Displayed the first few rows and basic information about the dataset, including data types and summary statistics.

### 2. Data Visualization
- **Joint Plots:** 
  - Analyzed the relationship between 'Time on Website' and 'Yearly Amount Spent'.
  - Analyzed the relationship between 'Time on App' and 'Yearly Amount Spent'.
- **Pair Plot:** Visualized pairwise relationships between all numerical columns.
- **Linear Model Plot:** Examined the relationship between 'Length of Membership' and 'Yearly Amount Spent'.

### 3. Data Splitting
- **Feature Selection:** Selected features ('Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership') as predictors (`X`) and 'Yearly Amount Spent' as the target variable (`Y`).
- **Train-Test Split:** Divided the data into training and testing sets with a 70-30 split using `train_test_split`.

### 4. Model Training
- **Linear Regression:**
  - Trained a linear regression model on the training data.
  - Extracted model coefficients to interpret feature importance.

### 5. Model Evaluation
- **Predictions:** Predicted values on the test set and compared them with actual values.
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### 6. Residual Analysis
- **Residual Calculation:** Calculated residuals to evaluate model accuracy.
- **Visualization:** 
  - Plotted the distribution of residuals.
  - Created a Q-Q plot to check the normality of residuals.

## Technologies Used
- **Python Libraries:** 
  - Pandas for data manipulation
  - Matplotlib and Seaborn for data visualization
  - Scikit-learn for machine learning
# CODE

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

import math

import pylab

import scipy.stats as stats


# Load the dataset
df = pd.read_csv('Ecommerce Customers')

# Initial exploration
df.head()

df.info()

df.describe()

# Data visualization

sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=df, alpha=0.5)

sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df, alpha=0.5)

sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})

# Feature selection and data splitting

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

Y = df['Yearly Amount Spent']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Model training

lm = LinearRegression()

lm.fit(X_train, Y_train)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['coef'])

print(cdf)

# Predictions and evaluation

Predictions = lm.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(Y_test, Predictions))

print('Mean Squared Error:', mean_squared_error(Y_test, Predictions))

print('Root Mean Squared Error:', math.sqrt(mean_squared_error(Y_test, Predictions)))

# Residual analysis

residuals = Y_test - Predictions

sns.displot(residuals, bins=30, kde=True)

# Q-Q plot

stats.probplot(residuals, dist="norm", plot=pylab)

pylab.show()
