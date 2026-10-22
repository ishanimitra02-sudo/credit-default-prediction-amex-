# AMEX Credit Default Prediction

A machine learning project using logistic regression,
behavioural feature engineering, risk segmentation,
and approval cutoff simulation.

# American Express Credit Default Prediction

## Project Overview

This project uses machine learning to predict credit card defaults using customer-level behavioural data from the American Express Default Prediction Kaggle competition. The objective is to understand how historical customer behaviour can help identify customers with a higher likelihood of default and support data-driven credit risk decisions.

## Dataset

The project uses the American Express Default Prediction dataset available on Kaggle. It contains anonymized customer information recorded across multiple monthly statements, along with a target variable indicating whether a customer defaulted.

The analysis uses a sample of 50,000 customers to keep the modelling process computationally manageable.

Dataset: https://www.kaggle.com/competitions/amex-default-prediction

The dataset must be downloaded separately from Kaggle because the original data files are large and subject to Kaggle's access conditions.

## Methodology

### 1. Data Preparation

The customer-level dataset was prepared from monthly statement records. Since each customer can have multiple observations over time, the monthly information was transformed into features that summarize behavioural patterns.

### 2. Behavioural Feature Engineering

Features were created from customers' monthly statement histories to capture different aspects of their behaviour:

- **Average level:** Typical values across monthly observations.
- **Variability:** Changes and fluctuations in behaviour over time.
- **Latest value:** The most recent recorded value.
- **Trend:** The direction of change across the available statement history.

These features provide a customer-level representation of historical behaviour for predicting default risk.

### 3. Logistic Regression Model

A logistic regression model was trained to estimate the probability of credit card default for each customer.

The model produces a predicted default probability, allowing customers to be ranked from lower to higher predicted risk.

### 4. Model Evaluation

The model was evaluated using metrics suited to credit risk prediction:

- **ROC-AUC:** Measures how well the model distinguishes between customers who default and those who do not.
- **KS statistic:** Measures the maximum separation between the cumulative distributions of defaulting and non-defaulting customers.
- **AMEX metric:** Evaluates ranking performance using the competition's weighted Gini measure and the proportion of defaults captured in the highest-risk portion of the ranked population.

These metrics help assess the model's ability to differentiate customer risk.

### 5. Risk Banding

Customers were grouped into risk bands based on their predicted default probabilities. The bands were examined to compare customer distributions and observed default rates across risk levels.

This helps assess whether the model meaningfully separates lower-risk and higher-risk customers.

### 6. Approval Cutoff Simulation

A cutoff simulation was developed to examine how different predicted default probability thresholds affect credit approval decisions.

For each cutoff, the analysis calculates:

- Number and percentage of customers approved.
- Number of customers declined.
- Observed default rate among approved customers.
- Number of observed defaults among approved and declined customers.
- Percentage of observed defaults excluded from the approved group.

The simulation illustrates the trade-off between approval volume and portfolio default risk under alternative thresholds.

These thresholds are analytical scenarios for the project, not actual American Express credit approval rules.

## Technology Stack

- Python
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning and model evaluation
- Matplotlib for visualizations
- Google Colab for notebook execution
- GitHub for version control and project documentation

## How to Run the Project

1. Join the American Express Default Prediction competition on Kaggle and download the required dataset files.
2. Open the project notebook in Google Colab.
3. Upload or connect the dataset files using the paths expected by the notebook.
4. Run the notebook cells in order to prepare the data, engineer behavioural features, train the logistic regression model, and generate predictions.
5. Evaluate the model using ROC-AUC, KS, and the AMEX competition metric.
6. Run the risk banding and approval cutoff simulation cells to inspect the results.
7. Save the notebook and relevant output tables to GitHub.

The exact data-loading steps depend on how the dataset is made available in the Colab environment.

## Limitations

- The dataset contains anonymized features, so the precise business meaning of individual variables is not always known.
- Results depend on the sampled customers, feature engineering choices, and evaluation split.
- Approval cutoff analysis is retrospective and does not establish the causal impact of changing a real credit policy.
- Model performance on this dataset does not establish production readiness or guarantee similar performance on a live credit card portfolio.


