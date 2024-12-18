# Predicting Client Subscription to Term Deposits

## Project Overview

Marketing to potential clients is vital for banking success. Banks often use channels like social media, customer service, and strategic partnerships to connect with customers. But how can they target specific demographics more accurately? Machine learning transforms this process by leveraging data and analytics to predict which clients are most likely to subscribe to financial products.

This project aims to predict whether a client will subscribe to a term deposit based on a set of features from a banking institution’s marketing campaign dataset. By building a predictive model, the objective is to provide actionable insights that can help optimize future campaigns.

## Objectives
	1.	Conduct Exploratory Data Analysis (EDA) to uncover patterns and relationships in the data.
	2.	Engineer features to improve predictive accuracy.
	3.	Build and evaluate machine learning models to predict client subscription.
	4.	Provide actionable insights for marketing teams.

## Dataset Information
The data originates from direct marketing campaigns of a banking institution, involving phone calls to clients.
	Main Dataset Used: bank-additional-full.csv
	Features: 20 input features such as age, job, marital status, education, campaign duration, etc.
	Target variable (y): Whether the client subscribed to a term deposit (yes or no).
	Key Challenges: Class imbalance (Fewer clients subscribed compared to those who didn’t)

Steps and Methodology
	1.	Exploratory Data Analysis (EDA):
	•	Visualized data distributions and relationships between features and the target variable.
	•	Identified missing values, outliers, and imbalanced classes.
	•	Explored correlations and important categorical patterns.
	2.	Data Preprocessing:
	•	Handled missing values and outliers.
	•	Encoded categorical variables using one-hot encoding.
	•	Scaled numerical features to normalize distributions.
	•	Addressed class imbalance using the SMOTE technique.
	3.	Feature Engineering:
	•	Selected relevant features based on correlations and significance tests.
	•	Created new features such as call success rates or campaign duration statistics.
	4.	Model Building and Training:
	•	Baseline model: Logistic Regression.
	•	Advanced models: Random Forest and Gradient Boosting (e.g., XGBoost).
	•	Used cross-validation and hyperparameter tuning to improve performance.
	5.	Model Evaluation:
	•	Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
	•	Addressed class imbalance through proper metric selection and model tuning.
	6.	Insights and Recommendations:
	•	Highlighted key features driving subscription (e.g., call duration, previous campaign outcomes).
	•	Suggested strategies for targeting specific client demographics and optimizing campaign timing.

Key Results
	•	Best Model: Random Forest 
	•	Performance Metrics:
	•	Accuracy: 97.45%
	•	Precision: Class 0 = 1.00, Class 1 = 0.95
	•	Recall: Class 0 = 0.95, Class 1 = 1.00
	•	ROC-AUC: 0.9956

Usage Instructions
1.	Setup Environment:
	•	Install required libraries: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
2.	Run Code:
	•	Use the provided Python script or Jupyter Notebook to reproduce results.
	•	Ensure the dataset (bank-additional-full.csv) is placed in the working directory.
3.	Files:
	•	data/: Contains the dataset file(s).
	•	notebook.ipynb: Jupyter notebook with step-by-step analysis and model building.
•	script.py: Python script for end-to-end execution.
	

Key Insights for Marketing Teams

1. Target Age-Specific Campaigns: Focus marketing efforts on age groups most likely to invest, such as middle-aged individuals.

2. Leverage Economic Indicators: Monitor Euribor3m rates and employment trends. A favorable economic environment can be used to promote term deposits as a better alternative to savings accounts.

3. Optimize Campaign Contact Frequency: A balance in the number of contacts is critical—persistent follow-ups can increase subscriptions, but excessive calls may lead to customer fatigue.

4. Segment Clients Based on Housing Status: Create tailored campaigns for clients with and without housing loans, as their financial priorities differ.

5. Timing Matters: Clients recently contacted in previous campaigns are more responsive. Re-engage them strategically.

Contact Information

For questions or feedback, please reach out to:
	•	Name: Annette Akonnor
	•	Email: akuaakonnor43@gmail.com
