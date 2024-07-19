# Consumer Complaint Resolution Prediction Model

This is a self-made project which involves using a machine learning classification model in order to predict the grievance status of a consumer in various banking firms, involving multitide of financial grievances from loans to debts and credit faults. 

## Project Overview and Dataset

The dataset was picked up from kaggle. You may find it here : https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints

The dataset consists of 18 columns and 555957 rows, with information about the company concerned, their products and sub-products, issue and sub-issue and the residential information of the consumer. Each individual consumer is identified by his/her consumer ID. No personal information of any consumer is exploited for the analysis and model building. 

For the prediction task, I have used an AdaBoost Classifier model in order to predict if consumers end up satisfied or dissatisfied even after the company has catered to their issues. The problem is thus a 'binary classification task'. 

## Requirements 
- Python 3.11.0
- Numpy
- Pandas
- Seaborn and Matplotlib
- Scikit-learn for models and preprocessing
  
## Tasks Accomplished
- Used techniques like data cleaning, EDA and data imputation to prepare data for analysis and model building
- Used Python libraries like matplotlib and seaborn for data visualisation and draw inferences from the data
- Test various classification models and figure out the appropriate one based on the accuracy and precision scores
- Finally, use AdaBoost Classifier to predict if the consumer ends up disputed or not
- Using streamlit to deploy the model and make it function

## Model Performance 
- Accuracy Score = 0.7981
- Precision Score = 1.0

## Access 
You may access the model from : https://consumer-app-f0rr.onrender.com
