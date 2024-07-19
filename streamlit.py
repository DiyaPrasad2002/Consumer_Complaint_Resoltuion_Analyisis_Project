#!/usr/bin/env python
# coding: utf-8

# In[39]:



import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle


# In[40]:

model_path = 'adaboostmodel.pkl'
try:
    model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")


# In[41]:


product_encoder = LabelEncoder()
company_encoder = LabelEncoder()
state_encoder = LabelEncoder()
submit_encoder = LabelEncoder()
response_encoder = LabelEncoder()
time_encoder = LabelEncoder()


# In[55]:


product_encoder.fit(['Mortgage', 'Credit reporting', 'Student loan', 'Debt collection',
       'Credit card', 'Bank account or service', 'Consumer Loan',
       'Money transfers', 'Payday loan', 'Prepaid card',
       'Other financial service'])
company_encoder.fit(['Bank of America', 'Wells Fargo & Company', 'JPMorgan Chase & Co.', 'Equifax', 'Citibank', 'TransUnion Intermediate Holdings, Inc.',
                     'Ocwen', 'Capital One', 'Nationstar Mortgage', 'U.S. Bancorp', 'Synchrony Financial', 'Ditech Financial LLC', 'Navient Solutions, Inc.', 
                     'PNC Bank N.A.', 'Encore Capital Group', 'HSBC North America Holdings Inc.', 'Amex', 'SunTrust Banks, Inc.', 'Discover',
                     'TD Bank US Holding Company', 'Select Portfolio Servicing, Inc', 'Portfolio Recovery Associates, Inc.',
                     'Citizens Financial Group, Inc.', 'Fifth Third Financial Corporation', 'Seterus, Inc.', 'Barclays PLC', 'ERC',
                     'BB&T Financial', 'M&T Bank Corporation', 'Ally Financial Inc.', 'Regions Financial Corporation', 'PayPal Holdings, Inc.',
                     'USAA Savings', 'Specialized Loan Servicing LLC', 'Santander Consumer USA Holdings Inc', 'Santander Bank US', 'AES/PHEAA',
                     'Expert Global Solutions, Inc.', 'Flagstar Bank'])
state_encoder.fit(['CA', 'NY', 'MD', 'GA', 'AZ', 'IL', 'NC', 'TX', 'DC', 'KY', 'RI',
       'TN', 'AR', 'AL', 'NJ', 'VA', 'FL', 'MN', 'AK', 'OH', 'OR', 'MO',
       'LA', 'SC', 'OK', 'WA', 'PA', 'MI', 'CO', 'KS', 'MA', 'NH', 'NV',
       'WV', 'PR', 'DE', 'IN', 'UT', 'ME', 'NE', 'NM', 'WY', 'CT', 'HI',
       'ID', 'MS', 'WI', 'IA', 'MT', 'MH', 'VT', 'AE', 'SD', 'FM',
       'VI', 'ND', 'GU', 'MP', 'AP', 'AS', 'PW', 'AA'])
submit_encoder.fit(['Referral', 'Postal mail', 'Email', 'Web', 'Phone', 'Fax'])
response_encoder.fit(['Closed with explanation', 'Closed with monetary relief',
       'Closed with non-monetary relief', 'Closed', 'Untimely response',
       'In progress', 'Closed without relief', 'Closed with relief'])
time_encoder.fit(['Yes', 'No'])


# In[56]:


st.title("Is the consumer disputed?")

product = st.selectbox("product", product_encoder.classes_)
company = st.selectbox("company associated", company_encoder.classes_)
state = st.selectbox("state", state_encoder.classes_)
submission = st.selectbox("mode of submission", submit_encoder.classes_)
response = st.selectbox("company response", response_encoder.classes_)
time = st.selectbox("was it a timely response?", time_encoder.classes_)

latency = st.number_input("delay in submission", min_value = 0, max_value = 1000)
day = st.number_input("date of submission", min_value = 1, max_value = 31)
month = st.number_input("month of submission", min_value = 1, max_value = 12)


# In[57]:


product_enc = product_encoder.transform([product])[0]
company_enc = company_encoder.transform([company])[0]
state_enc = state_encoder.transform([state])[0]
submission_enc = submit_encoder.transform([submission])[0]
response_enc = response_encoder.transform([response])[0]
time_enc = time_encoder.transform([time])[0]


# In[58]:


input_array = np.array([[product_enc, company_enc, state_enc,submission_enc, response_enc,time_enc, latency, day, month]])


# In[59]:


if st.button("Predict"):
    prediction = model.predict(input_array)
    st.write(f"Prediction : {prediction[0]}")


# In[ ]:




