#!/usr/bin/env python
# coding: utf-8

# # Consumer Complaint Resolution Prediction Model

# Aim of the project is to develop a model that when provided with the user data and the complaint information, is able to predict if the particular complaint could be eliminated in the future.
# 
# ##### APPLICATIONS : 
# Such models could help organisations to improve customer services and fasten up their grievance catering methods in order to prevent piling up of complaints. These models can also be integrated with response mechanisms that can filter out grievances that were similar to some prior catered complaints, avoiding repeated actions. 

# ### Importing Libraries

# In[86]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from io import StringIO
from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:


from datetime import date
import time
import joblib


# ### Importing dataset and data overview

# In[3]:


data = pd.read_csv("consumer_complaints.csv")


# ##### Looking into a sample of data 

# In[4]:


data.sample(10)


# In[5]:


data.info()


# ##### Looking into the info, we infer that there are multiple rows with abundant null values which would not contribute much to prediction

# In[6]:


print("shape of the data = ", data.shape)


# ### Data pre-processing

# In[7]:


data.duplicated().sum()


# In[8]:


data.isnull().sum()


# ##### We'll drop the columns which have more than 25% of null values 

# In[9]:


for i in data.columns:
    if(data[i].isnull().sum()/len(data) >= 0.25):
        data = data.drop(i, axis = 1)


# In[10]:


data.columns


# In[11]:


data.isnull().sum()


# ##### We are left with only the zipcode and state to deal with

# In[12]:


data['state'].nunique()


# In[13]:


data["state"].value_counts()


# ##### Since the complaints from CA are the highest with a large margin, we can impute the null data with CA and drop the zipcodes

# In[14]:


data['state'].fillna("CA", inplace = True)


# In[15]:


data = data.drop(['zipcode'], axis = 1)


# In[16]:


data['issue'].nunique()


# In[17]:


data['date_received'] = pd.to_datetime(data['date_received'])
data['date_sent_to_company'] = pd.to_datetime(data['date_sent_to_company'])


# In[18]:


data = data.drop(['complaint_id'], axis = 1)


# ##### We have dropped the IDs since they do not contribute to prediction

# In[19]:


data['company_response_to_consumer'].unique()


# ##### We will take into account the delay in registering the concern to the company and the date and month of the registration

# In[20]:


data['latency'] = (data['date_sent_to_company'] - data['date_received']).dt.days


# In[21]:


data['day'] = data['date_sent_to_company'].dt.day
data['month'] = data['date_sent_to_company'].dt.month


# In[22]:


data = data.drop(['date_received', 'date_sent_to_company'], axis = 1)


# In[23]:


data.head()


# ##### Checking the number of classes in each categorical column

# In[25]:


data['product'].nunique()


# In[26]:


data['state'].nunique()


# In[27]:


data['submitted_via'].nunique()


# In[28]:


data['company_response_to_consumer'].nunique()


# In[29]:


data['product'].value_counts()


# ### Data Analysis and Visualisation

# In[30]:


plt.figure(figsize = (10, 4))
fig = sns.countplot(data = data, x = 'product')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("product name")
plt.title("Product types and count")
plt.show()


# ##### We find that the most concerns are regarding Mortgages, followed by Debts and Credits

# In[31]:


print("percentage share of products in the complaints", data['product'].value_counts()/len(data)*100)


# In[32]:


plt.figure(figsize = (10, 4))
fig = sns.countplot(data = data, x = 'state', order=data['state'].value_counts().iloc[:15].index)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("state names")
plt.title("complaints registered per state")
plt.show()


# In[33]:


#CA is the state with the most number of complaints


# In[34]:


ca_data = data[data['state'] == 'CA']


# In[35]:


print("percentage share of products in the complaints in CA", ca_data['product'].value_counts()/len(ca_data)*100)


# In[36]:


print("percentage share of products in the complaints in CA", ca_data['company_response_to_consumer'].value_counts()/len(ca_data)*100)


# In[37]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = ca_data, x = 'consumer_disputed?')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Consumer Disputed")
plt.show()


# In[38]:


#maybe complaints are high because they are being resolved well


# In[39]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = ca_data, x = 'timely_response')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Timely Response?")
plt.show()


# In[40]:


#consumers file more complaints in CA because they are catered well


# In[41]:


fl_data = data[data['state'] == 'FL']


# In[42]:


print("percentage share of products in the complaints in CA", fl_data['product'].value_counts()/len(fl_data)*100)


# In[43]:


print("percentage share of products in the complaints in CA", fl_data['company_response_to_consumer'].value_counts()/len(fl_data)*100)


# In[44]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = fl_data, x = 'consumer_disputed?')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Consumer Disputed")
plt.show()


# In[45]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = fl_data, x = 'timely_response')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Timely Response?")
plt.show()


# ##### We observe that in the top 2 states with the most complaints "mortgage" was the highest cause of complaints and the consumers ended up satisfied 

# ##### It could be concluded that complaints were high in these states because of the efficient resolution of the complaints

# In[47]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = data, x = 'submitted_via')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Submission Methods")
plt.show()


# In[48]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = data, x = 'consumer_disputed?')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Consumer Disputed")
plt.show()


# In[49]:


plt.figure(figsize = (5,5))
fig = sns.countplot(data = data, x = 'company_response_to_consumer')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.xlabel("Response Distribution")
plt.show()


# #### Analysing Latency and Dispute Behaviour

# In[51]:


print("maximum delay in receiving concern",data['latency'].max(), "days")
print("average delay in receiving concern",round(data['latency'].mean(), 3), "days")


# In[52]:


print("Number of customers disputed even after timely response :", len(data[(data["timely_response"] == "Yes") & (data["consumer_disputed?"] == "Yes")]))


# In[53]:


print("Number of customers disputed even after untimely response :", len(data[(data["timely_response"] == "No") & (data["consumer_disputed?"] == "Yes")]))


# In[54]:


print("Number of customers not disputed after timely response :", len(data[(data["timely_response"] == "Yes") & (data["consumer_disputed?"] == "No")]))


# In[55]:


print("Number of customers not disputed even after untimely response :", len(data[(data["timely_response"] == "No") & (data["consumer_disputed?"] == "No")]))


# #### Company Analysis

# In[56]:


data["company"].value_counts().sort_values(ascending = False).head(20)


# In[57]:


len(data[(data["company"] == "Bank of America") & (data["consumer_disputed?"] == "Yes")])


# In[58]:


len(data[(data["company"] == "Bank of America") & (data["consumer_disputed?"] == "No")])


# In[59]:


data.head()


# #### Analysing Timelines of Complaint Registration and Results

# In[60]:


len(data[(data["consumer_disputed?"] == "Yes") & (data["day"] > 15)])


# In[61]:


len(data[(data["consumer_disputed?"] == "Yes") & (data["day"] < 15)])


# In[62]:


plt.figure(figsize = (5,5))
sns.countplot(data = data, x = "month")
plt.xlabel("month of the year")
plt.ylabel("number of complaints")
plt.show()


# In[63]:


plt.figure(figsize = (5,5))
sns.countplot(data = data[data["timely_response"] == "No"], x = "month")
plt.xlabel("month of the year")
plt.ylabel("number of untimely responses")
plt.show()


# ### Data Preparation for Model Training

# In[65]:


lc_prod = LabelEncoder()


# In[66]:


lc_prod.fit(data["product"])
data["product"] = lc_prod.transform(data["product"])


# In[67]:


lc_comp = LabelEncoder()
lc_comp.fit(data["company"])
data["company"] = lc_comp.transform(data["company"])


# In[68]:


lc_state = LabelEncoder()
lc_state.fit(data["state"])
data["state"] = lc_state.transform(data["state"])


# In[69]:


lc_submit = LabelEncoder()
lc_submit.fit(data["submitted_via"])
data["submitted_via"] = lc_submit.transform(data["submitted_via"])


# In[70]:


lc_resp = LabelEncoder()
lc_resp.fit(data["company_response_to_consumer"])
data["company_response_to_consumer"] = lc_resp.transform(data["company_response_to_consumer"])


# In[71]:


lc_opt = LabelEncoder()
lc_opt.fit(data["timely_response"])
data["timely_response"] = lc_opt.transform(data["timely_response"])
data["consumer_disputed?"] = lc_opt.transform(data["consumer_disputed?"])


# In[72]:


data = data.drop(["issue"], axis = 1)


# In[73]:


data.head()


# ### Training and Testing Sets

# In[74]:


y = data["consumer_disputed?"]
x = data.drop(['consumer_disputed?'], axis = 1)


# In[75]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# ### Testing Models

# In[76]:


def test_model(model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #comparing with y_test
    prec_score = precision_score(y_test, y_pred, zero_division = 1)
    acc_score = accuracy_score(y_test, y_pred)
    print("precision score :", prec_score)
    print("accuracy score :", acc_score)


# In[77]:


test_model(LogisticRegression())


# In[78]:


test_model(DecisionTreeClassifier(criterion = 'entropy', random_state = 42))


# In[79]:


test_model(RandomForestClassifier())


# In[80]:


test_model(AdaBoostClassifier(n_estimators = 10))


# In[81]:


test_model(KNN())


# ### Model Peformance

# In[82]:


adaboost = AdaBoostClassifier()
adaboost.fit(x_train, y_train)

y_pred = adaboost.predict(x_test)

print(round(accuracy_score(y_test, y_pred), 3))


# In[83]:


print("accuracy score = ", round(accuracy_score(y_test, y_pred), 3))


# ### Saving the Model

# In[84]:


joblib.dump(adaboost, "model.pkl")


# In[ ]:




