#!/usr/bin/env python
# coding: utf-8

# In[2]:


#The purpose of this project is to investigate and understand the data provided.

#The goal is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.

#Part 1: Understand the situation
#Prepare to understand and organize the provided taxi cab dataset and information.

#Part 2: Understand the data
#Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities.
#Compile summary information about the data to inform next steps.

#Part 3: Understand the variables
#Use insights from your examination of the summary data to guide deeper investigation into specific variables.


# In[3]:


import pandas as pd               
import numpy as np              

df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')


# In[4]:


df.head(10)


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


#Findings : 22699 Rows By 18 Columns, No Missing data(NaN),Datetime Columns should be in datetime not object.
#         : The fare_amount column shows wide distribution, with a maximum value of around $1000, which is significantly higher than 25th and 75th percentiles.
#           This suggests the presence of extreme outliers,There is also the existence of negative fare values which may represent data entry errors or refund records.
#         : For trip_distance most trips appear to be relatively short (1-3 miles), However there are trips exceeding 33 miles which may suggest potential outliers.


# In[ ]:


# Sort and Interpret trip_distance and total_amount.


# In[9]:


# Sort the data by trip distance from maximum to minimum value
df_sort = df.sort_values(by=['trip_distance'],ascending=False)
df_sort.head(10)


# In[10]:


# Sort the data by total amount and print the top 20 values
total_amount_sorted = df.sort_values(
    ['total_amount'], ascending=False)['total_amount']
total_amount_sorted.head(20)


# In[11]:


# Sort the data by total amount and print the bottom 20 values
total_amount_sorted.tail(20)


# In[12]:


# How many of each payment type are represented in the data?
df['payment_type'].value_counts()


# In[13]:


# What is the average tip for trips paid for with credit card?
avg_cc_tip = df[df['payment_type']==1]['tip_amount'].mean()
print('Avg. cc tip:', avg_cc_tip)

# What is the average tip for trips paid for with cash?
avg_cash_tip = df[df['payment_type']==2]['tip_amount'].mean()
print('Avg. cash tip:', avg_cash_tip)


# In[14]:


# How many times is each vendor ID represented in the data?
df['VendorID'].value_counts()


# In[15]:


# What is the mean total amount for each vendor?
df.groupby(['VendorID']).mean(numeric_only=True)[['total_amount']]


# In[16]:


# Filter the data for credit card payments only
credit_card = df[df['payment_type']==1]

# Filter the credit-card-only data for passenger count only
credit_card['passenger_count'].value_counts()


# In[17]:


# Calculate the average tip amount for each passenger count (credit card payments only)
credit_card.groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]


# In[18]:


#Findings : The values in trip_distance align with our earlier data discovery, where we noticed that the longest rides are approximately 33 miles.
#         : The first two values total_amountof are significantly higher than the others.
#         : The most expensive rides are not necessarily the longest ones.


# In[19]:


#Verdict : After looking at the dataset, the two variables that are most likely to help build a predictive model for taxi ride fares are total_amount and trip_distance.


# 
