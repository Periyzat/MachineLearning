#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
import numpy as np
get_ipython().system('pip install word2number # installing "word2number" module')
get_ipython().system('pip install --upgrade pip # updating this module')
from word2number import w2n # transferring word values to number
from sklearn import linear_model


# In[171]:


df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/2/hiring.csv")
df.head(3)


# In[172]:


# 1. Data Preprocessing


# In[173]:


df.experience = df.experience.fillna('zero') # filling NaN values with "zero"  
df.head(3)


# In[174]:


df['experience'] = df['experience'].astype(str) # Transferring experience column data type to str
experience_type = df["experience"].dtypes # chacking data type
print(experience_type)


# In[175]:


df["experience"] = df["experience"].apply(w2n.word_to_num) # after changing, assign it back to it's previous column
df


# In[176]:


import math 
m_test_score = math.floor(df["test_score(out of 10)"].mean()) # getting mean value to fill df["test_score(out of 10)"]=NaN
m_test_score


# In[177]:


df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(m_test_score) # assigning that mean to NaN
df


# In[178]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']], df["salary($)"])


# In[179]:


reg.coef_


# In[180]:


reg.intercept_


# In[181]:


reg.predict([[2,9,6]])


# In[182]:


reg.predict([[12,10,10]])


# In[ ]:





# In[ ]:




