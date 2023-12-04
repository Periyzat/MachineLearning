#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[49]:


df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/2/homeprices.csv")
df


# In[ ]:


# 1. Data Preprocessing/Cleaning


# In[50]:


import math 
med_brooms = math.floor(df.bedrooms.median())
med_brooms


# In[56]:


df.bedrooms = df.bedrooms.fillna(med_brooms) # assigning to the previous column 
df


# In[60]:


reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', "age"]], df.price) #creating a df using existing df


# In[67]:


reg.coef_


# In[66]:


reg.intercept_


# In[65]:


reg.predict([[3000,3,40]])


# In[68]:


112.06244194*3000+23388.88007794*3-3231.71790863*40+221323.00186540402


# In[69]:


reg.predict([[2500,4,5]])


# In[70]:


112.06244194*2500+23388.88007794*4-3231.71790863*5+221323.00186540402


# In[ ]:




