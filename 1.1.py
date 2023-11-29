#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[22]:


df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/2/homeprices.csv")
df


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area (sqr ft)')
plt.ylabel('price (US $)')
plt.scatter(df.area, df.price, color = 'red', marker = '+')

it = linear_model.LinearRegression() #this line of code isn't working
it.fit(df[['area']], df.price)
it.predict(3300) # this code makes prediction for this area
# In[31]:


it.intercept_ # y=mx+b - bul jerde b


# In[42]:


it.coef_ # y=mx+b - bul jerde m


# In[44]:


135.78767123*3300+180616.43835616432 #this is manual way how to predict that area above



