#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[10]:


df = pd.read_csv("/Users/periyzat/Documents/Study/Programming/MLCodebasics/1/canada_per_capita_income.csv")
df.head(3)


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('income(US $)')
plt.scatter(df.year, df.income, color = 'red', marker = '+')


# In[13]:


reg = linear_model.LinearRegression() #this line of code isn't working
reg.fit(df[['year']], df.income) 


# In[15]:


reg.predict([[2020]]) # this code makes prediction for this area


# In[16]:


reg.coef_


# In[18]:


reg.intercept_


# In[19]:


828.46507522*2020-1632210.7578554575


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Set x and y axis labels with correct syntax and font size
plt.xlabel("year", fontsize=20)
plt.ylabel("income", fontsize=20)

# Plot the data points and the regression line
plt.scatter(df.year, df.income, color="red", marker="+")
plt.plot(df.year, reg.predict(df[['year']]), color="blue")

# Show the plot
plt.show()


# In[ ]:




