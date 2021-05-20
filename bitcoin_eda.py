#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.offline as py
import plotly.graph_objs as go


# In[24]:


btc=pd.read_csv("bitcoin.csv")


# In[25]:


btc.head()


# In[26]:


btc.drop(columns=["Unnamed: 0","open","high","low"],inplace=True)


# In[27]:


btc.head()


# In[28]:


plt.plot(btc["time"],btc["close"])
plt.xlabel("date")
plt.ylabel("closeprice")
plt.show()


# In[29]:


from datetime import datetime
btc["time"] = pd.to_datetime(btc['time'])


# In[30]:


btc.info()


# In[31]:


btc["time"]=btc["time"].dt.strftime("%Y-%m-%d, %H:%M")


# In[32]:


btc.head()


# In[33]:


btc.to_csv("btc_eda.csv")


# In[ ]:




