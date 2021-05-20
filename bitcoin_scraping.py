#!/usr/bin/env python
# coding: utf-8

# In[25]:


import bitfinex


# In[26]:


# Create api instance of the v2 API
api_v2 = bitfinex.bitfinex_v2.api_v2()


# In[27]:


result = api_v2.candles()


# In[28]:


import datetime
import time
# Define query parameters
pair = 'btcusd' # Currency pair of interest
bin_size = '1m' # This will return minute data
limit = 10000    # We want the maximum of 1000 data points 
# Define the start date
t_start = datetime.datetime(2021,4,12, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000
# Define the end date
t_stop = datetime.datetime(2021, 4, 20, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000
result = api_v2.candles(symbol=pair, interval=bin_size,  
                     limit=limit, start=t_start, end=t_stop)


# In[30]:


import pandas as pd


# In[31]:


names = ['time', 'open', 'close', 'high', 'low', 'volume']


# In[32]:


df = pd.DataFrame(result, columns=names)


# In[33]:


df.head()


# In[34]:


df['time'] = pd.to_datetime(df['time'], unit='ms')


# In[35]:


df.head()


# In[36]:


df.tail()


# In[37]:


df.drop_duplicates(inplace=True)


# In[38]:


df.shape


# In[39]:


df.to_csv("bitcoin.csv")


# In[ ]:




