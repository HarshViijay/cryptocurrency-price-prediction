#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import time
import datetime
# pd.set_option('display.max_colwidth', -1)
tweepy.__version__


# In[2]:


consumer_key="AnlrMw88AVC7MkQDWYegpUFUP"
consumer_secret="ZFC7Ddxww4sbebgNy5qddUjvMUiKaF6QCYUPD8DwP43AAlWwjA"
access_token="95877051-ei1CoqW4adQeAVycIXIvfosmlZ7e6B4EfpQkxMqcp"
access_token_secret="E9OAvKeIcILhTICAJmWuzfx7oSwR84NihnlRukrr6uBSN"


# In[3]:


auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth)


# In[4]:


startDate = datetime.datetime(2021, 3, 20, 0, 0, 0)
# endDate =   datetime.datetime(2021, 3, 24, 0, 0, 0)


# In[5]:


df = pd.DataFrame(columns = ["tweets","time","likes","no:rt"])


# In[6]:


def stream(data, file_name):
    i = 0
    for tweet in tweepy.Cursor(api.search, q=data, count=100, lang='en',since=startDate).items():
        print(i, end='\r')
        df.loc[i, 'tweets'] = tweet.text
        df.loc[i, 'time'] = tweet.created_at
        df.loc[i, 'likes'] = tweet.favorite_count
        df.loc[i, 'no:rt'] = tweet.retweet_count
        
        df.to_csv('{}.csv'.format(file_name))
        i+=1
        if i == 10000:
            break
        else:
            pass


# In[7]:


stream(data = ['Bitcoin'], file_name = 'twitterdata')


# In[8]:


df.head()


# In[9]:


df.tail()


# In[ ]:




