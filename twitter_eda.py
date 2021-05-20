#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import re


# In[84]:


twitter_data=pd.read_csv("twitterdata.csv")
# bitcoin=pd.read_csv("bitcoin.csv")


# In[85]:


twitter_data.head()


# In[86]:


twitter_data.drop(columns="Unnamed: 0",inplace=True)


# In[87]:


twitter_data.tail()


# In[88]:


twitter_data=twitter_data.sort_values(by="time",ignore_index=True)


# In[89]:


twitter_data.info()


# In[90]:


from datetime import datetime
twitter_data["time"] = pd.to_datetime(twitter_data['time'])


# In[91]:


from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)


# In[92]:


def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())


# In[93]:


twitter_data["clean_tweet"]=twitter_data["tweets"].apply(lambda x:clean_tweet(x))


# In[94]:


def analyze_sentiment(tweet):
    analysis=TextBlob(tweet)
    if analysis.sentiment.polarity>0:
        return "positive"
    elif analysis.sentiment.polarity==0:
        return "neutral"
    else:
        return "negative"


# In[95]:


twitter_data["sentiment"]=twitter_data["clean_tweet"].apply(lambda x:analyze_sentiment(x))


# In[96]:


def getpolarity(text):
    return TextBlob(text).sentiment.polarity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
twitter_data["polarity"]=twitter_data["clean_tweet"].apply(getpolarity)
twitter_data["subjectivity"]=twitter_data["clean_tweet"].apply(getsubjectivity)


# In[97]:


twitter_data.head()


# In[98]:


twitter_data["sentiment"].value_counts().iplot(kind="bar",xTitle="sentiment",yTitle="count")


# In[99]:


all_tweets = ' '.join(tweet for tweet in twitter_data['clean_tweet'])


# In[100]:


wordcloud = WordCloud(stopwords=STOPWORDS).generate(all_tweets)


# In[101]:


plt.figure(figsize = (10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[102]:


df_freq = pd.DataFrame.from_dict(data = wordcloud.words_, orient='index')
df_freq = df_freq.head(20)
df_freq.plot.bar()
plt.show()


# In[103]:


twitter_data.drop(columns=["tweets"],inplace=True)


# In[104]:


twitter_data.head()


# In[105]:


twitter_data["time"]=twitter_data["time"].dt.strftime("%Y-%m-%d, %H:%M")


# In[106]:


twitter_data.head()


# In[107]:


twitter_data.to_csv("twitter_eda.csv")


# In[108]:


df=pd.read_csv("twitter_eda.csv")


# In[109]:


df.head()


# In[ ]:




