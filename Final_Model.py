#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler


# In[3]:


bitcoin = pd.read_csv('datasets/bitcoin.csv')
twitter = pd.read_csv('datasets/twitter_eda.csv')


# ## Bitcoin dataset

# In[4]:


bitcoin.tail(10)


# In[5]:


bitcoin.info()


# In[6]:


bitcoin[['date', 'time']] = bitcoin.time.str.split(' ', expand = True)
bitcoin['time'] = bitcoin[['date', 'time']].apply(lambda x: ' '.join(x), axis = 1)
bitcoin['time'] = pd.to_datetime(bitcoin['time'])


# In[7]:


bitcoin.head()


# In[8]:


#Reversing the dataset to match the time series of twitter data
bitcoin = bitcoin.loc[::-1].reset_index(drop = True)


# ## Twitter Dataset

# In[9]:


twitter.isnull().sum()


# In[10]:


twitter[['date', 'time']] = twitter.time.str.split(', ', expand = True)
twitter['time'] = twitter[['date', 'time']].apply(lambda x: ' '.join(x), axis = 1)
twitter['time'] = pd.to_datetime(twitter['time'])


# In[11]:


twitter.head(10)


# In[12]:


twitter.info()


# In[13]:


twitter.drop(columns="Unnamed: 0",inplace=True)
bitcoin.drop(columns="Unnamed: 0",inplace=True)


# In[14]:


merged = pd.merge_asof(bitcoin, twitter, on = 'time', by = 'date', tolerance = pd.Timedelta('4m'))


# In[15]:


merged.head()


# In[16]:


merged.isnull().sum()


# In[17]:


merged.shape


# In[18]:


merged.drop(['likes', 'no:rt'], axis =1, inplace = True)


# In[19]:


merged['clean_tweet'].fillna('Empty', inplace = True)
merged['sentiment'].fillna('neutral', inplace = True)
merged['polarity'].fillna(0, inplace = True)
merged['subjectivity'].fillna(0, inplace = True)
merged.head()


# In[20]:


merged.drop(['time', 'date'], axis =1, inplace = True)


# In[21]:


def getSentiment(score):
    if score < 0:
        return 0 #indicates negative
    elif score == 0:
        return 1  #indicates neutral
    else:
        return 2 #indicates positived


# In[22]:


merged['sentiment_score'] = merged['polarity'].apply(getSentiment)
merged.head()


# In[23]:


import seaborn as sns
sns.countplot(merged["sentiment"])
plt.title("Summary of Counts for Total tweets");


# In[24]:


for i in range(1,len(bitcoin['close'])):
    merged.at[i,'price_difference']=bitcoin.at[i,'close']-bitcoin.at[i-1,'close']


# In[25]:


merged.at[0,'price_difference']=0.0


# In[26]:


merged.head(2)


# In[27]:


merged['target'] = 0
for i in range(10000):
    if merged.price_difference[i] > 0:
        merged['target'][i] = 1 
        
# 0 - price goes down
# 1 - price goes up

merged.head()


# In[28]:


features = merged[['open','high','low','close','volume','polarity','subjectivity','sentiment_score']]
X = np.array(features)
y = np.array(merged['target'])


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# ## Decision Tree

# In[30]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', max_depth=8,
                                        min_samples_leaf=100,
                                        min_samples_split=6,
                                        random_state=42)
clf.fit(X_train,y_train)


# In[31]:


y_predicted = clf.predict(X_test)


# In[32]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print( classification_report(y_test, y_predicted) )


# In[33]:


accuracy_score(y_test,y_predicted)*100


# In[34]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis().fit(X_train, y_train)


# In[35]:


predictions = model.predict(X_test)
predictions


# In[36]:


print( classification_report(y_test, predictions) )


# In[37]:


accuracy_score(y_test,predictions)*100


# In[38]:


import pickle


# ## Dumping the model into a pickle file

# In[39]:


pickle.dump(model,open("model.pkl","wb"))


# In[ ]:




