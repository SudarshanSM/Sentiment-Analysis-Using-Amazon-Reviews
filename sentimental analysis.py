#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option('display.max_columns',None)


# In[2]:


pip install textblob


# In[3]:



get_ipython().system('pip install wordcloud')


# In[4]:


get_ipython().system('pip install cufflinks')


# In[5]:


import pandas as pd


# In[7]:


df=pd.read_csv('C:/Users/Sudarshan/Desktop/amazon.csv')


# In[8]:


df.head()


# In[9]:


constraints=['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#EB0CD5']
def categorical_variable_summary(df,coloumn_name):
    fig = make_subplots(rows = 1,cols = 2,
                     subplot_titles=('Countplot','Percentage'),
                     specs=[[{"type":"xy"},{'type':'domain'}]])
    
    fig.add_trace(go.Bar(y=df[coloumn_name].value_counts().values.tolist(),
                        x=[str(i)for i in df[coloumn_name].value_counts().index],
                        text=df[coloumn_name].value_counts().values.tolist(),
                        textfont=dict(size=14),
                        name=coloumn_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=constraints,
                                   line=dict(color='#DBE6EC',
                                            width=1))),
                 row=1,col=1)
    fig.add_trace(go.Pie(labels=df[coloumn_name].value_counts().keys(),
                         values=df[coloumn_name].value_counts().values,
                         textfont=dict(size=18),
                         textposition='auto',
                         showlegends=False,
                         name=coloumn_name,
                         marker=dict(colors=constraints)),
                  row=1,col=2)
    
    fig.update_layout(title={'text':coloumn_name,
                            'y':0.9,
                            'x':0.5,
                            'xanchor':'center',
                            'yanchor':'top'},
                     template='ploty_white')
    iplot(fig)


# In[10]:


from plotly.subplots import make_subplots


# In[11]:


import plotly.graph_objects as go


# In[12]:


constraints=['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#EB0CD5']
def categorical_variable_summary(df,coloumn_name):
    fig = make_subplots(rows = 1,cols = 2,
                     subplot_titles=('Countplot','Percentage'),
                     specs=[[{"type":"xy"},{'type':'domain'}]])
    
    fig.add_trace(go.Bar(y = df[coloumn_name].value_counts().values.tolist(),
                        x=[str(i)for i in df[coloumn_name].value_counts().index],
                        text=df[coloumn_name].value_counts().values.tolist(),
                        textfont=dict(size=14),
                        name=coloumn_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=constraints,
                                   line=dict(color='#DBE6EC',
                                            width=1))),
                 row=1,col=1)
    fig.add_trace(go.Pie(labels = df[coloumn_name].value_counts().keys(),
                         values=df[coloumn_name].value_counts().values,
                         textfont=dict(size=18),
                         textposition='auto',
                         showlegend=False,
                         name=coloumn_name,
                         marker=dict(colors=constraints)),
                  row=1,col=2)
    
    fig.update_layout(title={'text':coloumn_name,
                            'y':0.9,
                            'x':0.5,
                            'xanchor':'center',
                            'yanchor':'top'},
                     template='plotly_white')
    iplot(fig)


# In[13]:


import plotly.express as px


# In[14]:


categorical_variable_summary(df,'overall')


# In[15]:


import plotly.express as px

# Create a scatter plot
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

# Show the plot in Jupyter notebook
fig.show()


# In[ ]:





# In[16]:


df.head()


# In[17]:


df=pd.read_csv('C:/Users/Sudarshan/Desktop/amazon.csv')


# In[18]:


df.head()


# In[19]:


df = df.sort_values("wilson_lower_bound",ascending = False)
df.drop('Unnamed: 0',inplace=True,axis=1)
df.head()


# In[20]:


df


# In[21]:


def missing_values_analysis(df):
    na_columns_=[col for col in df.columns if df[col].isnull().sum()>0]
    n_miss=df[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_=(df[na_columns_].isnull().sum()/df.shape[0]*100).sort_values(ascending=True)
    missing_df=pd.concat([n_miss,np.round(ratio_, 2)],axis=1,keys=['Missing Values','Ratio'])
    missing_df=pd.DataFrame(missing_df)
    return missing_df

def check_dataframe(df,head=5,tail=5):
    print("SHAPE".center(82,'~'))
    print('Rows:{}'.format(df.shape[0]))
    print('coloumns:{}'.format(df.shape[1]))
    print("TYPES".center(82,'~'))
    print(df.dtypes)
    print("".center(82,'~'))
    print(missing_values_analysis(df))
    print('DUPLICATED VALUES'.center(82,'~'))
    print(df.duplicated().sum())
    print("QUANTILES".center(82,'~'))
    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)
    
check_dataframe(df)


# In[22]:


import numpy as np


# In[23]:


from plotly.offline import iplot


# In[24]:


get_ipython().system('pip install plotly')


# In[25]:


constraints=['#B34D22','#EBE00C','#1FEB0C','#0C92EB','#EB0CD5']
def categorical_variable_summary(df,coloumn_name):
    fig = make_subplots(rows = 1,cols = 2,
                     subplot_titles=('Countplot','Percentage'),
                     specs=[[{"type":"xy"},{'type':'domain'}]])
    
    fig.add_trace(go.Bar(y = df[coloumn_name].value_counts().values.tolist(),
                        x=[str(i)for i in df[coloumn_name].value_counts().index],
                        text=df[coloumn_name].value_counts().values.tolist(),
                        textfont=dict(size=14),
                        name=coloumn_name,
                        textposition='auto',
                        showlegend=False,
                        marker=dict(color=constraints,
                                   line=dict(color='#DBE6EC',
                                            width=1))),
                 row=1,col=1)
    fig.add_trace(go.Pie(labels = df[coloumn_name].value_counts().keys(),
                         values=df[coloumn_name].value_counts().values,
                         textfont=dict(size=18),
                         textposition='auto',
                         showlegend=False,
                         name=coloumn_name,
                         marker=dict(colors=constraints)),
                  row=1,col=2)
    
    fig.update_layout(title={'text':coloumn_name,
                            'y':0.9,
                            'x':0.5,
                            'xanchor':'center',
                            'yanchor':'top'},
                     template='plotly_white')
    iplot(fig)


# In[26]:


categorical_variable_summary(df,'overall')


# In[27]:


df.reviewText.head()


# In[28]:


review_example=df.reviewText[2031]
review_example


# In[29]:


import re


# In[30]:


review_example=re.sub("[^a-zA-Z]",'',review_example)
review_example


# In[31]:


review_example=review_example.lower().split()


# In[32]:


review_example


# In[33]:


rt=lambda x:re.sub("[^a-zA-Z]",' ',str(x))
df["reviewText"]=df["reviewText"].map(rt)
df["reviewText"]=df["reviewText"].str.lower()
df.head


# In[34]:


pip install vaderSentiment


# In[35]:


pip install textblob


# In[36]:


from textblob import TextBlob


# In[37]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))

for index,row in df['reviewText'].iteritems():
    
    score=SentimentIntensityAnalyzer().polarity_scores(row)
    
    neg=score['neg']
    neu=score['neu']
    pos=score['pos']
    if neg>pos:
        df.loc[index, 'sentiment']="Negative"
    elif pos>neg:
            df.loc[index, 'sentiment']="Positive"
    else:
                df.loc[index, 'sentiment']="Neutral"


# In[38]:


df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound",
                                            ascending=False).head(5)


# In[39]:


categorical_variable_summary(df,'sentiment')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




