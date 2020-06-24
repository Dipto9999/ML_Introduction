#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[1]:


df = pd.read_csv('http://bit.ly/autompg-csv')
df.head


# In[2]:


df = pd.read_csv('http://bit.ly/autompg-csv')
df.head()


# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('http://bit.ly/autompg-csv')
df.head()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.plot.scatter(x='hp', y='mpg')


# In[ ]:




