
# coding: utf-8

# # Data Loading, Storage, and File Formats

# In[1]:

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -u -d -v')


# In[27]:

import datetime
datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd


# ## Reading and Writing Data in Text Format

# In[10]:

print(open('pydata-book/ch06/ex1.csv').read())


# In[4]:

df = pd.read_csv('pydata-book/ch06/ex1.csv')
df


# 区切り指定

# In[6]:

pd.read_table('pydata-book/ch06/ex1.csv', sep=',')


# ヘッダー行なし

# In[11]:

print(open('pydata-book/ch06/ex2.csv').read())


# In[13]:

pd.read_csv('pydata-book/ch06/ex2.csv', header=None)


# In[14]:

names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('pydata-book/ch06/ex2.csv', names=names, index_col='message')


# In[15]:

print(open('pydata-book/ch06/csv_mindex.csv').read())


# In[16]:

parsed = pd.read_csv('pydata-book/ch06/csv_mindex.csv', index_col=['key1', 'key2'])
parsed


# 正規表現で区切る

# In[17]:

list(open('pydata-book/ch06/ex3.txt'))


# In[18]:

result = pd.read_table('pydata-book/ch06/ex3.txt', sep='\s+')
result


# 任意の行をスキップできる

# In[19]:

list(open('pydata-book/ch06/ex4.csv'))


# In[20]:

pd.read_csv('pydata-book/ch06/ex4.csv', skiprows=[0, 2, 3])


# 欠測値

# In[21]:

print(open('pydata-book/ch06/ex5.csv').read())


# In[22]:

pd.read_csv('pydata-book/ch06/ex5.csv')


# In[23]:

result = pd.read_csv('pydata-book/ch06/ex5.csv', na_values=['NULL'])
result


# In[24]:

sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
result = pd.read_csv('pydata-book/ch06/ex5.csv', na_values=sentinels)
result


# ### Reading Text Files in Pieces

# In[41]:

datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[42]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -v -p $imports')

