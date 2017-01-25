
# coding: utf-8

# # Data Loading, Storage, and File Formats

# In[1]:

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -u -d -v')


# In[2]:

import datetime
datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[3]:

get_ipython().magic('matplotlib inline')


# In[4]:

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

# In[5]:

result = pd.read_csv('pydata-book/ch06/ex6.csv')
result


# In[6]:

pd.read_csv('pydata-book/ch06/ex6.csv', nrows=5)


# In[12]:

chunker = pd.read_csv('pydata-book/ch06/ex6.csv', chunksize=1000)
chunker


# In[13]:

tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
    tot = tot.sort_values(ascending=False)
tot


# ### Writing Data Out to Text Format

# In[14]:

data = pd.read_csv('pydata-book/ch06/ex5.csv')
data


# In[15]:

data.to_csv('out.csv')
print(open('out.csv').read())


# In[17]:

import sys
data.to_csv(sys.stdout, sep='|')


# In[18]:

data.to_csv(sys.stdout, na_rep='NULL')


# In[19]:

data.to_csv(sys.stdout, index=False, header=False)


# In[21]:

# data.to_csv(sys.stdout, index=False, cols=['a', 'b', 'c']) # no such arg "cols"
data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])


# In[22]:

dates = pd.date_range('1/1/2000', periods=7)
dates


# In[23]:

ts = Series(np.arange(7), index=dates)
ts


# In[24]:

ts.to_csv('tseries.csv')
print(open('tseries.csv').read())


# In[25]:

Series.from_csv('tseries.csv', parse_dates=True)


# ### Manually Working with Delimited Formats

# In[26]:

datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[27]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -v -p $imports')

