
# coding: utf-8

# In[1]:

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -u -d -v')


# In[2]:

get_ipython().magic('matplotlib inline')


# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd


# # Chapter 5 Getting Started with pandas

# ## Introduction to pandas Data Structures
# ### Series

# In[3]:

obj = Series([4, 7, -5, 3])
obj


# In[4]:

obj.values


# In[5]:

obj.index


# In[6]:

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[7]:

obj2.index


# In[8]:

obj2[obj2 > 0]


# In[9]:

obj2 * 2


# In[10]:

np.exp(obj2)


# In[11]:

'b' in obj2


# In[12]:

'e' in obj2


# * dict から生成

# In[14]:

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3


# In[15]:

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index = states)
obj4


# * 欠損値

# In[16]:

pd.isnull(obj4)


# In[17]:

pd.notnull(obj4)


# In[18]:

obj4.isnull()


# In[19]:

obj3 + obj4


# * 名前
#     * 列名
#     * テーブル名

# In[20]:

obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[22]:

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj


# In[25]:

obj.index.name = 'column'
obj.name = 'series'
obj


# ### DataFrame

# In[26]:

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 
        'year': [2000, 2001, 2002, 2001, 2002], 
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame


# In[27]:

DataFrame(data, columns=['year', 'state', 'pop'])


# In[29]:

frame2 = DataFrame(data, 
                   columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2


# In[30]:

frame2.columns


# In[31]:

frame2.year


# In[33]:

frame2.ix['three']


# In[34]:

frame2['debt']


# In[35]:

frame2['debt'] = 16.5
frame2['debt']


# In[36]:

frame2['debt'] = np.arange(5.)
frame2['debt']


# In[37]:

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2


# In[38]:

frame2['eastern'] = frame2.state == 'Ohio'
frame2


# In[39]:

del frame2['eastern']
frame2


# In[41]:

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
frame3


# In[42]:

frame3.T


# In[43]:

DataFrame(pop, index=[2001, 2002, 2003])


# In[44]:

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)


# In[45]:

frame3


# In[46]:

frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# In[47]:

frame3.values


# In[48]:

frame2.values


# ### Index Objects

# In[49]:

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index


# In[50]:

index[1:]


# index は変更不可．
# これはエラー

# In[ ]:

index[1] = 'd'


# In[52]:

index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index


# In[53]:

frame3


# In[55]:

'Ohio' in frame3.columns


# In[56]:

2003 in frame3.index


# ## Essential Functionality
# ### Reindexing

# In[57]:

obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# In[58]:

obj


# In[59]:

obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)


# In[61]:

obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')


# In[62]:

frame = DataFrame(np.arange(9).reshape((3, 3)), 
                  index=['a', 'c', 'd'],
                  columns=['Ohio', 'Texas', 'California'])
frame


# In[63]:

frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2


# In[64]:

states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)


# In[65]:

frame.reindex(index=['a', 'b', 'c', 'd'], 
              method='ffill', 
              columns=states)


# In[66]:

frame.ix[['a', 'b', 'c', 'd'], states]


# ### Dropping entries from an axis

# In[67]:

obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
obj


# In[68]:

new_obj = obj.drop('c')
new_obj


# In[69]:

obj


# In[70]:

obj.drop(['d', 'c'])


# In[71]:

obj


# In[72]:

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data


# In[73]:

data.drop(['Colorado', 'Ohio'])


# In[74]:

data.drop('two', axis=1)


# In[75]:

data.drop(['two', 'four'], axis=1)


# ### Indexing, selection, and filtering

# In[76]:

obj = Series(np.arange(4.), index = ['a', 'b', 'c', 'd'])
obj


# In[77]:

obj['a']


# In[78]:

obj[0]


# In[79]:

obj[1:3]


# In[80]:

obj[['b', 'd', 'c']]


# In[81]:

obj[[3, 1]]


# In[82]:

obj[obj < 3]


# In[83]:

obj['a':'c']


# In[84]:

obj['b':'c'] = 5
obj


# In[85]:

data = DataFrame(np.arange(16).reshape((4, 4)),
                 index=['Ohio', 'Colorado', 'Utah', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data


# In[86]:

data['two']


# In[88]:

data[['two', 'one']]


# In[89]:

data[:2]


# In[90]:

data[data['three'] > 5]


# In[91]:

data < 5


# In[92]:

data[data < 5] = 0
data


# In[93]:

data.ix['Colorado', ['two', 'three']]


# In[94]:

data.ix[['Colorado', 'Utah'], [3, 0, 1]]


# In[95]:

data.ix[2]


# In[96]:

data.ix['Utah']


# In[97]:

data.ix[:'Utah', 'two']


# In[98]:

data.ix[data.three > 5, :3]


# ### Arithmetic and data alignment

# In[99]:

s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1


# In[100]:

s2


# In[101]:

s1 + s2


# In[102]:

df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'),
                 index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1


# In[103]:

df2


# In[104]:

df1 + df2


# #### Arithmetic methods with fill values

# In[105]:

df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1


# In[106]:

df2


# In[107]:

df1 + df2


# In[108]:

df1.add(df2, fill_value=0)


# In[109]:

df1.reindex(columns=df2.columns, fill_value=0)


# #### Operations between DataFrame and Series

# In[110]:

arr = np.arange(12.).reshape((3, 4))
arr


# In[111]:

arr[0]


# In[112]:

arr - arr[0]


# In[113]:

frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[114]:

series = frame.ix[0]
series


# In[115]:

frame - series


# In[116]:

series2 = Series(range(3), index=['b', 'e', 'f'])
series2


# In[118]:

frame + series2


# In[119]:

series3 = frame['d']
series3


# In[120]:

frame.sub(series3, axis = 0)


# ### Function application and mapping

# In[121]:

frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame


# In[122]:

frame.abs()


# In[123]:

np.abs(frame)


# In[126]:

f = lambda x: x.max() - x.min()
f


# In[127]:

frame.apply(f)


# In[128]:

frame.apply(f, axis = 1)


# In[129]:

def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])
f


# In[130]:

frame.apply(f)


# In[131]:

frame.apply(f, axis = 1)


# In[132]:

format = lambda x: '%.2f' % x
format


# In[133]:

frame.applymap(format)


# In[134]:

frame['e'].map(format)


# ### Sorting and ranking

# In[135]:

obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()


# In[136]:

frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
frame.sort_index()


# In[137]:

frame.sort_index(axis = 1)


# In[139]:

frame.sort_index(ascending=False)


# In[142]:

obj = Series([4, 7, -3, 2])
# obj.order() # deprecated
obj.sort_values()


# In[143]:

obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()


# In[144]:

frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame


# In[146]:

# frame.sort_index(by = 'b') # deprecated
frame.sort_values(by = 'b')


# In[147]:

# frame.sort_index(by = 'b') # deprecated
frame.sort_values(by = ['a', 'b'])


# * タイは平均

# In[148]:

obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()


# * method='first'で先に出てきたものが上

# In[149]:

obj.rank(method='first')


# In[150]:

obj.rank(ascending=False, method='max')


# In[151]:

frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1],
                   'c': [-2, 5, 8, -2.5]})
frame


# In[152]:

frame.rank(axis = 1)


# In[153]:

frame.rank()


# ### Axis indexes with duplicate values

# In[154]:

obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj


# In[155]:

obj.index.is_unique


# In[156]:

obj['a']


# In[157]:

obj['c']


# In[158]:

df = DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'b'])
df


# In[160]:

df.ix['a']


# ## Summarizing and Computing Descriptive Statistics

# In[161]:

df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
               index=['a', 'b', 'c', 'd'],
               columns=['one', 'two'])
df


# In[162]:

df.sum()


# In[163]:

df.sum(axis = 1)


# In[164]:

df.mean(axis=1, skipna=False)


# In[165]:

df.mean(axis=1)


# In[169]:

df.idxmax()


# In[170]:

df.cumsum()


# In[173]:

df.describe()


# In[174]:

obj = Series(['a', 'a', 'b', 'c'] * 4)
obj


# In[175]:

obj.describe()


# ### Correlation and Covariance

# `pandas.io.data` は `pandas-datareader` に分離
# 
# ```
# conda install pandas-datareader
# ```
# `0.2.1` が入る（最新版は`0.3.0`）
# 
# ---
# 
# `dict.iteritems()` は Python3 では削除
# 
# * `dict.items` を使う?

# ```
# pydoc -p 1234
# ```
# で `localhost:1234` からドキュメントが見られる

# In[4]:

# import pandas.io.data as web
import pandas_datareader.data as web


# In[15]:

all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker)
price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.items()})
volume = DataFrame({tic: data['Volume'] for tic, data in all_data.items()})


# percent change

# In[17]:

returns = price.pct_change()
returns.tail()


# correlation (Series)

# In[18]:

returns.MSFT.corr(returns.IBM)


# covariance (Series)

# In[19]:

returns.MSFT.cov(returns.IBM)


# In[20]:

returns.corr()


# In[21]:

returns.cov()


# In[22]:

returns.corrwith(returns.IBM)


# In[23]:

returns.corrwith(volume)


# ### Unique Values, Value Counts, and Membership

# In[4]:

obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()
uniques


# In[5]:

obj.value_counts()


# In[7]:

pd.value_counts(obj.values, sort=False)


# In[8]:

mask = obj.isin(['b', 'c'])
mask


# In[9]:

obj[mask]


# In[10]:

data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
data


# In[12]:

data.apply(pd.value_counts)


# In[11]:

result = data.apply(pd.value_counts).fillna(0)
result


# ## Handling Missing Data

# In[5]:

string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data


# In[6]:

string_data.isnull()


# In[7]:

string_data[0] = None
string_data.isnull()


# ### Filtering Out Missing Data

# In[8]:

from numpy import nan as NA


# In[9]:

data = Series([1, NA, 3.5, NA, 7])
data.dropna()


# In[10]:

data[data.notnull()]


# In[11]:

data = DataFrame([
        [1., 6.5, 3.], 
        [1., NA, NA],
        [NA, NA, NA], 
        [NA, 6.5, 3.]
    ])
data


# In[12]:

cleaned = data.dropna()
cleaned


# In[14]:

data.dropna(how='all')


# In[15]:

data[4]= NA
data


# In[16]:

data.dropna(axis=1, how='all')


# In[17]:

df = DataFrame(np.random.randn(7, 3))
df


# In[18]:

df.ix[:4, 1] = NA; df.ix[:2, 2] = NA
df


# 3つ以上値が残っているもの

# In[32]:

df.dropna(thresh=3)


# ### Filling in Missing Data

# In[20]:

df.fillna(0)


# In[34]:

df.fillna({1: 0.5, 2: -1})


# In[35]:

_ = df.fillna(0, inplace=True)


# In[36]:

df


# In[37]:

df = DataFrame(np.random.randn(6, 3))
df


# In[38]:

df.ix[2:, 1] = NA; df.ix[4:, 2] = NA
df


# In[39]:

df.fillna(method='ffill')


# In[40]:

df.fillna(method='ffill', limit=2)


# In[42]:

data = Series([1., NA, 3.5, NA, 7])
data


# In[43]:

data.fillna(data.mean())


# ## Hierarchical Indexing

# In[4]:

data = Series(np.random.randn(10),
              index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                     [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
data


# In[5]:

data.index


# In[6]:

data['b']


# In[7]:

data['b':'c']


# In[8]:

data.ix[['b', 'd']]


# In[9]:

data[:, 2]


# In[10]:

data.unstack()


# In[11]:

data.unstack().stack()


# In[5]:

frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],['Green', 'Red', 'Green']])
frame


# In[14]:

frame.index.names


# In[6]:

frame.index.names = ['key1', 'key2']
frame.index.names


# In[16]:

frame.columns.names


# In[7]:

frame.columns.names = ['state', 'color']
frame.columns.names


# In[18]:

frame


# In[19]:

frame['Ohio']


# In[27]:

pd.MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
                       names=['state', 'color'])


# ### Reordering and Sorting Levels

# In[9]:

frame


# In[10]:

frame.swaplevel('key1', 'key2')


# In[13]:

frame.index


# In[14]:

frame.swaplevel('key1', 'key2').index


# In[15]:

frame.sortlevel(1)


# In[17]:

frame.sortlevel(0)


# In[18]:

frame.swaplevel(0, 1).sortlevel(0)


# ### Summary Statistics by Level

# In[19]:

frame.sum(level='key2')


# In[20]:

frame.sum(level='color', axis=1)


# ### Using a DataFrame’s Columns

# In[21]:

frame = DataFrame({'a': range(7), 
                   'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame


# In[22]:

frame2 = frame.set_index(['c', 'd'])
frame2


# In[23]:

frame.set_index(['c', 'd'], drop=False)


# In[24]:

frame2.reset_index()


# ## Other pandas Topics
# ### Integer Indexing

# index の番号か名前かがわからいので，エラーになる

# In[26]:

ser = Series(np.arange(3.))
ser


# In[ ]:

# error
# ser[-1]


# こちらはならない

# In[27]:

ser2 = Series(np.arange(3.), index=['a', 'b', 'c'])
ser2


# In[28]:

ser2[-1]


# In[29]:

ser.ix[:1]


# In[31]:

ser3 = Series(range(3), index=[-5, 1, 3])
ser3


# In[33]:

## ser3.iget_value(2) # depricated
ser3.iloc[2]


# or

# In[35]:

ser3.iat[2]


# In[36]:

frame = DataFrame(np.arange(6).reshape(3, 2), index=[2, 0, 1])
frame


# In[38]:

# frame.irow(0) # depricated
frame.iloc[0]


# ### Panel Data

# In[39]:

import pandas_datareader.data as web


# In[40]:

pdata = pd.Panel(
    dict(
        (stk, web.get_data_yahoo(stk)) for stk in ['AAPL', 'GOOG', 'MSFT', 'DELL']))
pdata


# In[41]:

pdata = pdata.swapaxes('items', 'minor')
pdata


# In[42]:

pdata['Adj Close']


# In[43]:

pdata.ix[:, '6/1/2012', :]


# In[44]:

pdata.ix['Adj Close', '5/22/2012':, :]


# In[45]:

stacked = pdata.ix[:, '5/30/2012':, :].to_frame()
stacked


# In[46]:

stacked.to_panel()


# In[47]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -v -p $imports')

