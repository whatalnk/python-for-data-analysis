
# coding: utf-8

# # Data Loading, Storage, and File Formats

# In[74]:

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -u -d -v')


# In[75]:

import datetime
datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[63]:

get_ipython().magic('matplotlib inline')


# In[64]:

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

# In[5]:

print(open('pydata-book/ch06/ex7.csv').read())


# In[6]:

import csv
f = open('pydata-book/ch06/ex7.csv')
reader = csv.reader(f)
for line in reader:
    print(line)


# In[7]:

lines = list(csv.reader(open('pydata-book/ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict


# In[ ]:

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
reader = csv.reader(f, dialect=my_dialect)


# In[ ]:

reader = csv.reader(f, delimiter='|')


# ### JSON Data

# In[8]:

obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
{"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""


# In[9]:

import json


# In[11]:

result = json.loads(obj)
result


# In[12]:

result['name']


# In[13]:

result['pet']


# In[14]:

result['places_lived']


# In[15]:

result['siblings']


# In[16]:

siblings = DataFrame(result['siblings'], columns=['name', 'age'])
siblings


# ### XML and HTML: Web Scraping

# * `urllib2` は `urllib.request`, `urllib.parse`, `urllib.error` に分割
# * API変更で動かないらしい（tableがとれない）

# In[2]:

from lxml.html import parse
from urllib.request import urlopen


# In[44]:

doc.findall('.//table')[0].findall('.//tr')[0].text_content()


# In[45]:

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
doc


# In[47]:

doc.base_url


# In[ ]:

links = doc.findall('.//a')


# In[5]:

lnk = links[28]
lnk


# In[6]:

lnk.get('href')


# In[7]:

lnk.text_content()


# In[8]:

urls = [lnk.get('href') for lnk in doc.findall('.//a')]
urls


# In[10]:

tables = doc.findall('.//table')
tables


# In[20]:

tables[0]


# In[21]:

tables[0].findall('.//tr')


# In[25]:

row = tables[0].findall('.//tr')


# In[23]:

tables[0].findall('.//tr')[0].findall('.//td')


# In[24]:

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]


# In[26]:

_unpack(row[0], kind='th')


# In[27]:

_unpack(row[0], kind='td')


# In[29]:

from pandas.io.parsers import TextParser


# In[30]:

def parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()


# In[34]:

_unpack(tables[0].findall('.//tr')[0], kind='th')


# #### Parsing XML with lxml.objectify

# In[58]:

from lxml import objectify


# In[59]:

path = 'pydata-book/ch06/mta_perf/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()


# In[60]:

print(open(path).read())


# In[61]:

data = []
skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
'DESIRED_CHANGE', 'DECIMAL_PLACES']
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
        data.append(el_data)
data


# In[65]:

perf = DataFrame(data)
perf


# In[67]:

# from StringIO import StringIO
from io import StringIO


# In[68]:

tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()


# In[69]:

root


# In[70]:

root.get('href')


# In[72]:

root.text


# ## Binary Data Formats

# In[76]:

datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[78]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -v -p $imports')

