
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

# In[5]:

frame = pd.read_csv('pydata-book/ch06/ex1.csv')
frame


# In[7]:

# frame.save()
frame.to_pickle('frame_pickle')


# In[8]:

# pd.load()
pd.read_pickle('frame_pickle')


# ### Using HDF5 Format

# `conda install pytables`

# In[11]:

store = pd.HDFStore('mydata.h5')
store


# In[12]:

store['obj1'] = frame
store['obj1_col'] = frame['a']
store


# In[13]:

store['obj1']


# ### Reading Microsoft Excel Files

# `conda install xlrd`

# In[15]:

xls_file = pd.ExcelFile('data.xls')
xls_file


# In[16]:

xls_file.sheet_names


# In[17]:

table = xls_file.parse('Sheet1')
table


# ## Interacting with HTML and Web APIs

# In[18]:

import requests


# In[19]:

url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url)
resp


# In[20]:

data = resp.json()
data[:5]


# In[21]:

issue_labels = DataFrame(data)
issue_labels


# In[23]:

issue_labels.ix[7]


# ## Interacting with Databases

# In[24]:

import sqlite3


# In[25]:

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL, d INTEGER
);"""
con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()


# In[26]:

data = [('Atlanta', 'Georgia', 1.25, 6),
('Tallahassee', 'Florida', 2.6, 3),
('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()


# In[27]:

cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows


# In[28]:

cursor.description


# [Python3] zip はlistにする
# http://stackoverflow.com/questions/27431390/typeerror-zip-object-is-not-subscriptable#27431433

# In[30]:

DataFrame(rows, columns=list(zip(*cursor.description))[0])


# In[31]:

pd.read_sql('select * from test', con)


# In[32]:

con.close()


# ### Storing and Loading Data in MongoDB

# In[34]:

import pymongo


# In[36]:

# con = pymongo. Connection('localhost', port=27017)
con = pymongo.MongoClient('localhost', port=27017)
tweets = con.db.tweets


# In[37]:

import requests, json


# Twitter は API 変更

# In[38]:

url = 'http://search.twitter.com/search.json?q=python%20pandas'
data = json.loads(requests.get(url).text)


# In[40]:

data


# In[ ]:

for tweet in data['results']:
    tweets.save(tweet)


# In[ ]:

cursor = tweets.find({'from_user': 'wesmckinn'})


# In[ ]:

tweet_fields = ['created_at', 'from_user', 'id', 'text']
result = DataFrame(list(cursor), columns=tweet_fields)


# In[41]:

datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# In[42]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -v -p $imports')

