
# coding: utf-8

# # Chapter 7 Data Wrangling: Clean, Transform, Merge, Reshape

# In[3]:

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -u -d -t -n -m -v')


# In[4]:

get_ipython().magic('matplotlib inline')


# In[5]:

import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd


# ## Combining and Merging Data Sets

# In[6]:

imports = get_ipython().magic('imports_')
get_ipython().magic('watermark -u -d -t -n -m -v -p $imports')

