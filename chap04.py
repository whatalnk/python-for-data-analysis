
# coding: utf-8

# # Chapter 4: NumPy Basics: Arrays and Vectorized Computation

# ## The NumPy ndarray: A Multidimensional Array Object

# In[40]:

get_ipython().magic('matplotlib inline')


# In[1]:

import numpy as np


# In[4]:

from numpy.random import randn


# In[5]:

data = randn(2, 3)
data


# In[6]:

data * 10


# In[7]:

np.set_printoptions(precision=4, suppress=True)


# In[8]:

data


# In[9]:

data * 10


# In[10]:

data + data


# In[11]:

data.shape


# In[12]:

data.dtype


# ### Creating ndarrays

# In[13]:

data1 = [6, 7.5, 8, 0, 1]
data1


# In[14]:

arr1 = np.array(data1)
arr1


# In[15]:

data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2


# In[16]:

arr2.ndim


# In[17]:

arr2.shape


# In[18]:

arr2.dtype


# In[19]:

arr1.dtype


# In[20]:

np.zeros(10)


# In[21]:

np.zeros((3, 6))


# In[22]:

np.empty((2, 3, 2))


# In[24]:

np.arange(15)


# ### Data Types for ndarrays

# In[25]:

arr1 = np.array([1, 2, 3], dtype = np.float64)
arr1


# In[26]:

arr2 = np.array([1, 2, 3], dtype = np.int32)
arr2


# In[27]:

arr1.dtype


# In[28]:

arr2.dtype


# In[29]:

arr = np.array([1,2,3,4,5])
arr.dtype


# In[30]:

float_arr = arr.astype(np.float64)
float_arr.dtype


# In[31]:

arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr


# In[32]:

arr.astype(np.int32)


# In[33]:

numeric_string = np.array(['1.25', '-9.6', '42'], dtype=np.string_)


# In[34]:

numeric_string.astype(float)


# In[35]:

int_array = np.arange(10)


# In[36]:

calibers = np.array([.22, .270, .357, .380, .44, .50], dtype = np.float64)


# In[37]:

int_array.astype(calibers.dtype)


# In[38]:

empty_uint32 = np.empty(8, dtype = 'u4')


# In[39]:

empty_uint32


# ### Operations between Arrays and Scalars

# In[42]:

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr


# In[43]:

arr * arr


# In[44]:

arr - arr


# In[45]:

1 / arr 


# In[46]:

arr ** 0.5


# ### Basic Indexing and Slicing

# In[47]:

arr = np.arange(10)
arr


# In[48]:

arr[5]


# In[49]:

arr[5:8]


# In[50]:

arr[5:8] = 12


# In[51]:

arr


# In[52]:

arr_slice = arr[5:8]
arr_slice


# In[53]:

arr_slice[1] = 12345
arr_slice


# In[54]:

arr


# In[55]:

arr_slice[:] = 64
arr


# In[56]:

arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr2d


# In[57]:

arr2d[2]


# In[58]:

arr2d[0][2]


# In[59]:

arr2d[0, 2]


# In[61]:

arr3d = np.array([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
arr3d


# In[62]:

arr3d[0]


# In[63]:

old_values = arr3d[0].copy()


# In[64]:

arr3d[0] = 42
arr3d


# In[65]:

arr3d[0] = old_values
arr3d


# In[66]:

arr3d[1, 0]


# #### Indexing with slices

# In[67]:

arr[1:6]


# In[68]:

arr2d


# In[69]:

arr2d[:2]


# In[71]:

arr2d[:2, 1:]


# In[72]:

arr2d[1, :2]


# In[73]:

arr2d[2, :1]


# In[74]:

arr2d[:, :1]


# ### Boolean Indexing

# In[75]:

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[76]:

data = randn(7, 4)
data


# In[77]:

names == 'Bob'


# In[78]:

data[names == 'Bob']


# In[79]:

data[names == 'Bob', 3]


# In[80]:

names != 'Bob'


# In[81]:

data[-(names == 'Bob')]


# In[86]:

data[~(names == 'Bob')]


# In[87]:

mask = (names == 'Bob') | (names == 'Will')


# In[88]:

mask


# In[89]:

data[mask]


# In[90]:

data[data < 0] = 0
data


# In[91]:

data[names != 'Joe'] = 7
data


# ### Fancy indexing

# In[92]:

arr = np.empty((8, 4))
arr


# In[95]:

for i in range(8):
    arr[i] = i 
arr


# In[96]:

arr[[4, 3, 1, 6]]


# In[97]:

arr[[-3, -5, -7]]


# In[98]:

arr = np.arange(32).reshape((8, 4))
arr


# In[99]:

arr[[1, 5, 7, 2], [0, 3, 1, 2]]


# In[100]:

arr[[1,5,7,2]][:, [0, 3, 1, 2]]


# In[102]:

arr[np.ix_([1,5,7,2], [0,3,1,2])]


# In[103]:

np.ix_([1,5,7,2], [0,3,1,2])


# ### Transposing Arrays and Swapping Axes
