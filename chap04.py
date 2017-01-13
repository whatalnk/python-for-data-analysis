
# coding: utf-8

# # Chapter 4: NumPy Basics: Arrays and Vectorized Computation

# ## The NumPy ndarray: A Multidimensional Array Object

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
from numpy.random import randn


# In[3]:

np.set_printoptions(precision=4, suppress=True)


# In[5]:

data = randn(2, 3)
data


# In[6]:

data * 10


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

# In[4]:

arr = np.arange(15).reshape((3, 5))


# In[5]:

arr


# In[7]:

arr.T


# In[8]:

arr = randn(6, 3)


# In[9]:

np.dot(arr.T, arr)


# In[10]:

arr = np.arange(16).reshape((2, 2, 4))
arr


# In[11]:

arr.transpose((1, 0, 2))


# In[12]:

arr.swapaxes(1, 2)


# ## Universal Functions: Fast Element-wise Array Functions

# In[13]:

arr = np.arange(10)


# In[15]:

arr


# In[16]:

np.sqrt(arr)


# In[17]:

np.exp(arr)


# In[21]:

x = randn(8)
x


# In[22]:

y = randn(8)
y


# In[23]:

np.maximum(x, y)


# In[24]:

arr = randn(7) * 5
arr


# In[25]:

np.modf(arr)


# ## Data Processing Using Arrays

# In[26]:

points = np.arange(-5, 5, 0.01)


# In[27]:

xs, ys = np.meshgrid(points, points)
ys


# In[28]:

import matplotlib.pyplot as plt


# In[29]:

z = np.sqrt(xs ** 2 + ys ** 2)


# In[30]:

z


# In[33]:

plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")


# ### Expressing Conditional Logic as Array Operations

# In[34]:

xarr = np.arange(1.1, 1.6, 0.1)
xarr


# In[35]:

yarr = np.arange(2.1, 2.6, 0.1)
yarr


# In[36]:

cond = np.array([True, False, True, True, False])


# In[38]:

result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result


# In[39]:

result = np.where(cond, xarr, yarr)
result


# In[40]:

arr = randn(4, 4)
arr


# In[41]:

np.where(arr > 0, 2, -2)


# In[42]:

np.where(arr > 0, 2, arr)


# In[ ]:

result = []
for i in range(n):
    if cond1[i] and cond2[i]:
        result.append(0)
    elif cond1[i]:
        result.append(1)
    elif cond2[i]:
        result.append(2)
    else:
        result.append(3)


# In[ ]:

np.where(cond1 & cond2, 0, 
        np.where(cond1, 1, 
                np.where(cond2, 2, 3)))


# In[ ]:

result = 1 * (cond1 & -cond2) + 2 * (cond2 & -cond1) + 3 * -(cond1 | cond2)


# ### Mathematical and Statistical Methods

# In[45]:

arr = randn(5, 4)
arr


# In[46]:

arr.mean()


# In[47]:

np.mean(arr)


# In[48]:

arr.sum()


# * row

# In[49]:

arr.mean(axis=1)


# * column

# In[52]:

arr.mean(0)


# In[50]:

arr.sum(0)


# In[56]:

arr = np.arange(0, 9, 1).reshape((3, 3))
arr


# In[57]:

arr.cumsum(0)


# In[59]:

arr.cumprod(1)


# ### Methods for Boolean Arrays

# In[60]:

arr = randn(100)
(arr > 0).sum()


# In[61]:

bools = np.array([False, False, True, False])
bools.any()


# In[62]:

bools.all()


# ### Sorting

# In[63]:

arr = randn(8)
arr


# In[64]:

arr.sort()


# In[65]:

arr


# In[67]:

arr = randn(5, 3)
arr


# In[69]:

arr.sort(0)
arr


# In[70]:

arr = randn(8)
arr


# In[71]:

np.sort(arr)


# In[72]:

arr


# In[73]:

large_arr = randn(1000)


# In[74]:

large_arr.sort()


# In[75]:

large_arr[int(0.05 * len(large_arr))]


# ### Unique and Other Set Logic

# In[76]:

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names


# In[77]:

np.unique(names)


# In[78]:

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)


# In[79]:

sorted(set(names))


# In[80]:

values = np.array([6, 0, 0, 3, 2, 5, 6])


# In[81]:

np.in1d(values, [2, 3, 6])


# ## File Input and Output with Arrays

# ### Storing Arrays on Disk in Binary Format

# In[82]:

arr = np.arange(10)
np.save('some_array', arr)


# In[83]:

arr


# In[84]:

np.load('some_array.npy')


# In[85]:

np.savez('array_archive', a = arr, b = arr)


# In[86]:

arch = np.load('array_archive.npz')


# In[88]:

arch['a']


# In[89]:

arch


# In[92]:

arch.keys()


# ### Saving and Loading Text Files

# In[ ]:

# not work
get_ipython().system('cat array_ex.txt')


# In[5]:

arr = np.loadtxt('array_ex.txt', delimiter=",")
arr


# ## Linear Algebra

# In[4]:

x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])


# In[5]:

x


# In[6]:

y


# In[7]:

x.dot(y)


# In[8]:

y.dot(x)


# In[9]:

np.dot(x, np.ones(3))


# In[10]:

np.ones(3)


# In[11]:

from numpy.linalg import inv, qr


# In[12]:

X = randn(5, 5)


# In[13]:

mat = X.T.dot(X)


# In[14]:

inv(mat)


# In[15]:

mat.dot(inv(mat))


# In[16]:

q, r = qr(mat)
q


# In[17]:

r


# ## Random Number Generation

# In[18]:

samples = np.random.normal(size = (4,4))
samples


# In[19]:

from random import normalvariate


# In[20]:

N = 1000000


# In[22]:

get_ipython().magic('timeit samples = [normalvariate(0, 1) for _ in range(N)]')


# In[23]:

get_ipython().magic('timeit np.random.normal(size=N)')


# ## Example: Random Walks

# * built in Python random

# In[35]:

import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)


# In[31]:

import matplotlib.pyplot as plt


# In[36]:

plt.plot(walk[:100])
plt.title("Random walk with +1/-1 steps")


# * np.random

# In[37]:

nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()


# In[38]:

plt.plot(walk[:100])
plt.title("Random walk with +1/-1 steps")


# In[28]:

walk.min()


# In[29]:

walk.max()


# * 最初に原点から10離れるのはいつか

# In[39]:

(np.abs(walk) >= 10).argmax()


# ### Simulating Many Random Walks at Once

# In[40]:

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size = (nwalks, nsteps))
draws


# In[41]:

steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks


# In[42]:

walks.max()


# In[43]:

walks.min()


# In[44]:

hits30 = (np.abs(walks) >= 30).any(1)
hits30


# In[45]:

hits30.sum()


# In[46]:

crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
crossing_times


# In[47]:

crossing_times.mean()


# In[48]:

steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
steps

