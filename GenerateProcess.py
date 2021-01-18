#!/usr/bin/env python
# coding: utf-8

# # This script
# p.56, generate process of topic model

# # import modules

# In[1]:


import numpy as np


# In[3]:


# word size 
V = 10
# topic size
K = 3

beta = 0.1
k = 1
phi_k = np.random.dirichlet(beta,V)


# In[8]:


s = np.random.dirichlet((1), 20)
s

