#!/usr/bin/env python
# coding: utf-8

# # This script
# Check dirichlet distribution, before understanding topic model

# # Import modules

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Bernoulli
# Fliped coin is up or down?<br>
# Coin's up rate is up_rate.

# In[ ]:


from scipy.stats import bernoulli
test_size = 30
up_rate = 0.3

# samping
rs = bernoulli.rvs(up_rate, size=test_size)

plt.scatter(np.arange(len(rs)),rs)
print(np.mean(rs))


# # Binomial
# How many times are coins uped, after fliping coin trial_size times.<br>
# Coin's up rate is up_rate.

# In[ ]:


from scipy.stats import binom
trial_size = 10
up_rate = 0.3
test_size = 10

# sample
rs = binom.rvs(trial_size, up_rate, size=test_size)

plt.scatter(np.arange(len(rs)),rs)
print(np.mean(rs))


# # Categorical
# Which number is realised, after rolling a dice?<br>
# Dice's each realized rate is dice_rates.

# In[ ]:


dice_vals   = np.arange(4)
dice_rates = np.array([0.1,0.2,0.3,0.4])
test_size    = 100

rs = np.random.choice(dice_vals,test_size,replace = True,p=dice_rates)

plt.scatter(np.arange(len(rs)),rs)
for dice_val in dice_vals:
    print(dice_val,":",np.mean(rs == dice_val))


# # Multinomial
# How many times are each dice's value is realised, after rolling dice trial_size times?<br>
# Dice's rates of each numbers are dice_rates.

# In[ ]:


from scipy.stats import multinomial
trial_size = 10
dice_rates = np.array([0.1,0.2,0.3,0.4])
test_size = 20

rs = multinomial.rvs(trial_size,dice_rates,test_size)
rs_df = pd.DataFrame(rs).T
display(rs_df)

for i in range(len(rs_df)):
    plt.bar(rs_df.columns, rs_df.iloc[i], bottom=rs_df.iloc[:i].sum())
plt.xlabel('trial')
plt.ylabel('realized count')
plt.legend(rs_df.index)

for val in range(len(dice_rates)):
    print(val,":", np.mean(rs[:,val]))


# # Beta
# Distribution of Bern distribution's parameter.<br>
# 0<=Beta<=1

# In[ ]:


from scipy.stats import beta

a_s = np.array([0.1,0.5,1,2])
b_s = np.array([0.1,0.5,1,2])
xs = np.linspace(0,1,100)
plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.4,hspace=0.5)
pi = 1
for a in a_s:
    for b in b_s:
        plt.subplot(4,4,pi)
        plt.title(f"a:{a}, b:{b}")
        plt.plot(xs, beta.pdf(xs, a =a, b = b))
        pi += 1
#         plt.pause(0.1)


# # Dirichlet
# Distribution of categorical distribution's parameter.<br>
# 0<=Dirichlet<=1, sum(Dirichlet) = 1

# In[ ]:


from scipy.stats import dirichlet
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
xs = np.linspace(0,1,20) + 1e-3
xyzs = []
for x, y, z in itertools.product(xs,xs,xs):
    sum_xyz = x+y+z
    if sum_xyz == 0:
        continue
    x /= sum_xyz
    y /= sum_xyz
    z /= sum_xyz
    if [x,y,z] not in xyzs:
        xyzs.append([x,y,z])
        
plt.figure(figsize=(10,10))
betas = np.array([0.1,0.2,0.3])
ps = []
for xyz in xyzs:
    p = dirichlet.pdf(xyz,betas)
    ps.append(p)
xyzs = np.array(xyzs)
ps = np.array(ps)
ps = np.log(ps)
ps -= min(ps)
ps /= max(ps)
colors = []w
for p in ps:
    colors.append((p,0,0))
plt.scatter(xyzs[:, 0],xyzs[:, 1],color = colors, s=10)

