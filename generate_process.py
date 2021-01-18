#!/usr/bin/env python
# coding: utf-8

# # This script
# p.56, generate process of topic model

# # import modules

# In[ ]:


import numpy as np
from scipy.stats import dirichlet
import pandas as pd

from collections import OrderedDict
np.random.seed(0)


# ## assist functions

# In[ ]:


def sample_dirichlet(betas=None, beta =None, size = 1):
    if betas is not None:
        return dirichlet.rvs(betas).reshape(-1)
    elif size == 1:
        return dirichlet.rvs(beta)
    else:
        return dirichlet.rvs([beta]*size).reshape(-1)
    raise RuntimeError("do not match any supposed args ")
    return None

def sample_categorical(ps, vals=None):
    if vals is None:
        vals = np.arange(len(ps))
    realized = np.random.choice(vals,1,replace = True,p=ps)[0]
    return realized


# # generate docs

# In[ ]:


def generate_topic_model_docs(word_size = 5, topic_size = 3, doc_size = 20,each_doc_word_size =30,
                             topic_word_para =1,doc_topic_para = 2):

    # topic word realized
    phis = np.zeros((topic_size, word_size))
    for ki in range(topic_size):
        phi_k = sample_dirichlet(beta = topic_word_para, size = word_size)
        phis[ki] = phi_k
    # print("phis:", phis)

    # doc word realized
    doc_thetas = np.zeros((doc_size, topic_size))
    doc_topic_word_dict = OrderedDict()
    for di in range(doc_size):
    #     print(di,"----------")
        theta_d = sample_dirichlet(beta = doc_topic_para, size = topic_size)
        doc_thetas[di, :] =theta_d

        topic_words_list = []
    #     print("theta_d:", theta_d)
        for wi in range(each_doc_word_size):
    #         print("--")
            topic_d = sample_categorical(theta_d)
    #         print("topic_d:", topic_d)
    #         print("phis[topic_d]:", phis[topic_d])
            word_d = sample_categorical(phis[topic_d])
    #         print("word_d:", word_d)
            topic_words_list.append(OrderedDict({"topic":topic_d, "word":word_d}))    
        # output reshape
        topic_words_df = pd.DataFrame(topic_words_list)
        topic_words_df.index = ["word"+str(i) for i in range(each_doc_word_size)]
        doc_topic_word_dict["doc"+str(di)] = topic_words_df

    # output reshape
    phi_df = pd.DataFrame(phis).T
    phi_df.index = ["word"+str(wi) for wi in range(word_size)]
    phi_df.columns = ["topic"+str(ti) for ti in range(topic_size)]

    doc_theta_df = pd.DataFrame(doc_thetas)
    doc_theta_df.index = ["doc"+str(i) for i in range(doc_size)]
    doc_theta_df.columns = ["topic"+str(i) for i in range(topic_size)]

    return phi_df, doc_theta_df, doc_topic_word_dict


# # main

# In[ ]:


if __name__ == "__main__":
    word_size = 5
    topic_size = 3
    doc_size = 20
    each_doc_word_size = 30
    topic_word_para = 1
    doc_topic_para = 1

    phi_df, doc_theta_df, doc_topic_word_dict = generate_topic_model_docs(word_size = 5, topic_size = 3, doc_size = 20,each_doc_word_size =30,
                                 topic_word_para =1,doc_topic_para = 2)
    print("phi_df:")
    display(phi_df)
    print("doc_theta_df:")
    display(doc_theta_df)
    
    for doc in doc_topic_word_dict.keys():
        print(doc)
        display(doc_topic_word_dict[doc])

