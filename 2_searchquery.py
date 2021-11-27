#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from rank_bm25 import *
from gensim.parsing.preprocessing import STOPWORDS
from rank_bm25 import BM25Okapi
import pickle

def stopwords_removal(lst):
    global all_stopwords
    lst1=list()
    for str in lst:
        text_tokens = word_tokenize(str)
        tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
        str_t = "".join(tokens_without_sw)
#         print(tokens_without_sw)
        lst1.append(str_t)
    lst2 = [x.lower() for x in lst1 if x!="" and len(x)>2]
    return lst2


# In[2]:


stopWordsEng = stopwords.words('english')
nltkstopw = stopwords.words('english')
all_stopwords = STOPWORDS.union(set(nltkstopw))
ps = PorterStemmer() 


# In[3]:


with open('objs//bm25', 'rb') as handle:
    bm25 = pickle.load(handle)
    
with open('objs//tokenized_corpus', 'rb') as handle:
    tokenized_corpus = pickle.load(handle)

with open('objs//fpathdict', 'rb') as handle:
    fpathdict = pickle.load(handle)


# In[4]:


def getSearchResults(query):
    words = RegexpTokenizer(r'\w+').tokenize(query)
    wordswithoutstops = stopwords_removal(words)
    tokenized_query = [ps.stem(x) for x in wordswithoutstops]
    doc_scores = bm25.get_scores(tokenized_query)
    matchingdocs = np.argsort(doc_scores)[::-1][:20]
    resultslist = [fpathdict[x] for x in matchingdocs]
    return resultslist


# In[5]:


query = input ("Type your search query :")
resultslist = getSearchResults(query)
rank=1
for path in resultslist:
    print(rank," => ",path)
    rank+=1


# In[ ]:




