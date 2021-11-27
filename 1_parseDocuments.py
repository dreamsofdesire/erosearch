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
import re
from gensim.parsing.preprocessing import STOPWORDS
from rank_bm25 import BM25Okapi
import pickle

def stopwords_removal(lst):
    lst = [x.strip().lower() for x in lst]
    all_stopwords = STOPWORDS
    lst1=list()
    for str in lst:
        text_tokens = word_tokenize(str)
        tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
        str_t = "".join(tokens_without_sw)
#         print(tokens_without_sw)
        lst1.append(str_t)
    lst2 = [x.lower() for x in lst1 if x!="" and len(x)>2]
    return lst2


# In[2]:


# STOPWORDS


# In[3]:


pathToDocuments = 'stories'

ps = PorterStemmer() 

def flattenlist(nonflatlist):
    for item in nonflatlist:
        if not isinstance(item, (str,bytes)) and isinstance(item , collections.Iterable):
            yield from flattenlist(item)
        else:
            yield item


# In[4]:



def startPreprocessing():
    datadict ={}
    fpathdict = {}
    global pathToDocuments,stopWordsEng
    global fileListing,allTokens,titleTokens,otherTokens
    global docNumpairs,allDocumentsTokens
    pathListing = os.walk(pathToDocuments)
    indexv=0
    for root,dirlist,filelist in pathListing:
        for filename in filelist:            
            pathvar = os.path.join(root,filename)
            print(pathvar)
            with open(pathvar, 'r', encoding='latin-1') as infile:
                allLines  = infile.readlines()
            datadict[indexv]=allLines
            fpathdict[indexv]=pathvar
            indexv+=1
    return datadict,fpathdict


# In[5]:


datadict,fpathdict = startPreprocessing()


# In[6]:


tokenized_corpus = []
nonstemmed_tokenized_corpus = []
for docindex in datadict.keys():
    print(docindex, end=" ")
    doctext = " ".join(datadict[docindex])
    words = RegexpTokenizer(r'\w+').tokenize(doctext)
    # words = word_tokenize(doctext)
    wordswithoutstops = stopwords_removal(words)
    nonstemmed_tokenized_corpus.append(wordswithoutstops)
    stemmedtokens = [ps.stem(x) for x in wordswithoutstops]
    tokenized_corpus.append(stemmedtokens)
    


# In[7]:


bm25 = BM25Okapi(tokenized_corpus)


# In[13]:


with open('objs//bm25', 'wb') as handle:
    pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("exported objs//bm25")
    
with open('objs//tokenized_corpus', 'wb') as handle:
    pickle.dump(tokenized_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("exported objs//tokenized_corpus")
    
with open('objs//fpathdict', 'wb') as handle:
    pickle.dump(fpathdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("exported objs//fpathdict")
    
# with open('objs//nonstemmed_tokenized_corpus', 'wb') as handle:
#     pickle.dump(nonstemmed_tokenized_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("exported objs//nonstemmed_tokenized_corpus")


# In[9]:


# with open('objs//bm25', 'rb') as handle:
#     bm25 = pickle.load(handle)
    
# with open('objs//tokenized_corpus', 'rb') as handle:
#     tokenized_corpus = pickle.load(handle)

# with open('objs//fpathdict', 'rb') as handle:
#     fpathdict = pickle.load(handle)

# with open('objs//nonstemmed_tokenized_corpus', 'rb') as handle:
#     nonstemmed_tokenized_corpus = pickle.load(handle)


# In[14]:


query = "brother"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)


# In[15]:


matchingdocs = np.argsort(doc_scores)[::-1][:10]
matchingdocs


# In[16]:


resultslist = [fpathdict[x] for x in matchingdocs]
resultslist


# In[ ]:





# In[ ]:




