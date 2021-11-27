"# erosearch"   
A simple python based utility to build index of locally stored erotica stories or any text files and then search in them to obtain top 20 results for specific terms in query. Have included a corpus of around 300+ erotica stories obtained from literotica and other sites..  

To search for erotica with query terms using the existing index built, only the file 2_searchquery.ipynb or 2_searchquery.py needs to be used.  
To add new text stories from literotica or other sites, they can be added by adding their txt files in the stories directory (subdirectory can be created) and new index would be required to be built by running 1_parseDocuments.ipynb or 1_parseDocuments.py  
It uses nltk, gensim and rank-bm25 python libraries to be installed prior to use, using an anaconda installation is and then installing them over that is recommended.  