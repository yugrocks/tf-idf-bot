from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from  sklearn.metrics.pairwise import cosine_similarity
from preprocess_sentences import *
import numpy as np


# import data
data = pd.read_csv("data/chat.csv").iloc[:,:].values

preprocessed_messages = [preprocess(msg) for msg in data[:,0]] # preprocess by lemmatizing and stemming
# no need to filter stopwords

# now converting messages to tfidf weights
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(preprocessed_messages).toarray()

while True:
    message = input("SAY:- ")
    if message == "q":
        break
    # preprocess query
    query = preprocess(message)
    # find the tf-idf weights for query
    query_vec = tfidf.transform([query,]).toarray()
    # now find similarity
    sims = cosine_similarity(query_vec, X)  # shape = (1 , len(X))
    index = np.argmax(sims[0])
    print(data[:,1][index])
    
    
    
    
    
