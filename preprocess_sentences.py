import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from clean_string import clean_string


stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()





def preprocess(sentence, remove_stopwords=False,lemmatize=True,stem=True):
    sentence = clean_string(str(sentence)).lower().split() # irrelevant characters removed
    # remove stopwords
    sentence2 = []
    for word in sentence:
        if (not word in stopwords) or not remove_stopwords:
            sentence2.append(word)
    del sentence
    # lemmatize
    if lemmatize:
        for i in range(len(sentence2)):
            sentence2[i] = lemmatizer.lemmatize(sentence2[i])
    # stem
    if stem:
        for i in range(len(sentence2)):
            sentence2[i] = stemmer.stem(sentence2[i])
    return " ".join(sentence2)


