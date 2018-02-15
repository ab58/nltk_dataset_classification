# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 23:16:15 2018

@author: arjunb
"""

import sys, re, time
import numpy as np
import pandas as pd
from sklearn import linear_model, naive_bayes, neighbors, svm
from sklearn.feature_extraction import text
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


def get_uniq_ngrams(X_train, n):
    
    uniq_ngrams = set()
    
    for utterance in X_train:
        
        utterance = utterance.lower()
        utt_words = utterance.split(" ")
        
        if len(utt_words) >= n:
            for j in range(len(utt_words)-n+1):
                ngram = " ".join(utt_words[j:j+n])
                uniq_ngrams.add(ngram)
            
    return list(uniq_ngrams)


#This will convert vectors containing text to numbers
def convert_data(X, y, vectorizer, fit):
    
    #First convert X into a one-hot matrix containing each utterance
    #represented by an array of length number of total ngrams
    
    if (fit):
        X_converted = vectorizer.fit_transform(X)
    else:
        X_converted = vectorizer.transform(X)
    
    #Convert y from string class labels to integers
    y_converted = np.zeros((len(y)), dtype=np.int32)
    classes = list(np.unique(y))
    
    for i in range(len(y)):
        label = y[i]
        y_converted[i] = classes.index(label) + 1   
    
    #Randomly shuffle X and y in same order
    seed = np.arange(X_converted.shape[0])
    np.random.shuffle(seed)
    X_converted = X_converted[seed]
    y_converted = y_converted[seed]
            
    return X_converted, y_converted
        
def remove_stopwords(X):
    
    for i in range(len(X)):
        utt = X[i]
        X[i] = " ".join([w for w in utt.split() if w not in stopwords.words("english")])

def load_text_data(training_file, test_file, vectorizer):
    
    training_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)
    
    X_train = training_data["Utterance"].values
    X_train = np.array([utt.lower() for utt in X_train if type(utt) is str])
    #remove_stopwords(X_train)
    y_train = training_data["Label"].values
    
    X_test = test_data["Utterance"].values
    X_test = np.array([utt.lower() for utt in X_test if type(utt) is str])
    #remove_stopwords(X_test)
    y_test = test_data["Label"].values
            
    X_train, y_train = convert_data(X_train, y_train, vectorizer, True)
    X_test, y_test = convert_data(X_test, y_test, vectorizer, False)
        
    return X_train, y_train, X_test, y_test


def model_results(model_choice, X_train, y_train, X_test, y_test):
    
    utt_clf = model_choice
    utt_clf.fit(X_train, y_train)
    
    utt_train_pred = utt_clf.predict(X_train)
    utt_test_pred = utt_clf.predict(X_test)
    
    utt_train_acc = np.mean(utt_train_pred == y_train)
    print("\nTRAINING ACCURACY: " + str(utt_train_acc))   
    utt_test_acc = np.mean(utt_test_pred == y_test)
    print("\nTEST ACCURACY: " + str(utt_test_acc))


def main():
    
    tic = time.time()
    vectorizer = text.CountVectorizer(ngram_range=(1,1))
    X_train, y_train, X_test, y_test = load_text_data(sys.argv[1], sys.argv[2], vectorizer)
    
    print("\nLOGISTIC REGRESSION CLASSIFIER")
    model_results(linear_model.LogisticRegression(), X_train, y_train, X_test, y_test)
    
    print("\n\nNAIVE BAYES CLASSIFIER")
    model_results(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)

    #print("\n\nK-NEIGHBORS CLASSIFIER")
    #model_results(neighbors.KNeighborsClassifier(), X_train, y_train, X_test, y_test)
    
    print("\n\nLINEAR SVC CLASSIFIER")
    model_results(svm.LinearSVC(), X_train, y_train, X_test, y_test)

    toc = time.time()
    print("\n" + str(int((toc-tic) // 60)) + "m " + str(int(toc-tic) % 60) + "s")
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    