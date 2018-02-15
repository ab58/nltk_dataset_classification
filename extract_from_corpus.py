# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:47:34 2018

@author: arjunb
"""

from nltk.corpus import *
import numpy as np
import pandas as pd
from pandas import DataFrame
import sys, string, re


punct = re.compile("["+string.punctuation+"]")

def get_labeled_sents(corpus):
    
    if corpus == "inaugural":
        corpus = inaugural
    elif corpus == "reuters":
        corpus = reuters
    elif corpus == "gutenberg":
        corpus = gutenberg
    else:
        print("Invalid corpus: quitting")
        sys.exit()
        
    sent_strings = []
    label_vector = []
    for fileid in corpus.fileids():
        
        label = fileid[:-4]
        sents = corpus.sents(fileid)
        for sent in sents:
            sent = [word.lower() for word in sent if not punct.search(word)]
            sentence = " ".join(sent)[:]
            sent_strings.append(sentence)
            label_vector.append(label)
    
    #For holdout set, take out every 5th entry, beginning at 5th entry
    sent_strings_holdout = sent_strings[4::5]
    label_vector_holdout = label_vector[4::5]
    
    return sent_strings, label_vector, sent_strings_holdout, label_vector_holdout


def data_to_csv(sent_strings, label_vector, corpus_name, train_or_test):
    df = pd.DataFrame({"Utterance" : sent_strings, "Label" : label_vector})
    df = df[["Utterance", "Label"]]
    df.to_csv("corpus_labeled_"+corpus_name+"_"+train_or_test+".csv", encoding="utf-8", index=False)


def main():
    
    sent_strings, label_vector, sent_strings_holdout, label_vector_holdout = get_labeled_sents(sys.argv[1])
    
    #Convert both train and test sets to csv format
    data_to_csv(sent_strings, label_vector, sys.argv[1], "train")
    data_to_csv(sent_strings_holdout, label_vector_holdout, sys.argv[1], "test")
            
        
 
    
    
if __name__ == "__main__":
    main()