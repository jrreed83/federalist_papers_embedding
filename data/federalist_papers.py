import torch.utils.data as data  

import re
import requests
import numpy as np
from os import path
import string
import pickle

PATH = 'federalist_papers.txt'

class FederalistPapers(data.Dataset):
    def __init__(self):
        contexts, centers = get_training_data()
        self.contexts, self.centers = contexts, centers
        self.word2id, self.id2word = get_vocabulary()
    def __len__(self):
        return len(self.centers)
    def __getitem__(self, i):
        return np.array(self.contexts[i]), self.centers[i]


def process_article(article):
    '''
    Convert a single article into a list of words
    '''
    # Split based on blank space
    words = article.split()
    
    # Convert all words to lower case
    words = [w.lower() for w in words]
    
    # Remove any punctuation from the beginning and end of words.
    # Note that if this is applied to a string containing only puncutation,
    # we would be left with an empty string.  
    words = [w.strip(string.punctuation) for w in words]
    
    # Based on the previous remarks, remove any empty strings and numeric
    # strings by keeping only those words that are alphabetic and have a length
    # greater than 0.
    words = [w for w in words if w.len() > 0 and w.isalpha()]
    
    # Convert to a list for easy processing
    return list(words)

def word_frequencies(articles):
    '''
    Iterate over all articles, aggregating the vocabulary as we go
    '''
    histogram = {}
    for a in articles:
        
        # Get a list of words for the article
        words = process_article(a)
        
        # Grab the article vocabulary
        for word in words:
            if word not in histogram.keys():
                histogram[word] = 1
            else:
                histogram[word] += 1
    return histogram


def get_vocabulary(histogram):
    '''
    Iterate over all articles, aggregating the vocabulary as we go
    '''
    vocabulary = []
    for word, count in histogram.items():
        vocabulary.append(word)
    return vocabulary    

def article_windows(words):
    '''
    The input words array 
    '''
    windows = []
    for k in range(2, len(words)-2):
        window = [words[k-2], words[k-1], words[k], words[k+1], words[k+2]]
        center = window[2]
        context = [window[0], window[1], window[3], window[4]]
        windows.append((context, center))
    return windows

def get_training_data():
    # Fetch the articles of confederation
    with open(path, 'rb') as fid:
        data = pickle.load(fid)

    for article in data['articles']:
        # Convert each article to a list of preprocessed words
        words = process_article(article)
            
        # Get the windows for this particular article
        windows = get_windows(words)
            
        # Write the context windows to a file, note that
        # the center word comes last
        contexts = []
        centers = []

        for context, center in windows:
            contexts.append(context)
            centers.append(center)
    return contexts, centers