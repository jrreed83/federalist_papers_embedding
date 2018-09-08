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
        with open(PATH, 'rb') as fid:
            data = pickle.load(fid)
        article_list = data['articles']

        self.word2id, self.id2word = get_vocabulary(article_list)        
        contexts, centers = get_training_data(article_list, self.word2id)

        self.contexts, self.centers = contexts, centers

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
    words = [w for w in words if len(w) > 0 and w.isalpha()]
    
    # Convert to a list for easy processing
    return list(words)

def word_frequencies(article_list):
    '''
    Iterate over all articles, aggregating the vocabulary as we go
    '''
    histogram = {}
    for article in article_list:
        
        # Get a list of words for the article
        words = process_article(article)
        
        # Grab the article vocabulary
        for word in words:
            if word not in histogram.keys():
                histogram[word] = 1
            else:
                histogram[word] += 1
    return histogram

def get_vocabulary(article_list):
    '''
    Iterate over all articles, aggregating the vocabulary as we go
    '''
    histogram = word_frequencies(article_list)
    words = [w for w in histogram]
    word2id = {w:i for i, w in enumerate(words)}
    id2word = {i:w for i, w in enumerate(words)}
    
    return word2id, id2word

def get_windows(words, win_size=None):
    '''
    The input words array 
    '''
    center_id = int(win_size / 2)
    windows = []
    for k in range(2, len(words)-2):
        window = [words[k-2], words[k-1], words[k], words[k+1], words[k+2]]
        center = window[2]
        context = [window[0], window[1], window[3], window[4]]
        windows.append((context, center))
    return windows

def get_training_data(article_list, word2id):

    for article in article_list:
        # Convert each article to a list of preprocessed words
        words = process_article(article)

        # Convert words to ids based on vocabulary 
        word_ids = [word2id[w] for w in words]

        # Get the windows for this particular article
        windows = get_windows(word_ids)
            
        # Write the context windows to a file, note that
        # the center word comes last
        contexts = []
        centers = []

        for context, center in windows:
            contexts.append(context)
            centers.append(center)
    return contexts, centers

def fetch_federalist_papers():
    doc_path = 'federalist_papers.txt'
    # Check if file already exists in cache
    if not path.isfile(doc_path): 
        # If it isn't, pull it down and write it to a specific file
        response = requests.get('http://www.gutenberg.org/cache/epub/18/pg18.txt')
        with open(doc_path,'w') as fid:
            fid.write(response.text)            
    
    # Define various states
    start_state = 0
    author_state = 1
    greeting_state = 2
    content_state = 3

    # And declare the initial state
    state = start_state

    # The possible author names
    author_names = ['JAY', 'HAMILTON', 'MADISON', 'HAMILTON OR MADISON', 'HAMILTON AND MADISON']
    
    greeting = 'To the People of the State of New York'
    pattern = re.compile('FEDERALIST(\s+|.\s+)No.\s+(\d+)')

    # Data structure we are building
    data = {
        'author_name': [],
        'author_id': [],
        'article_id': [],
        'articles': []
    }
    
    content = []
    
    with open(doc_path,'r') as fid:
        for l in fid:    
            # Strip away white space
            line = l.strip()

            # Only process non-empty lines
            if len(line) > 0:
                if state == start_state:
                    # Looking the meta-data associated with the article
                    m = pattern.match(line)
                    if m is not None:
                        article_number = m.group(2)
                        data['article_id'].append(article_number)
                        state = author_state
                elif state == author_state:
                    # Looking for the author name
                    if line in author_names:
                        author = line
                        author_id = author_names.index(author)
                        data['author_name'].append(line)
                        data['author_id'].append(author_id)
                        state = greeting_state
                elif state == greeting_state:
                    # Looking for the greeting, this is the same for each article
                    if greeting in line:
                        state = content_state
                elif state == content_state:
                    if 'PUBLIUS' in line:
                        data['articles'].append(' '.join(content))
                        temp = []
                        state = start_state
                    else:
                        content.append(line)
    return data