# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:07:34 2020

@author: Wenxi
"""

import re
from collections import Counter
import numpy as np
import json
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle

def preprocess(text:'str; the text to preprocess',
               ignore_word_list:'list; the list of noise words (lower) that will be removed from input text'):
    '''
    
    Returns
    -------
    words : list of words after preprocess: lower, with token, noise words removed
   
    '''

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    # word_counts = Counter(words)
    if len(ignore_word_list)>0:
        trimmed_words = [word for word in words if word not in ignore_word_list]
    else:
        trimmed_words = words
    return trimmed_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def calculate_drop_prob(int_words, threshold):
    word_counts = Counter(int_words)
    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}    
    return p_drop


def load_curpus(file_path):
    with open(file_path,'r',errors='ignore') as f:
        json_list = list(f)

    total_response = []
    total_context = []
    total_input = []
    total_label = []

    
    for json_str in json_list:
        result = json.loads(json_str)
        total_label.append(result['label'])
        total_input.extend(preprocess(result['response'],[]))
        total_response.append(preprocess(result['response'],[]))
        context = []
        context_item = []
        for item in result['context']:
            context += preprocess(item, [])  
            context_item.append(preprocess(item, []))
        total_input.extend(context)
        total_context.append(context_item)
        
    return total_input, total_response, total_context, total_label

def get_int_from_word(words, vocab_to_int):
    output = []
    for word in words:
        try:
            output.append(vocab_to_int[word])
        except KeyError:
            continue    
    return output

def load_response_context_label(file_path, label_orig, vocab_to_int):
    with open(file_path,'r',errors='ignore') as f:
        json_list = list(f)

    total_label = []
    total_response = []
    total_context = []
    for json_str in json_list:
        result = json.loads(json_str)
        
        total_label.append(label_orig.index(result['label']))
        
        total_response.append(get_int_from_word(preprocess(result['response'], []), vocab_to_int))
        
        context = []
        result_context = result['context'][::-1]
        for item in result_context:
            context += get_int_from_word(preprocess(item, []), vocab_to_int)
        total_context.append(context)
        
    return total_label, total_response, total_context

def get_remove_words(total_input,p_drop, remove_p_threshold, remove_count_threshold):
    count_word = Counter(total_input)
    input_word = list(set(total_input))
    remove_words = [key for key, item in p_drop.items() if item > remove_p_threshold]
    count_remove_by_prob = len(remove_words)
    print(f'remove {count_remove_by_prob} because of drop probability > threshold of {remove_p_threshold}\n')
    remove_words.extend([word for word in input_word if count_word[word]<remove_count_threshold])
    print(f'remove {len(remove_words) - count_remove_by_prob} because of count < threshold of {remove_count_threshold}\n')
    return remove_words


def pad_features(int_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    
    # getting the correct rows x cols shape
    features = np.zeros((len(int_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(int_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


def get_training_validation_test(training_input, training_label, training_frac, validation_frac, random_state=1234):
    
    ## split data into training, validation, and test data (features and labels, x and y)
    training_input, training_label = shuffle(training_input,training_label, random_state = random_state)
    split_idx = int(len(training_input)*training_frac)
    train_x, remaining_x = training_input[:split_idx], training_input[split_idx:]
    train_y, remaining_y = training_label[:split_idx], training_label[split_idx:]
    
    test_idx = int(len(remaining_x)*validation_frac)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    
    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape), 
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))
    
    return train_x, val_x, test_x, train_y, val_y, test_y

def get_data_loaders(batch_size, train_x, val_x, test_x, train_y, val_y, test_y):

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    if val_x is not None:
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    else:
        valid_loader = None
    
    if test_x is not None:
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    else:
        test_loader = None
    
    return train_loader, valid_loader, test_loader

def get_test_loader(test_x, test_y):
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    test_loader = DataLoader(test_data, shuffle=False, batch_size=len(test_x))    
    return test_loader

def combine_train_features(training_response, training_context, pad_length):
    total_train = [a+b for a, b in zip(training_response, training_context)]
    train_pad =  pad_features(total_train, pad_length)
    return train_pad

def get_features_pad(file_path,label_orig,vocab_to_int, pad_length):
    
    training_label, training_response, training_context = load_response_context_label(file_path, label_orig, vocab_to_int)
    train_pad = combine_train_features(training_response, training_context, pad_length)
    
    return train_pad, training_label