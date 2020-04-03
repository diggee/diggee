# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:08:03 2020

@author: admin
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from nltk.corpus import stopwords
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, GlobalAveragePooling1D, Dropout, GlobalMaxPool1D, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% #%% reading data

def get_data():
    full_train_data = pd.read_csv('train.csv')
    full_test_data = pd.read_csv('test.csv')

    # there are no null values in either the train set or the test set.
    print('Total null values in the train set - '+str(full_train_data.isnull().sum().sum()))
    print('Total null values in the test set - '+str(full_test_data.isnull().sum().sum()))
    return full_train_data, full_test_data

#%% preprocessing data

def clean_data(full_train_data, full_test_data):

    def comment_clean(comment):
        # convert every tweet to lower case
        comment = ''.join([j.lower() for j in comment])
        # remove website links
        comment = re.sub('www.|https://|http://|.com|t.co/','',comment)
        # remove all punctuation
        comment = ''.join([j for j in comment if j not in string.punctuation])
        # remove all digits
        comment = ''.join([j for j in comment if j not in string.digits])
        # remove stopwords
        comment = ' '.join([j for j in comment.split() if j not in stopwords.words('english')])
        # remove non ASCII characters
        comment = ''.join([j for j in comment if ord(j) < 128])
        return comment

    full_train_data['comment_text'] = full_train_data['comment_text'].apply(lambda x: comment_clean(x))
    full_test_data['comment_text'] = full_test_data['comment_text'].apply(lambda x: comment_clean(x))
    return full_train_data, full_test_data
    
#%% tweet tokenization

def tokenization(vocab_size, oov_tok, train_tweets, test_tweets, trunc_type, padding_type, max_length):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
    tokenizer.fit_on_texts(train_tweets)
    train_sequences = tokenizer.texts_to_sequences(train_tweets)
    train_padded = pad_sequences(train_sequences, maxlen = max_length, truncating = trunc_type, padding = padding_type)
    test_sequences = tokenizer.texts_to_sequences(test_tweets)
    test_padded = pad_sequences(test_sequences, maxlen = max_length, truncating = trunc_type, padding = padding_type)
    return train_padded, test_padded, tokenizer

#%% neural network

def neural_network(vocab_size, embedding_dim, max_length, train_padded, train_labels, validation_frac, num_epochs):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length = max_length))
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    # model.add(LSTM(60, return_sequences = True))
    # model.add(GlobalMaxPool1D())
    # model.add(Dropout(0.1))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(train_padded, train_labels, epochs = num_epochs, verbose = 2, validation_split = validation_frac, callbacks = [checkpoint])
    return model, history

#%% plotting data

def make_plots(history, col):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(col)
    plt.legend(['Training', 'Validation'])

#%% main()

if __name__ == '__main__':

    vocab_size = 15000
    embedding_dim = 128
    max_length = 200
    trunc_type='post'
    padding_type='post'
    oov_tok = '<OOV>'
    num_epochs = 5
    validation_frac = 0.1

    full_train_data, full_test_data = get_data()
    full_train_data, full_test_data = clean_data(full_train_data, full_test_data)
    full_train_data = shuffle(full_train_data)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    toxicity = full_train_data[categories]    
    train_padded, test_padded, tokenizer = tokenization(vocab_size, oov_tok, full_train_data['comment_text'], full_test_data['comment_text'], trunc_type, padding_type, max_length)
    df = pd.DataFrame({'id':full_test_data.id})
    for col in categories:
        model, history = neural_network(vocab_size, embedding_dim, max_length, train_padded, toxicity[col], validation_frac, num_epochs)
        make_plots(history, col)
        model.load_weights('model.h5')
        test_pred = model.predict_classes(test_padded, verbose = 2)
        df[col] = test_pred.reshape(test_pred.shape[0],)
    df.to_csv('prediction.csv', index = False)
