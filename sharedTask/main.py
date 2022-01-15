import multiprocessing

import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords


from gensim.models.word2vec import Word2Vec


cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
n_iterations = 1
n_exposures = 10  # 所有频数超过10的词语
window_size = 7

def loadData():
    train_path='scr/dataset/train.csv'
    trial_path='scr/dataset/trial.csv'
    train_dataset=pd.read_csv(train_path)
    trial_dataset=pd.read_csv(trial_path)
    combined=np.concatenate((train_dataset['content'],trial_dataset['content']))
    return combined

def tokenizer(dataset):
    '''
    Remove punctuation and stopwords, and cast words to lowercase
    :param dataset:
    :return:
    '''

    tokenized_words=[]
    for sent in dataset:
        text_noPunc=re.sub("[^a-zA-Z]"," ", sent)
        words=text_noPunc.lower().split() # cast word to lower and split them by space

        # remove stopwords
        stop_words=stopwords.words('english')
        words=[word for word in words if word not in stop_words]
        tokenized_words.append(words)
    return tokenized_words

def word2vec_train(dataset):
    model= Word2Vec(
        max_vocab_size=vocab_dim,
        min_count=n_exposures,
        window=window_size,
        workers=cpu_count,
        iter=n_iterations
    )



if __name__ == '__main__':
    combined=loadData()
    tokenized_combined=tokenizer(combined)
    print(tokenized_combined)





