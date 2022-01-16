import multiprocessing

import gensim.models
import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
nltk.download('stopwords')

from gensim.models.word2vec import Word2Vec

from nltk.stem.porter import *

from sklearn import svm
from sklearn.model_selection import GridSearchCV

import joblib

cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
n_iterations = 1
n_exposures = 10  # 所有频数超过10的词语
window_size = 7
train_path = 'src/dataset/train.csv'
trial_path = 'src/dataset/trial.csv'

def loadData():

    train_dataset=pd.read_csv(train_path)
    trial_dataset=pd.read_csv(trial_path)
    combined=np.concatenate((train_dataset['content'],trial_dataset['content']))
    return combined

def tokenizer(sentencesList):
    '''
    Remove punctuation and stopwords, and convert words into lowercase
    :param sentencesList:
    :return:
    '''
    stemmer=PorterStemmer()
    tokenized_words=[]
    for sent in sentencesList:

        restr=r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        text_noURL=re.sub(restr," ", sent)
        text_noHashTag=re.sub("#"," ", text_noURL)
        text_noPunc=re.sub("[^a-zA-Z#]"," ", text_noHashTag)
        words=text_noPunc.lower().split() # cast word to lower and split them by space

        # remove stopwords
        stop_words=stopwords.words('english')
        words=[stemmer.stem(word) for word in words if word not in stop_words and len(word)>2 ]
        tokenized_words.append(words)
    return tokenized_words

def word2vec_train(tokenized_tweets):
    model= gensim.models.Word2Vec(
        vector_size =vocab_dim,
        window=10,
        min_count=2,
        workers=cpu_count
    )

    max_vocab_size = vocab_dim,
    min_count = n_exposures,
    window = window_size,
    workers = cpu_count

    model.build_vocab(tokenized_tweets)
    model.train(tokenized_tweets, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('src/model/Word2vec_model.pkl')

def fea_sentence(words:list):
    '''
    Caculate avg(sum (vector of words))
    The characteristics of a sentence are obtained by summing the word vectors of the words that appear in the sentence
    and then dividing by the number of words
    :param words:
    :return:
    '''
    # Create an array whose elements were 0. originally
    n0=np.array([0. for i in range(vocab_dim)], dtype=np.float32)

    for i in words:
        n0+=i
    fe=n0/len(words)
    fe=fe.tolist()
    return fe

def parse_dataset(x_data,word2vec):
    '''

    :param x_data:
    :param word2vec:
    :return:
    '''

    wordlist=list()
    xVec=[]
    for x in x_data:
        sentence=[]
        for word in x:
            if word in word2vec.wv.key_to_index:
                sentence.append(word2vec.wv.get_vector(word))
            else:
                sentence.append([0. for i in range(vocab_dim)])
        xVec.append(fea_sentence(sentence))

    xVec=np.array(xVec)

    return xVec

def getData(modelWord2vec):
    '''

    :param modelWord2vec:
    :return: x_train: text from train dataset; y_train: labels from train dataset; x_test: text from test dataset
    '''

    x_train=pd.read_csv(train_path,header=None, index_col=None)
    y_train=pd.read_csv(train_path,header=None, index_col=None)
    x_test=pd.read_csv(trial_path,header=None, index_col=None)

    y_train=np.array(y_train[4][1:].tolist())



    x_train=x_train[3][1:].tolist()
    x_train=tokenizer(x_train)
    x_train=parse_dataset(x_train,modelWord2vec)

    x_test = x_test[3][1:].tolist()
    x_test=tokenizer(x_test)
    x_test=parse_dataset(x_test,modelWord2vec)

    return x_train,y_train,x_test

def svm_train(x_train, y_train):
    # svr=svm.SVC(verbose=True)
    # parameters={
    #     'kernel':('linear','rbf'),
    #     'C':[0.5,1,2,4,6],
    #     'gamma':[0.125,0.25,0.5,1,2]
    # }
    # clf=GridSearchCV(svr,parameters,scoring='f1')
    # clf.fit(x_train, y_train,)
    # print('The best parameters are: ')
    # print(clf.best_params_)
    # # {'C': 0.5, 'gamma': 0.125, 'kernel': 'linear'}

    clf=svm.SVC(kernel='linear',C=0.5,gamma=0.125,verbose=True)
    clf.fit(x_train,y_train)

    joblib.dump(clf,'src/model/svm_word2vec.pkl')

if __name__ == '__main__':
    combined=loadData()
    tokenized_combined=tokenizer(combined)
    print(tokenized_combined[0])

    # train data model
    word2vec_train(tokenized_combined)

    # Load word2vec model
    word2ver=Word2Vec.load('src/model/Word2vec_model.pkl')
    print(word2ver.wv.index_to_key)
    print(word2ver.wv.similar_by_word('peopl'))
    x_train, y_train, x_test=getData(word2ver)

    print(y_train)

    svm_train(x_train,y_train)

    svm_model=joblib.load('src/model/svm_word2vec.pkl')

    y_pred=svm_model.predict(x_train)

    print(y_pred)





