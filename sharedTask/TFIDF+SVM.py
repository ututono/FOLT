import multiprocessing

import pandas as pd
import numpy as np
import nltk
import re

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import joblib

from nltk.corpus import stopwords
nltk.download('stopwords')



def tokenizer(sentencesList):
    '''
    Remove punctuation and stopwords, and convert words into lowercase
    :param sentencesList:
    :return:
    '''

    tokenized_words=[]
    for sent in sentencesList:
        text_noPunc=re.sub("[^a-zA-Z]"," ", sent)
        words=text_noPunc.lower().split() # cast word to lower and split them by space

        # remove stopwords
        stop_words=stopwords.words('english')
        words=[word for word in words if word not in stop_words]
        tokenized_words.append(words)
    return tokenized_words

def getData():
    train_file=pd.read_csv('src/dataset/train.csv',header=None,index_col=None)
    test_file=pd.read_csv('src/dataset/trial.csv',header=None, index_col=None)

    x=np.concatenate((train_file[3][1:],test_file[3][1:]))
    y_train=train_file[4][1:]

    x=parse_dataset(x)

    print(x)





def parse_dataset(x_data):

    # x_data=tokenizer(x_data)

    tfidVectorizer =TfidfVectorizer(min_df=500)

    vectors=tfidVectorizer.fit_transform(x_data)
    print(tfidVectorizer.get_feature_names_out())

    print('vector.shape',vectors.shape)
    return vectors


def train_svm(x_train, y_train):
    svr=svm.SVC(verbose=True)
    parameters={
        'C':[1,2,4],
        'gamma':[0.5,1,2]
    }
    clf=GridSearchCV(svr,parameters,scoring='f1')
    clf.fit(x_train, y_train,)
    print('The best parameters are: ')
    print(clf.best_params_)

    joblib.dump(clf,'model/svm_tfidf.pkl')


if __name__ == '__main__':
    getData()
    pass