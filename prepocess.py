# -*- coding:utf-8 -*-
import pandas as pd
import re
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import sklearn
import numpy as np
import jieba
from keras.preprocessing.sequence import pad_sequences
def read_train_set():
    '''读取训练集'''
    trainSet_path = './DF_data/train.csv'
    origin  = pd.read_csv(trainSet_path)
    return origin

def read_test_set():
    '''读取训练集'''
    trainSet_path = './DF_data/test_public.csv'
    origin  = pd.read_csv(trainSet_path)
    return origin

def remove_punctuation(line):
    """
    移除标点符号
    """
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[：]","",line)
    return string

def split_sentence(sentence,gram):
    '''
    切分句子,根据我需求的长度进行切分
    gram 就是我需求的长度
    比如： 我是一个猪
    直接切成，我，是，一，个，猪，这样的东西
    '''
    result = []
    for idx in range(len(sentence)):
        if idx + gram <= len(sentence):
            result.append(sentence[idx:idx+gram] )
    return result

def unify_topic():
    trainSet = read_train_set()
    fs = open('./DF_data/train_topic.csv','a')
    fs.write('content_id,content,topics'+'\n')
    for i in trainSet['content_id'].value_counts().index:
        topics =' '.join(trainSet[trainSet.content_id == i].subject.tolist())
        content = trainSet[trainSet.content_id == i].content.tolist()[0]
        fs.write(i+','+content+','+topics+'\n')


def to_one_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0

    return vector
def load_train_test_set(trainSet_path,testSet_path):

    # print text,'\n',label
    trainSet  = pd.read_csv(trainSet_path)
    testSet = pd.read_csv(testSet_path)
    train_text = trainSet['content'].tolist()
    test_text = testSet['content'].tolist()
    text = train_text+test_text
    topics = trainSet['topics']
    text = list(map(lambda x:' '.join(jieba.cut(x)),text))
    train_text = list(map(lambda x:' '.join(jieba.cut(x)),train_text))
    test_text = list(map(lambda x: ' '.join(jieba.cut(x)), test_text))
    labels = set()
    for i in topics:
        for j in i.split():
            labels.add(j)
    label = np.zeros((len(text),len(labels))).astype(np.float32)
    for i, element in enumerate(topics):
        vec = to_one_category_vector(element.split(),list(labels))
        label[i] = vec
    print(label)
    # # 分词，构建单词-id词典
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(text)
    vocab = tokenizer.word_index
    # # # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(train_text)
    X_test_word_ids = tokenizer.texts_to_sequences(test_text)
    lens = []
    for i, element in enumerate(X_train_word_ids+X_test_word_ids):
        lens.append(len(element))
    alens = np.array(lens)
    ranget = np.percentile(alens,100)
    rangeb = np.percentile(alens, 0)
    print(ranget,rangeb,np.median(alens),vocab.__len__()  )
    x_train = pad_sequences(X_train_word_ids, maxlen=128)
    x_test = pad_sequences(X_test_word_ids, maxlen=128)
    del text, X_train_word_ids, X_test_word_ids
    return x_train, x_test, label,128,vocab.__len__(),list(labels)
def load_train_set_data(trainSet_path):


    # print text,'\n',label
    trainSet  = pd.read_csv(trainSet_path)
    text = trainSet['content']
    topics = trainSet['topics']
    text = list(map(lambda x:' '.join(jieba.cut(x)),text))
    labels = set()
    for i in topics:
        for j in i.split():
            labels.add(j)
    label = np.zeros((len(text),len(labels))).astype(np.float32)
    for i, element in enumerate(topics):
        vec = to_one_category_vector(element.split(),list(labels))
        label[i] = vec
    print(label)
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.1, random_state=42)

    # # 分词，构建单词-id词典
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(text)
    vocab = tokenizer.word_index
    # # # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    lens = []
    for i, element in enumerate(X_train_word_ids+X_test_word_ids):
        lens.append(len(element))
    alens = np.array(lens)
    ranget = np.percentile(alens,100)
    rangeb = np.percentile(alens, 0)
    print(ranget,rangeb,np.median(alens),vocab.__len__()  )
    x_train = pad_sequences(X_train_word_ids, maxlen=128)
    x_test = pad_sequences(X_test_word_ids, maxlen=128)
    del text,label,labels,X_train_word_ids,X_test_word_ids
    return x_train, x_test, y_train, y_test,128,vocab.__len__(),list(labels)
def load_train_set_data_setiments(trainSet_path):


    # print text,'\n',label
    trainSet  = pd.read_csv(trainSet_path)
    text = trainSet['content']
    sentiment = trainSet['sentiment_value']
    text = list(map(lambda x:' '.join(jieba.cut(x)),text))
    labels = to_categorical(map(lambda x: int(x)+1, sentiment))
    print('Shape of label tensor:', labels.shape)
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=0.1, random_state=42)

    # # 分词，构建单词-id词典
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(text)
    vocab = tokenizer.word_index
    # # # 将每个词用词典中的数值代替
    X_train_word_ids = tokenizer.texts_to_sequences(X_train)
    X_test_word_ids = tokenizer.texts_to_sequences(X_test)
    lens = []
    for i, element in enumerate(X_train_word_ids+X_test_word_ids):
        lens.append(len(element))
    alens = np.array(lens)
    ranget = np.percentile(alens,100)
    rangeb = np.percentile(alens, 0)
    print(ranget,rangeb,np.median(alens),vocab.__len__()  )
    x_train = pad_sequences(X_train_word_ids, maxlen=128)
    x_test = pad_sequences(X_test_word_ids, maxlen=128)
    del text,labels,X_train_word_ids,X_test_word_ids
    return x_train, x_test, y_train, y_test,128,vocab.__len__(),3