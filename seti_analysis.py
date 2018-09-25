# -*- coding:utf-8 -*-
import jieba
import jieba.analyse
import pandas as pd
import codecs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from prepocess import load_train_set_data_setiments
from my_utils.data_preprocess import simple_generator
from sklearn import metrics
from keras.layers.merge import concatenate
from model.Attention import *
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional,SpatialDropout1D,GRU,GlobalMaxPooling1D,GlobalMaxPool1D
from keras.layers.normalization import BatchNormalization
def bi_gru_model(maxlen,class_num,vocab_len):
    print("get_text_gru3")
    content = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(vocab_len+1, 128)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(GRU(200, return_sequences=True))(x)
    x = Bidirectional(GRU(200, return_sequences=True))(x)
    print x.shape
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def bilstm_h(maxlen,class_num,vocab_len):
    #一次就够
    print("get_text_lstm")
    content = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(vocab_len + 1, 128)

    x = SpatialDropout1D(0.2)(embedding(content))

    # x = Bidirectional(LSTM(200))(x)
    x = Bidirectional(LSTM(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.5)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def bilstmModel2(vocab_len):
    print u'创建模型...'
    model = Sequential()
    model.add(Embedding(vocab_len + 1, 128))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(output_dim=50,
                   activation='relu',
                   inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    print u'编译模型...'
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def bilstmModel(vocab_len):
    # 训练模型
    model = Sequential()
    model.add(Embedding(vocab_len+1, 128))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    return model

x_train, x_test, y_train, y_test,maxlen,vocab__len,labels = load_train_set_data_setiments('./DF_data/train.csv')
# model = bilstmModel2(vocab__len)
model = bi_gru_model(maxlen,3,vocab__len)
# model = get_han2(x_train.shape[1],128,vocab__len)
for i in range(10):
    print('---------------------EPOCH------------------------')
    print(i)
    batch_size = 128
    model.fit_generator(
        simple_generator(x_train, y_train, batch_size=batch_size),
        epochs=1,
        steps_per_epoch=int(x_train.shape[0] / batch_size),
        # callbacks=[metrics],
        validation_data=(x_test, y_test)
    )
    pred = model.predict(x_test)
    pred_oh = np.zeros(pred.shape)
    for i,ele in enumerate(pred):
        vec = np.zeros(3)
        index = np.argmax(ele)
        vec[index] = 1
        pred_oh[i] = vec
        # print 'pred',vec
        # print 'true',y_test[i]
    score = model.evaluate(x_test, y_test)
    print score
    print metrics.f1_score(y_test, pred_oh,average='weighted')
    print metrics.f1_score(y_test, pred_oh, average='samples')
    print metrics.classification_report(y_test, pred_oh)