# -*- coding:utf-8 -*-
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional,SpatialDropout1D,GRU,GlobalMaxPooling1D,GlobalMaxPool1D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from prepocess import load_train_set_data,load_train_test_set,load_train_set_data_hanlp
from sklearn import metrics
import sklearn
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import Concatenate
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.utils import np_utils
from keras import layers
from my_utils.data_preprocess import simple_generator
def bi_gru_model(maxlen,class_num,vocab_len):
    print("get_text_gru3")
    content = Input(shape=(maxlen,), dtype='int32')
    embedding = Embedding(vocab_len+1, 300)

    x = SpatialDropout1D(0.2)(embedding(content))

    x = Bidirectional(GRU(200, return_sequences=True))(x)
    x = Bidirectional(GRU(200, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])

    x = Dropout(0.2)(Activation(activation="relu")(BatchNormalization()(Dense(1000)(conc))))
    x = Activation(activation="relu")(BatchNormalization()(Dense(500)(x)))
    output = Dense(class_num, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)
def get_textcnn(maxlen):
    content = Input(shape=(maxlen,), dtype="int32")
    embedding = Embedding(vocab_len + 1, 300)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = convs_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=content, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model
def bigru_train():
    # x_train, x_test, y_train, y_test,maxlen,vocab_len,labels =load_train_set_data('./DF_data/train_topic.csv')
    x_train, x_test, y_train, y_test, maxlen, vocab_len, labels =load_train_set_data_hanlp('nlpt')
    model = bi_gru_model(maxlen,10,vocab_len)


    for i in range(2):
        print('---------------------EPOCH------------------------')
        print(i)
        batch_size = 128
        model.fit_generator(
            simple_generator(x_train, y_train, batch_size=batch_size),
            epochs=1,
            steps_per_epoch=int(x_train.shape[0] / batch_size),
            #callbacks=[metrics],
            validation_data=(x_test, y_test)
        )
        pred = model.predict(x_test)
        print pred
        print pred.shape
        pred_oh = np.zeros(pred.shape)

        for i,ele in enumerate(pred):
            vec = np.zeros(10)
            find = False
            for j,e in enumerate(ele):
                if e>0.5:
                   vec[j] = 1.
                   find = True
                else:
                   vec[j] = 0.
            if not find:
               vec[np.argmax(ele)] = 1.
            pred_oh[i] = vec
            print 'pred',vec
            print 'true',y_test[i]
        score = model.evaluate(x_test, y_test)
        print score
        print metrics.average_precision_score(y_test,pred_oh)
        print metrics.classification_report(y_test,pred_oh)
def to_categories_name(testSet_path,metrix,labels):
    origin  = pd.read_csv(testSet_path)
    id = origin.content_id
    content = origin.content
    fs = open('./DF_data/test_result','a')
    for i,ele in enumerate(metrix):
        fs.write(id[i]+','+content[i]+',')
        for j,e in enumerate(ele):
            if e == 1.:
                fs.write(labels[j]+' ')
        fs.write('\n')
    fs.close()
def bigru_test():
    x_train, x_test, y_train, maxlen, vocab_len, labels = load_train_test_set('./DF_data/train_topic.csv','./DF_data/test_public.csv')
    model = bi_gru_model(maxlen, 10,vocab_len)
    for i in range(2):
        print('---------------------EPOCH------------------------')
        print(i)
        batch_size = 128
        model.fit_generator(
            simple_generator(x_train, y_train, batch_size=batch_size),
            epochs=1,
            steps_per_epoch=int(x_train.shape[0] / batch_size),
        )
        pred = model.predict(x_test)
        print pred
        print pred.shape
        pred_oh = np.zeros(pred.shape)

        for i, ele in enumerate(pred):
            vec = np.zeros(10)
            find = False
            for j, e in enumerate(ele):
                if e > 0.5:
                    vec[j] = 1.
                    find = True
                else:
                    vec[j] = 0.
            if not find:
                vec[np.argmax(ele)] = 1.
            pred_oh[i] = vec

        to_categories_name('./DF_data/test_public.csv',pred_oh,labels)
bigru_train()