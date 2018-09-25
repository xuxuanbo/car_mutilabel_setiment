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

