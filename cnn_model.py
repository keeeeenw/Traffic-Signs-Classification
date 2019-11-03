import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import datetime
import os
import random

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

def process_data():
    with open('data/data2.pickle', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    # Preparing y_train and y_validation for using in Keras
    data['y_train'] = to_categorical(data['y_train'], num_classes=43)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)

    # Making channels come at the end
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    # Showing loaded data from file
    for i, j in data.items():
        if i == 'labels':
            print(i + ':', len(j))
        else: 
            print(i + ':', j.shape)


if __name__ == "__main__":
    process_data()