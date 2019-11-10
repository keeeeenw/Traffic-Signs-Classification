import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import datetime
import os
import random
import tensorflow as tf

from keras import backend as K
from keras import metrics
from keras.applications import resnet50
from keras.applications import vgg16
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


DEBUG = False
BATCH_SIZE = 128

def createImageModelFromResnet():
    # start with a standard ResNet50 network
    pre_trained_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # pre_trained_model_freeze = len(pre_trained_model.layers)
    # for layer in pre_trained_model.layers[:pre_trained_model_freeze]:
    #     layer.trainable = False
    # add trainable FC layers
    x = pre_trained_model.layers[-1].output
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    x = Flatten()(x)
    model_output = Dense(43, activation='softmax')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="img_model")
    print(img_model.summary())
    return img_model


def trainWithResnetModel(img_input_shape):
    img_input = Input(img_input_shape, name="img_embedding_1")
    img_model = createImageModelFromResnet()

    img_encoding = img_model(img_input)
    # img_encoding_1_norm = Lambda(adjustInput, name="img_encoding_1_norm")(img_encoding_1)

    if DEBUG:
        img_encoding = Lambda(lambda x: tf.Print(x, [tf.shape(x)], "img_encoding shape is: "))(img_encoding)
        # img_encoding_2_norm = Lambda(lambda x: tf.Print(x, [tf.shape(x)], "img_encoding_2_norm shape is: "))(img_encoding_2_norm)

    output = Lambda(lambda tensors: tf.concat(tensors, 1), name="assemble_output")([lang_encoding_norm, img_encoding_1_norm, img_encoding_2_norm])
    triplet_net = Model(inputs=[img_input], outputs=output)

    return triplet_net


def process_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['y_train'] = to_categorical(data['y_train'], num_classes=43)

    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)    

    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    return data


def getDataByType(input_data, input_type):
    return  len(input_data['x_{}'.format(input_type)]), input_data['x_{}'.format(input_type)], input_data['y_{}'.format(input_type)]


def trainDataGenerator(input_data, batch_size, input_type):
    # these are variables that holds states
    current_index = 0
    num_examples, x_input, y_input = getDataByType(input_data, input_type)
    
    while True:
        images, labels = [], []
        for index in range(current_index, min(current_index + batch_size, num_examples)):
            images.append(x_input[index])
            labels.append(y_input[index]) 
        current_index += batch_size

        x_input_batched = [np.array(images)]
        y_input_batched = np.array(labels)
        yield x_input_batched, y_input_batched

        if current_index >= num_examples:
            current_index = 0


def runMain():
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, 'data', 'data2.pickle')
    input_data = process_data(data_dir)
    
    checkpoint_name = 'cnn-model-{epoch:02d}.hdf5'
    # checkpoint_name = 'cnn-model-v1.hdf5'
    checkpoints_dir = os.path.join(project_root, 'checkpoints', checkpoint_name)
    checkpoint = ModelCheckpoint(checkpoints_dir, monitor='val_acc', verbose=1, period=5)

    tensorboard_dir = os.path.join(project_root, 'logs')
    tensorboard = TensorBoard(tensorboard_dir)

    model = createImageModelFromResnet()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

    initial_epoch = 0
    epochs = 100
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))

    # # generator test
    # x = trainDataGenerator(input_data, BATCH_SIZE, 'train')
    # first_batch = next(x)
    # print(first_batch[0][0].shape)
    # print(first_batch[1][0].shape)

    # num_samples, _, _ = getDataByType(input_data, 'train')
    # _, x_input_val, y_input_val = getDataByType(input_data, 'validation')

    # h = model.fit_generator(
    #     trainDataGenerator(input_data, BATCH_SIZE, 'train'),
    #     steps_per_epoch=np.ceil(num_samples/BATCH_SIZE),
    #     initial_epoch=initial_epoch,
    #     epochs=epochs,
    #     validation_data = (x_input_val, y_input_val),
    #     verbose=2,
    #     callbacks=[checkpoint, tensorboard])  # annealer

    h = model.fit(
        input_data['x_train'], input_data['y_train'],
        batch_size = BATCH_SIZE,
        epochs = epochs,
        validation_data = (input_data['x_validation'], input_data['y_validation']),
        verbose=1,
        callbacks=[checkpoint, tensorboard])  # annealer


def runTest():
    project_root = os.getcwd()

    model_dir = os.path.join(project_root, 'checkpoints')
    model_path = os.path.join(model_dir, 'cnn-model-20.hdf5')
    if os.path.exists(model_path):
        model = load_model(model_path)

    data_dir = os.path.join(project_root, 'data', 'data2.pickle')
    input_data = process_data(data_dir)
    num_samples, x_input, y_input = getDataByType(input_data, 'test')
    
    match_cnt = 0
    for i in range(num_samples):
        if i % 1000 == 0:
            print("Testing sample %d out of %d" % (i, num_samples))
        x = x_input[i:i+1]
        y = y_input[i:i+1]

        score = model.predict(x)
        prediction = np.argmax(score)
        
        if y[0] == prediction:
            match_cnt += 1

    accuracy = match_cnt / num_samples

    print('Accuracy:', accuracy)

def createBaselineModel():
    pre_trained_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = pre_trained_model.layers[-2].output
    x = Flatten()(x)
    model_output = Dense(43, activation='relu')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="pre_trained_img_model")
    print(img_model.summary())
    return img_model

if __name__ == "__main__":
    # runMain()
    runTest()