import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import datetime
import os
import random
from math import sqrt, ceil

import tensorflow as tf

from keras import backend as K
from keras import metrics
from keras.applications import resnet
from keras.applications import vgg16
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

DEBUG = False
BATCH_SIZE = 128

def createImageModelFromResnet():
    # start with a standard ResNet50 network
    # pre_trained_model = vgg16.VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    pre_trained_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # pre_trained_model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    # pre_trained_model_freeze = len(pre_trained_model.layers) - 12
    # for layer in pre_trained_model.layers[:18]:
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
    data_dir = os.path.join(project_root, 'data', 'data3.pickle')
    input_data = process_data(data_dir)

    checkpoint_name = 'vgg-model-{epoch:02d}.hdf5'
    # checkpoint_name = 'cnn-model-v1.hdf5'
    checkpoints_dir = os.path.join(project_root, 'checkpoints', checkpoint_name)
    checkpoint = ModelCheckpoint(checkpoints_dir, monitor='val_acc', verbose=1, period=5)

    tensorboard_dir = os.path.join(project_root, 'logs')
    tensorboard = TensorBoard(tensorboard_dir)

    model = createImageModelFromResnet()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])

    initial_epoch = 0
    epochs = 30
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

# Preparing function for ploting set of examples
# As input it will take 4D tensor and convert it to the grid
# Values will be scaled to the range [0, 255]
def convert_to_grid(x_input):
    N, H, W, C = x_input.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = x_input[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                next_idx += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1
    
    plt.imshow(grid.astype('uint8'), cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(15, 15)
    # plt.title('Some examples of mistmatched testing data', fontsize=18)
    plt.show()
    plt.close()

    # Saving plot
    # fig = plt.figure()
    # fig.savefig('testing_examples.png')
    # plt.close()

    return grid

def runTest(model_path):
    project_root = os.getcwd()
    if os.path.exists(model_path):
        model = load_model(model_path)

    data_dir = os.path.join(project_root, 'data', 'data2.pickle')
    input_data = process_data(data_dir)
    num_samples, x_input, y_input = getDataByType(input_data, 'test')
    
    match_cnt = 0
    mismatched_inputs = []
    mismatched_labels = [] #(expected, predicted)
    for i in range(num_samples):
        if i % 1000 == 0:
            print("Testing sample %d out of %d" % (i, num_samples))
        x = x_input[i:i+1]
        y = y_input[i:i+1]

        score = model.predict(x)
        prediction = np.argmax(score)
        
        if y[0] == prediction:
            match_cnt += 1
        else:
            mismatched_inputs.append(x[0])
            mismatched_labels.append((y[0], prediction))
    
    convert_to_grid(np.array(mismatched_inputs[:3]))

    accuracy = match_cnt / num_samples

    print('Accuracy:', accuracy)

def confusion_matrix_gen(model, x_test, y_test, labels, title=""):
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    confusion = confusion_matrix(y_test, y_pred)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    plt.figure(figsize = (15,10))
    ax = plt.axes()
    svm = sn.heatmap(confusion_df, ax = ax)
    ax.set_title(title)
    figure = svm.get_figure()
    figure.savefig('svm_confusion.png', dpi=400)

# Defining function for getting texts for every class - labels
def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []
    
    # Reading 'csv' file and getting image's labels
    r = pd.read_csv(file)
    # Going through all names
    for name in r['SignName']:
        # Adding from every row second column with name of the label
        label_list.append(name)
    
    # Returning resulted list with labels
    return label_list

def runConfusion(model_name, title=""):
    project_root = os.getcwd()

    model_dir = os.path.join(project_root, 'checkpoints')
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        model = load_model(model_path)

    data_dir = os.path.join(project_root, 'data', 'data2.pickle')
    input_data = process_data(data_dir)
    num_samples, x_input, y_input = getDataByType(input_data, 'test')

    labels = label_text(os.path.join(project_root, 'data', 'label_names.csv'))

    confusion_matrix_gen(model, x_input, y_input, labels, title)

def createBaselineModel():
    pre_trained_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = pre_trained_model.layers[-2].output
    x = Flatten()(x)
    model_output = Dense(43, activation='relu')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="pre_trained_img_model")
    print(img_model.summary())
    return img_model

if __name__ == "__main__":
    runMain()
    # project_root = os.getcwd()
    # model_dir = os.path.join(project_root, 'checkpoints')
    # model_path = os.path.join(model_dir, 'vgg16-model-35-progress-report.hdf5')
    # runTest(model_path)
    # runConfusion("resnet50-model-50-progress-report.hdf5")
    # runConfusion("vgg16-model-35-progress-report.hdf5", "Confusion Matrix VGG16 Based Model")