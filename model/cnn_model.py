import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import datetime
import os
import random
import tensorflow as tf
import collections

from skimage import color, exposure, transform, io

from keras import backend as K
from keras import metrics
from keras import optimizers
from keras.applications import vgg16, resnet50
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical


BATCH_SIZE = 128
IMG_SIZE = 96
USE_DATA_AUGMENTATION = False


def preprocessImage(imgPath):
    img = io.imread(imgPath)
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2: centre[0] + min_side // 2, centre[1] - min_side // 2: centre[1] + min_side // 2, :]
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return (img * 255).astype(np.uint8)


def getImageTensor(imgPath):
    # img = image.load_img(imgPath, target_size=(IMG_SIZE, IMG_SIZE))
    img = preprocessImage(imgPath)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = resnet50.preprocess_input(img_tensor)
    return np.squeeze(img_tensor)


def getImageLabel(imgName):
    return int(imgName.split('_')[0])


def createImageModelFromResnet():
    # start with a standard ResNet50 network
    pre_trained_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # pre_trained_model_freeze = 175
    # for layer in pre_trained_model.layers[:pre_trained_model_freeze]:
    #     layer.trainable = False
    # add trainable FC layers
    x = pre_trained_model.get_layer('activation_49').output
    x = Flatten()(x)
    # x = BatchNormalization()(x)
    # x = Dense(256, activation='relu')(x)
    model_output = Dense(43, activation='softmax')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="img_model")
    return img_model


def createImageModelFromVGG16():
    # start with a standard vgg network
    pre_trained_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    pre_trained_model_freeze = 17
    for layer in pre_trained_model.layers[:pre_trained_model_freeze]:
        layer.trainable = False
    x = pre_trained_model.output
    x = Flatten(input_shape=pre_trained_model.output_shape[1:])(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    model_output = Dense(43, activation='softmax')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="img_model")
    return img_model


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
    return len(input_data['x_{}'.format(input_type)]), input_data['x_{}'.format(input_type)], input_data['y_{}'.format(input_type)]


def trainDataGeneratorKaggle(input_data, batch_size, input_type):
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


def trainDataGeneratorRaw(image_dir, augmt_dir, batch_size):
    # these are variables that holds states
    current_index = 0

    all_images = list(os.listdir(image_dir))
    if USE_DATA_AUGMENTATION:
        all_images += list(os.listdir(augmt_dir))

    num_examples = len(all_images)
    random.shuffle(all_images)
    
    while True:
        images, labels = [], []
        for index in range(current_index, min(current_index + batch_size, num_examples)):
            image_path = os.path.join(image_dir, all_images[index])
            if USE_DATA_AUGMENTATION and all_images[index].count('_') > 3:
                image_path = os.path.join(augmt_dir, all_images[index])
            images.append(getImageTensor(image_path))
            labels.append(getImageLabel(all_images[index]))
        current_index += batch_size

        x_input_batched = [np.array(images)]
        y_input_batched = to_categorical(np.array(labels), num_classes=43)
        yield x_input_batched, y_input_batched

        if current_index >= num_examples:
            current_index = 0


def generateAugumentedData(image_dir, save_dir):
    images = collections.defaultdict(list)

    for image_name in list(os.listdir(image_dir)):
        images[getImageLabel(image_name)].append(image_name)

    max_count = max([len(value) for _, value in images.items()])

    for label, bucket in images.items():
        generateAugumentedImage(image_dir, save_dir, bucket, label, max_count - len(bucket))


def generateAugumentedImage(image_dir, save_dir, image_names, label, count):
    datagen = image.ImageDataGenerator(rotation_range=20, zoom_range=0.2, brightness_range=(0.5, 1.5))

    for _ in range(count):
        original_image_name = random.choice(image_names)
        original_image_path = os.path.join(image_dir, original_image_name)

        original_image = image.load_img(original_image_path)
        original_image_array = image.img_to_array(original_image)
        original_image_array = original_image_array.reshape((1, ) + original_image_array.shape)

        for batch in datagen.flow(original_image_array, batch_size=1, save_to_dir=save_dir, save_prefix=os.path.splitext(original_image_name)[0], save_format='ppm'):
            break


def generateAugumentedTruth(image_dir, save_dir, count):
    datagen = image.ImageDataGenerator(rotation_range=20, zoom_range=(1, 1.5), brightness_range=(0.5, 1.5))

    for image_name in list(os.listdir(image_dir)):
        original_image_path = os.path.join(image_dir, image_name)
        original_image = image.load_img(original_image_path)
        original_image_array = image.img_to_array(original_image)
        original_image_array = original_image_array.reshape((1, ) + original_image_array.shape)
        for _ in range(count):
            for batch in datagen.flow(original_image_array, batch_size=1, save_to_dir=save_dir, save_prefix=int(os.path.splitext(image_name)[0]), save_format='jpg'):
                break


def getValidationSetRaw(image_dir):
    all_images = list(os.listdir(image_dir))

    images, labels = [], []
    for image in all_images:
        images.append(getImageTensor(os.path.join(image_dir, image)))
        labels.append(getImageLabel(image))

    return np.array(images), to_categorical(np.array(labels), num_classes=43)


def TrainModel():
    project_root = os.getcwd()

    checkpoint_name = 'cnn-model-{epoch:02d}.hdf5'
    checkpoints_dir = os.path.join(project_root, 'checkpoints', checkpoint_name)
    checkpoint = ModelCheckpoint(checkpoints_dir, monitor='val_acc', verbose=0)

    tensorboard_dir = os.path.join(project_root, 'logs')
    tensorboard = TensorBoard(tensorboard_dir)

    model = createImageModelFromResnet()
    # model = createImageModelFromVGG16()
    adam = optimizers.Adam(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

    initial_epoch = 0
    epochs = 30

    # ------------------------------------------------------------------------------------------------------- #

    image_dir = os.path.join(project_root, 'data', 'train')
    augmt_dir = os.path.join(project_root, 'data', 'train_aug')

    num_samples = len(list(os.listdir(image_dir)))
    if USE_DATA_AUGMENTATION:
        num_samples += len(list(os.listdir(augmt_dir)))

    # Raw generator test
    # x = trainDataGeneratorRaw(image_dir, augmt_dir, BATCH_SIZE)
    # first_batch = next(x)
    # print(first_batch[0][0].shape)
    # print(first_batch[1][0].shape)

    x_input_val, y_input_val = getValidationSetRaw(os.path.join(project_root, 'data', 'valid'))

    h = model.fit_generator(
        trainDataGeneratorRaw(image_dir, augmt_dir, BATCH_SIZE),
        steps_per_epoch=np.ceil(num_samples / BATCH_SIZE),
        initial_epoch=initial_epoch,
        validation_data=(x_input_val, y_input_val),
        epochs=epochs,
        verbose=2,
        callbacks=[checkpoint, tensorboard])

    # ------------------------------------------------------------------------------------------------------- #

    # data_dir = os.path.join(project_root, 'data', 'kaggle', 'data2.pickle')
    # input_data = process_data(data_dir)

    # # Kaggle generator test
    # x = trainDataGeneratorKaggle(input_data, BATCH_SIZE, 'train')
    # first_batch = next(x)
    # print(first_batch[0][0].shape)
    # print(first_batch[1][0].shape)

    # num_samples, _, _ = getDataByType(input_data, 'train')
    # _, x_input_val, y_input_val = getDataByType(input_data, 'validation')

    # h = model.fit_generator(
    #     trainDataGeneratorKaggle(input_data, BATCH_SIZE, 'train'),
    #     steps_per_epoch=np.ceil(num_samples / BATCH_SIZE),
    #     initial_epoch=initial_epoch,
    #     validation_data=(x_input_val, y_input_val),
    #     epochs=epochs,
    #     verbose=2,
    #     callbacks=[checkpoint, tensorboard])


def runTest():
    project_root = os.getcwd()

    model_dir = os.path.join(project_root, 'checkpoints')
    model_path = os.path.join(model_dir, 'cnn-model-15.hdf5')
    if os.path.exists(model_path):
        model = load_model(model_path)

    data_dir = os.path.join(project_root, 'data', 'kaggle', 'data2.pickle')
    input_data = process_data(data_dir)
    num_samples, x_input, y_input = getDataByType(input_data, 'test')

    match_cnt = 0
    for i in range(num_samples):
        x = x_input[i:i + 1]
        y = y_input[i:i + 1]

        score = model.predict(x)
        prediction = np.argmax(score)

        if y[0] == prediction:
            match_cnt += 1

    accuracy = match_cnt / num_samples

    print('Accuracy:', accuracy)


def TestModel():
    project_root = os.getcwd()

    model_dir = os.path.join(project_root, 'checkpoints')
    model_path = os.path.join(model_dir, 'cnn-model-12.hdf5')
    if os.path.exists(model_path):
        model = load_model(model_path)

    image_dir = os.path.join(project_root, 'data', 'valid')
    num_samples = len(list(os.listdir(image_dir)))

    match_cnt = 0
    for image in list(os.listdir(image_dir)):
        x = getImageTensor(os.path.join(image_dir, image))
        y = getImageLabel(image)
        x = np.array([x])

        score = model.predict(x)
        prediction = np.argmax(score)

        if y == prediction:
            match_cnt += 1

    accuracy = match_cnt / num_samples

    print('Accuracy:', accuracy)


if __name__ == "__main__":
    TrainModel()
    # TestModel()
    # generateAugumentedData(os.path.join(os.getcwd(), 'data', 'train'), os.path.join(os.getcwd(), 'data', 'train_aug'))
    # generateAugumentedTruth(os.path.join(os.getcwd(), 'data', 'truth'), os.path.join(os.getcwd(), 'data', 'truth_aug'), 3)