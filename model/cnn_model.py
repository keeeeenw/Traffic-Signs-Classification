import collections
import csv
import numpy as np  # linear algebra
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import random
import shutil
import sys
import tensorflow as tf

from keras import backend as K
from keras import metrics
from keras import optimizers
from keras.applications import vgg16, resnet50
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from skimage import color, exposure, transform, io
from sklearn.metrics import confusion_matrix


BATCH_SIZE = 128
IMG_SIZE = 96
TOTAL_EPOCH = 30


np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


def preprocessImage(imgPath):
    img = io.imread(imgPath)
    # Histogram normalization in v channel
    # hsv = color.rgb2hsv(img[:, :, 0:3])
    # hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    # img = color.hsv2rgb(hsv)
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
    if '_' in imgName:
        return int(imgName.split('_')[0])
    return int(imgName.split('.')[0])


def createImageModelFromResnet():
    # start with a standard ResNet50 network
    pre_trained_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # pre_trained_model_freeze = 175
    # for layer in pre_trained_model.layers[:pre_trained_model_freeze]:
    #     layer.trainable = False
    # add trainable FC layers
    x = pre_trained_model.get_layer('activation_49').output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    model_output = Dense(43, activation='softmax')(x)
    img_model = Model(inputs=pre_trained_model.input, outputs=model_output, name="img_model")
    return img_model


def createImageModelFromResnetWithAttention():
    # start with a standard ResNet50 network
    pre_trained_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Construct attention
    x = pre_trained_model.get_layer('activation_49').output
    image_attention_input = Convolution2D(2048, (3, 3), activation='relu', padding='same', name="attention_image_conv")(x)
    attention = Lambda(lambda tensor: K.softmax(tensor), name="attention_softmax")(image_attention_input)
    # Apply attention
    x = Lambda(lambda tensors: tensors[0] * tensors[1], name="attention_apply")([attention, x])
    # Return to original network
    x = GlobalAveragePooling2D()(x)
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


def trainDataGeneratorRaw(image_dir, augmt_dir, batch_size, use_augmentation):
    # these are variables that holds states
    current_index = 0
    path_dictionary = {}

    all_images = list(os.listdir(image_dir))
    for image in list(os.listdir(image_dir)):
        path_dictionary[image] = os.path.join(image_dir, image)

    if use_augmentation:
        all_images += list(os.listdir(augmt_dir))
        for image in list(os.listdir(augmt_dir)):
            path_dictionary[image] = os.path.join(augmt_dir, image)

    num_examples = len(all_images)
    random.shuffle(all_images)

    while True:
        images, labels = [], []
        for index in range(current_index, min(current_index + batch_size, num_examples)):
            images.append(getImageTensor(path_dictionary[all_images[index]]))
            labels.append(getImageLabel(all_images[index]))
        current_index += batch_size

        x_input_batched = [np.array(images)]
        y_input_batched = to_categorical(np.array(labels), num_classes=43)
        yield x_input_batched, y_input_batched

        if current_index >= num_examples:
            current_index = 0


def testDataGeneratorRaw(image_dir, all_images, batch_size):
    # these are variables that holds states
    current_index = 0
    num_examples = len(all_images)

    while True:
        images = []
        for index in range(current_index, min(current_index + batch_size, num_examples)):
            images.append(getImageTensor(os.path.join(image_dir, all_images[index])))

        current_index += batch_size
        yield [np.array(images)]

        if current_index >= num_examples:
            current_index = 0


def generateAugmentedData(image_dir, save_dir, max_item=100000):
    images = collections.defaultdict(list)

    for image_name in list(os.listdir(image_dir)):
        images[getImageLabel(image_name)].append(image_name)

    # DEBUG: Statistics about the dataset
    # for label, bucket in sorted(images.items(), key=lambda item: len(item[1]), reverse=True):
    #     print('Label {} has {} samples.'.format(label, len(bucket)))

    max_count = min(max_item, max([len(value) for _, value in images.items()]))
    for label, bucket in images.items():
        generateAugmentedImage(image_dir, save_dir, bucket, label, max(0, max_count - len(bucket)))


def generateAugmentedImage(image_dir, save_dir, image_names, label, count):
    if count <= 0:
        return

    datagen = image.ImageDataGenerator(rotation_range=20, zoom_range=(1, 1.2), brightness_range=(0.8, 1.2))

    for _ in range(count):
        original_image_name = random.choice(image_names)
        original_image_path = os.path.join(image_dir, original_image_name)

        original_image = image.load_img(original_image_path)
        original_image_array = image.img_to_array(original_image)
        original_image_array = original_image_array.reshape((1, ) + original_image_array.shape)

        for batch in datagen.flow(original_image_array, batch_size=1, save_to_dir=save_dir, save_prefix=os.path.splitext(original_image_name)[0], save_format='ppm'):
            break


def generateAugmentedTruth(image_dir, save_dir, count):
    datagen = image.ImageDataGenerator(rotation_range=20, zoom_range=(1, 2), brightness_range=(0.8, 1.2))

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


def TrainModel(dataset='train', from_epoch=0, use_validation=False, use_augmentation=False, use_limited_augmentation=False):
    project_root = os.getcwd()

    checkpoint_name = 'cnn-model-{epoch:02d}.hdf5'
    checkpoints_dir = os.path.join(project_root, 'checkpoints', checkpoint_name)
    checkpoint = ModelCheckpoint(checkpoints_dir, monitor='val_acc', verbose=0)

    tensorboard_dir = os.path.join(project_root, 'logs')
    tensorboard = TensorBoard(tensorboard_dir)

    if from_epoch > 0:
        model_path = os.path.join(project_root, 'checkpoints', checkpoint_name.format(epoch=from_epoch))
        model = load_model(model_path, custom_objects={'tf': tf})
        initial_epoch = from_epoch
    else:
        model = createImageModelFromResnet()
        # model = createImageModelFromVGG16()
        # model = createImageModelFromResnetWithAttention()
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])
        initial_epoch = 0

    # ------------------------------------------------------------------------------------------------------- #

    image_dir = os.path.join(project_root, 'data', dataset)
    augmt_dir = os.path.join(project_root, 'data', '{}_aug_limited'.format(dataset) if use_limited_augmentation else '{}_aug'.format(dataset))

    num_samples = len(list(os.listdir(image_dir)))
    if use_augmentation:
        num_samples += len(list(os.listdir(augmt_dir)))

    # # Raw generator test
    # x = trainDataGeneratorRaw(image_dir, augmt_dir, BATCH_SIZE, use_augmentation)
    # first_batch = next(x)
    # print(first_batch[0][0].shape)
    # print(first_batch[1][0].shape)

    if use_validation:
        x_input_val, y_input_val = getValidationSetRaw(os.path.join(project_root, 'data', 'valid'))

        h = model.fit_generator(
            trainDataGeneratorRaw(image_dir, augmt_dir, BATCH_SIZE, use_augmentation),
            steps_per_epoch=np.ceil(num_samples / BATCH_SIZE),
            initial_epoch=initial_epoch,
            validation_data=(x_input_val, y_input_val),
            epochs=TOTAL_EPOCH,
            verbose=1,
            callbacks=[checkpoint, tensorboard])

    else:
        h = model.fit_generator(
            trainDataGeneratorRaw(image_dir, augmt_dir, BATCH_SIZE, use_augmentation),
            steps_per_epoch=np.ceil(num_samples / BATCH_SIZE),
            initial_epoch=initial_epoch,
            epochs=TOTAL_EPOCH,
            verbose=1,
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


def ReadLabelFromCSV(filename):
    annotations = {}

    with open(filename) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # skip header

        # loop over all images in current annotations file
        for row in reader:
            filename = row[0]  # filename is in the 0th column
            label = int(row[7])  # label is in the 7th column
            annotations[filename] = label

    return annotations


def ErrorAnalysis(analysis_path, image_name, image_dir, label, prediction):
    new_name = '{}_|_{}_|_{}'.format(prediction, label, image_name)
    source_path = os.path.join(image_dir, image_name)
    destination_path = os.path.join(analysis_path, new_name)
    # shutil.copyfile(source_path, destination_path)


def ErrorStatistics(y_true, y_pred):
    tally = {}
    for i in range(len(y_true)):
        label, pred = y_true[i], y_pred[i]
        if label not in tally:
            tally[label] = collections.defaultdict(int)
        tally[label][pred] += 1

    for label, perf in sorted(tally.items(), key=lambda item: item[0]):
        to_print = 'Sign {} mispredicted as'.format(label)
        should_print = False
        correct, mispredicted = 0, 0
        for pred, count in sorted(perf.items(), key=lambda item: item[1], reverse=True):
            if label != pred and count != 0:
                to_print += ' ({}: {}) '.format(pred, count)
                should_print = True
                mispredicted += count
            else:
                correct += count
        if should_print:
            to_print += 'accuracy is {} with {}/{}'.format(correct/(correct+mispredicted), correct, correct+mispredicted)
            print(to_print)


def TestModel(model_path, analysis_path=None):
    if os.path.exists(model_path):
        model = load_model(model_path)

    project_root = os.getcwd()
    image_dir = os.path.join(project_root, 'data', 'test')
    all_images = list(os.listdir(image_dir))
    num_samples = len(all_images)

    labels = ReadLabelFromCSV(os.path.join(project_root, 'data', 'GT-final_test.csv'))
    y_true = [labels[image] for image in all_images]

    # # generator test
    # x = testDataGeneratorRaw(image_dir, all_images, BATCH_SIZE)
    # first_batch = next(x)
    # print(first_batch[0][22])

    y_pred = model.predict_generator(testDataGeneratorRaw(image_dir, all_images, BATCH_SIZE), steps=np.ceil(num_samples / BATCH_SIZE))
    y_pred = np.argmax(y_pred, axis=1)

    match_cnt = 0
    for i in range(num_samples):
        if y_true[i] == y_pred[i]:
            match_cnt += 1
        elif analysis_path:
            ErrorAnalysis(analysis_path, all_images[i], image_dir, y_true[i], y_pred[i])

    accuracy = match_cnt / num_samples
    print('Accuracy: {}'.format(accuracy))
    # ErrorStatistics(y_true, y_pred)
    # print(confusion_matrix(y_true, y_pred))


def _get_dataset_by_name(name):
    path = os.path.join(os.getcwd(), 'data', name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == "__main__":
    # TrainModel(from_epoch=0, dataset='train', use_validation=True, use_augmentation=True, use_limited_augmentation=False)

    # Test Model
    # for i in range(21, 30):
    #     TestModel(os.path.join(os.getcwd(), 'checkpoints', 'cnn-model-{:02d}.hdf5'.format(i)))

    # # Error Analysis
    # model = 'resnet50_trainAndValid_aug_no_lighting_change'
    # epoch = '17'
    # TestModel(os.path.join(os.getcwd(), 'best_model', model, 'cnn-model-{}.hdf5'.format(epoch)), os.path.join(os.getcwd(), 'error_analysis'))

    # # Generate Augmented Data
    # original_dataset = 'train_and_valid'
    # augmented_dataset = '{}_aug'.format(original_dataset)
    # augmented_limited_dataset = '{}_aug_limited'.format(original_dataset)
    # generateAugmentedData(_get_dataset_by_name(original_dataset), _get_dataset_by_name(augmented_dataset))
    # generateAugmentedData(_get_dataset_by_name(original_dataset), _get_dataset_by_name(augmented_limited_dataset), max_item=1000)

    # # Generate Augmented Truth
    # original_dataset = 'truth'
    # augmented_dataset = '{}_aug'.format(original_dataset)
    # generateAugmentedTruth(_get_dataset_by_name(original_dataset), _get_dataset_by_name(augmented_dataset), 50)
