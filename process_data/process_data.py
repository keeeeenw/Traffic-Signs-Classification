"""
Module to process GTSRB dataset

Adapted from "ResNet for Traffic Sign Classification With PyTorch"
https://towardsdatascience.com/resnet-for-traffic-sign-classification-with-pytorch-5883a97bbaa3

Modified by: Yu Zhao
"""

import csv
import os
import shutil

from collections import defaultdict, namedtuple
from PIL import Image


Annotation = namedtuple('Annotation', ['filename', 'label'])


def read_annotations(filename):
    annotations = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # skip header

        # loop over all images in current annotations file
        for row in reader:
            filename = row[0]  # filename is in the 0th column
            label = int(row[7])  # label is in the 7th column
            annotations.append(Annotation(filename, label))

    return annotations


def load_training_annotations(source_path):
    annotations = []
    for c in range(0, 43):
        filename = os.path.join(source_path, format(c, '05d'), 'GT-' + format(c, '05d') + '.csv')
        annotations.extend(read_annotations(filename))
    return annotations


def copy_files(label, filenames, source, destination, move=False):
    func = os.rename if move else shutil.copyfile
    for filename in filenames:
        destination_path = os.path.join(destination, '{}_{}'.format(str(label), filename))
        if not os.path.exists(destination_path):
            func(os.path.join(source, format(label, '05d'), filename), destination_path)


def split_train_validation_sets(source_path, train_path, validation_path, all_path, validation_fraction=0.2):
    """
    Splits the GTSRB training set into training and validation sets.
    """

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    if not os.path.exists(all_path):
        os.makedirs(all_path)

    annotations = load_training_annotations(source_path)
    filenames = defaultdict(list)
    for annotation in annotations:
        filenames[annotation.label].append(annotation.filename)

    for label, filenames in filenames.items():
        filenames = sorted(filenames)

        validation_size = int(len(filenames) // 30 * validation_fraction) * 30
        train_filenames = filenames[validation_size:]
        validation_filenames = filenames[:validation_size]

        copy_files(label, filenames, source_path, all_path, move=False)
        copy_files(label, train_filenames, source_path, train_path, move=True)
        copy_files(label, validation_filenames, source_path, validation_path, move=True)


def generate_train_and_validation():
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, 'data')

    source_path = os.path.join(data_dir, 'raw_data')
    train_path = os.path.join(data_dir, 'train')
    validation_path = os.path.join(data_dir, 'valid')
    all_path = os.path.join(data_dir, 'all')
    validation_fraction = 0.2
    split_train_validation_sets(source_path, train_path, validation_path, all_path, validation_fraction)


def convert_transparent_to_white(image_dir):
    for image_name in list(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)

        image = Image.open(image_path)
        image.convert('RGBA')
        pixel_data = image.load()

        if image.mode == "RGBA":
            for y in range(image.size[1]):
                for x in range(image.size[0]):
                    if pixel_data[x, y][3] < 255:
                        pixel_data[x, y] = (255, 255, 255, 255)

        image.save(image_path)


def generateReportImages():
    images = [
        '1_00052_00027.ppm',
        '2_00036_00027.ppm',
        '9_00022_00027.ppm',
        '12_00021_00027.ppm',
        '14_00023_00027.ppm',
        '15_00009_00027.ppm',
        '16_00002_00027.ppm',
        '17_00019_00027.ppm',
        '18_00014_00027.ppm',
        '19_00004_00027.ppm',
        '20_00008_00027.ppm',

        '3_00000_00027.ppm',
        '4_00003_00027.ppm',
        '5_00007_00027.ppm',
        '6_00000_00027.ppm',
        '7_00004_00027.ppm',
        '8_00008_00027.ppm',
        '10_00004_00027.ppm',
        '11_00000_00027.ppm',
        '13_00001_00026.ppm',
    ]
    project_root = os.getcwd()
    input_dir = os.path.join(project_root, 'data', 'valid')
    copy_dir = os.path.join(project_root, 'data', 'hsv_before')
    output_dir = os.path.join(project_root, 'data', 'hsv_transformed')
    for img in images:
        input_imgPath = os.path.join(input_dir, img)
        copy_imgPath = os.path.join(copy_dir, img)
        output_imgPath = os.path.join(output_dir, img)
        shutil.copyfile(input_imgPath, copy_imgPath)
        generateHSVBalancedImage(input_imgPath, output_imgPath)


def generateHSVBalancedImage(input_imgPath, output_imgPath):
    img = io.imread(input_imgPath)
    hsv = color.rgb2hsv(img[:, :, 0:3])
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    io.imsave(output_imgPath, (img * 255).astype(np.uint8))


if __name__ == "__main__":
    generate_train_and_validation()
    # convert_transparent_to_white(os.path.join(os.getcwd(), 'data', 'truth'))
    # generateReportImages()
