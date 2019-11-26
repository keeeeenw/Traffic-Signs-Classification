import csv
from collections import defaultdict, namedtuple
import os
import shutil


Annotation = namedtuple('Annotation', ['filename', 'label'])

def read_annotations(filename):
    annotations = []
    
    with open(filename) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # skip header

        # loop over all images in current annotations file
        for row in reader:
            filename = row[0]  # filename is in the 0th column
            label = int(row[7])  # label is in the 7th column
            annotations.append(Annotation(filename, label))
            
    return annotations

def load_training_annotations(source_path):
    annotations = []
    for c in range(0,43):
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


def runMain():
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, 'data')

    source_path = os.path.join(data_dir, 'raw_data')
    train_path = os.path.join(data_dir, 'train')
    validation_path = os.path.join(data_dir, 'valid')
    all_path = os.path.join(data_dir, 'all')
    validation_fraction = 0.2
    split_train_validation_sets(source_path, train_path, validation_path, all_path, validation_fraction)


if __name__ == "__main__":
    runMain()