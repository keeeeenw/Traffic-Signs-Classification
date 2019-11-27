import os
import pickle

def process_data(data_name='data3.pickle', categorial=True):
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, 'data', data_name)

    with open(data_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    if categorial:
        from keras.utils.np_utils import to_categorical
        data['y_train'] = to_categorical(data['y_train'], num_classes=43)

    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    if categorial:
        data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)    

    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    return data

def getDataByType(input_data, input_type):
    return  len(input_data['x_{}'.format(input_type)]), input_data['x_{}'.format(input_type)], input_data['y_{}'.format(input_type)]
