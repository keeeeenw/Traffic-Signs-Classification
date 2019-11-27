from common import *
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, exposure

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def showimg_n_hog(grayimg, hogImage):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(grayimg)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    ax2.axis('off')
    ax2.imshow(hogImage, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

def process_hog_data(data_name='data3.pickle'):
    # Only get grey scale images
    data = process_data(data_name, False)
    data_processed = defaultdict(list);

    # Convert training data
    num_count = 0
    for data_key in ['x_train', 'x_validation', 'x_test']:
        for ig in data[data_key]:
            num_count += 1
            (hogFeat, hogImage) = feature.hog(ig[:,:,0], orientations=9, pixels_per_cell=(4, 4),
                                    cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
            hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255)).astype("uint8")

            if num_count % 500 == 0:
                print("Processed %d out of %d %s HOG Data" % (num_count, len(data[data_key]), data_key))
            
            data_processed[data_key].append(hogFeat)
            print(hogFeat)
            # showimg_n_hog(ig[:,:,0], hogImage)

        data[data_key] = np.asarray(data_processed[data_key])
    with open('data/HoGData.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data

def read_hog_data(data_name="HoGData.pickle"):
    process_data(data_name, False)

if __name__ == "__main__":
    data = process_hog_data()
    print(data['x_validation'].shape)
    data = read_hog_data()
    print(data['x_validation'].shape)