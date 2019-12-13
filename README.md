# Traffic-Signs-Classification

## Introduction
In this project, we introduce a traffic sign classifier for single-image, multi-class classification task.
Specifically, given an input image containing a traffic sign, our system output the class of the traffic
sign. We would like to explore different types of machine learning and/or deep learning models in order to
obtain high accuracy and robustness for our system.

## File Structure
### model/cnn_model.py

This file includes our core modeling functions for CNN based models. You can run it by modifying the main
function with the appropriate data processing, train and testing calls.

```
python model/cnn_model.py
```

### process_data/process_data.py

This file help us processing GTSRB raw image and prepare them for consuming in CNN based models.

```
python process_data/process_data.py
```

### StarterCode.ipynb

You will need to open and execute this file using Jupyter Notebook.

This notebook includes our initial experiments and our simple CNN model.

### HogModel.ipynb

You will need to open and execute this file using Jupyter Notebook.

This notebook includes all of Hog and Random Forests model related code. It includes everything
from data process to modeling to running experiments. 

We are using a separate notebook here because
the pipeline for Hog and Random Fortest model is different and does not depend on Deep Learning frameworks such
as Tensorflow and Keras.

### Final-Report.ipynb

You will need to open and execute this file using Jupyter Notebook.

This notebook is used to run experiments and final report for our CNN based models. To reuse most
of our existing code, we imported funcions from model/cnn_model.py and modified some of functions to
allow us to run parameterized experiments.

You can also find the code for plotting confusion matrix and saliency map in here.