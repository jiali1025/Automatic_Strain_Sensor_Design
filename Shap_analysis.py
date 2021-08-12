from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import os, time, gc, pickle, itertools, math
from typing import List
import numpy as np
import pandas as pd
import shap
from own_package.features_labels_setup import load_data_to_fl, load_testset_to_fl
def load_model_ensemble(model_directory) -> List:
    """
    Load list of trained keras models from a .h5 saved file that can be used for prediction later
    :param model_directory: model directory where the h5 models are saved in. NOTE: All the models in the directory will
     be loaded. Hence, make sure all the models in there are part of the ensemble and no unwanted models are in the
     directory
    :return: [List: keras models]
    """
    # Loading model names into a list
    model_name_store = []
    directory = model_directory
    for idx, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        model_name_store.append(directory + '/' + filename)
    print('Loading the following models from {}. Total models = {}'.format(directory, len(model_name_store)))
    # Loading model class object into a list
    model_store = []
    for name in model_name_store:
        if name.endswith(".pkl"):  # DTR models
            model_store.append(pickle.load(open(name, 'rb')))
        elif name.endswith('.h5'):  # ANN models
            try:
                model_store.append(load_model(name))
            except ValueError:
                model_store.append(load_model(name, compile=False))
        else:
            print('{} found that does not end with .pkl or .h5'.format(name))
        print('Model {} has been loaded'.format(name))

    return model_store

def model_ensemble_prediction( input_feature):
    """
    Run prediction given one set of feactures_c_norm input, using all the models in model store.
    :param model_store: List of models returned by the def load_model_ensemble
    :param features_c_norm: ndarray of shape (1, -1). The columns represents the different features.
    :return: List of metrics.
    """
    predictions_store = []
    arr = np.array([[1,0,0]])
    repeat_num = input_feature.shape[0]
    manual_feature = arr.repeat(repeat_num, axis=0)
    feature_append = np.concatenate((input_feature,manual_feature),axis=1)

    for model in model_store:
        p_y = model.predict(feature_append)
        for row, p_label in enumerate(p_y.tolist()):
            if p_label[1] > p_label[2]:
                p_y[row, 1] = p_y[row, 2]
            if p_label[0] > p_y[row, 1]:
                p_y[row, 0] = p_y[row, 1]
        predictions_store.append(p_y)
    predictions_store = np.array(predictions_store).squeeze()
    predictions_mean = np.mean(predictions_store, axis=0)
    return predictions_mean

loader_file = './excel/0D_raw.xlsx'
# test_file = './excel/ett30_small.xlsx'
model_store = load_model_ensemble('/home/lijiali1025/projects/redo/models')
analysis_data = load_data_to_fl(loader_file, norm_mask=[0, 1, 3, 4, 5], normalise_labels=False)

# test_data =  load_testset_to_fl(test_file, norm_mask=[0, 1, 3, 4, 5], scaler=fl.scaler)
features_c_norm = analysis_data.features_c_norm
input_feature =features_c_norm[:, :3]
explainer = shap.explainers.Exact(model_ensemble_prediction, input_feature)
X_idx = 10
shap_value_single = explainer(input_feature[10:11,:])
print('cool')