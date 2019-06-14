import os

import cv2
import numpy as np

from keras.models import load_model

from config import *
from data import get_imgpath_labels

model_path = '../2_outputs/trainedModels/model.h5'

model = load_model(model_path)

test_matfile = 'devkit/cars_test_annos.mat'

img_names_list, _ = get_imgpath_labels(test_matfile, train=False)

preds_list = []
conf_list = []

total_img_count = len(img_names_list)

for i, filename in enumerate(img_names_list):
    filepath = os.path.join(DATA_DIR, IMG_DIR, filename)
    if not os.path.exists(filepath):
        raise Exception('Image not found on the specified path {0:s}'.format(filepath))

    img = cv2.imread(filepath).astype(np.float)
    img = np.expand_dims(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), axis=0)

    preds_raw = model.predict(img)
    preds_raw = np.squeeze(preds_raw)

    pred_class = np.argmax(preds_raw)
    conf_class = preds_raw[pred_class]

    preds_list.append(pred_class)
    conf_list.append(conf_class)

    print('Processed: {0:d}/{1:d}'.format(i, total_img_count))
