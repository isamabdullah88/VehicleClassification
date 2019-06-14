import os

from config import *
from mat4py import loadmat


def get_imgpath_labels(matfile_name, debug=False, train=True):
    matpath = os.path.join(DATA_DIR, matfile_name)
    data = loadmat(matpath)
    annos = data['annotations']

    meta_path = os.path.join(DATA_DIR, 'devkit/cars_meta.mat')
    meta = loadmat(meta_path)

    imgname_list = annos['fname']

    if debug:
        return imgname_list[0:500], labels[0:500]

    if not train:
        return imgname_list, None

    labels = [l - 1 for l in annos['class']]

    return imgname_list, labels
