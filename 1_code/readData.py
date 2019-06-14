import os

import numpy as np
from mat4py import loadmat
from config import *


def get_imgpath_labels(matfile_name, debug=False):
	matpath = os.path.join(DATA_DIR, matfile_name)
	data = loadmat(matpath)
	annos = data['annotations']

	meta_path = os.path.join(DATA_DIR, 'devkit/cars_meta.mat')
	meta = loadmat(meta_path)


	imgname_list = annos['fname']
	labels = [l-1 for l in annos['class']]
	class_names = meta['class_names']

	if debug:
		return imgname_list[0:500], labels[0:500]

	return imgname_list, labels, class_names



