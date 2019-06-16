import os

import cv2
import numpy as np
from config import *
from data import get_imgpath_labels
from keras.models import load_model
from mat4py import loadmat

from sklearn.metrics import accuracy_score, precision_score, recall_score


class TestModel(object):
	"""Class for testing the trained model with an input image"""

	def __init__(self, model_path):
		super(TestModel, self).__init__()
		self.model_path = model_path

		self.model = load_model(model_path)

		class_names_path = '../0_data/devkit/cars_meta.mat'

		if not os.path.exists(class_names_path):
			self.class_names = None
		else:
			self.class_names = loadmat(class_names_path)['class_names']

	def get_preds(self, img_path):
		if not os.path.exists(img_path):
			raise Exception('Image not found on the specified path {0:s}'.format(filepath))

		img = cv2.imread(img_path)
		img = np.expand_dims(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), axis=0)

		preds_raw = self.model.predict(img)

		preds_raw = np.squeeze(preds_raw)

		pred_class = np.argmax(preds_raw)
		conf_class = preds_raw[pred_class]

		if self.class_names is None:
			class_name = None
		else:
			class_name = self.class_names[pred_class]

		return pred_class, conf_class, class_name



def evaluate():
	model_path = '../2_outputs/trainedModels/model.h5'

	testModel = TestModel(model_path)

	test_matfile = 'devkit/cars_test_annos.mat'

	img_names_list, labels_list = get_imgpath_labels(test_matfile, train=False)

	pred_class_list = []
	conf_list = []

	total_img_count = len(img_names_list)

	for i, filename in enumerate(img_names_list):
		filepath = os.path.join(DATA_DIR, IMG_DIR, filename)

		pred_class, conf_class, class_name = testModel.get_preds(filepath)

		pred_class_list.append(pred_class)
		conf_list.append(conf_class)

		print('Processed: {0:d}/{1:d}'.format(i, total_img_count))

	print('Accuracy: ', accuracy_score(labels_list, pred_class_list))
	print('Precision: ', precision_score(labels_list, pred_class_list))
	print('Recall: ', recall_score(labels_list, pred_class_list))

if __name__ == '__main__':
	evaluate()
