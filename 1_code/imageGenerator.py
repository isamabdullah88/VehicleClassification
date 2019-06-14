import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from keras.utils import Sequence, to_categorical

from config import *


class ImageGenerator(Sequence):
	def __init__(self, imgpath_list, labels=None, class_names=None):
		self.IMG_SIZE = IMG_SIZE
		self.BATCH_SIZE = BATCH_SIZE
		self.DATA_DIR = DATA_DIR
		self.IMG_DIR = IMG_DIR
		self.NUM_CLASSES = NUM_CLASSES

		self.img_filenames = imgpath_list
		self.labels = labels
		self.class_names = class_names


	def __len__(self):
		return int(np.ceil(len(self.img_filenames)/self.BATCH_SIZE))


	def __getitem__(self, idx):
		idx_strt, idx_end = idx*self.BATCH_SIZE, (idx+1)*self.BATCH_SIZE
		batch_name = self.img_filenames[idx_strt:idx_end]

		# Get image data
		data_batch = []
		for i,filename in enumerate(batch_name):
			filepath = os.path.join(self.DATA_DIR, self.IMG_DIR, filename)
			if not os.path.exists(filepath):
				raise Exception('Image not found on the specified path {0:s}'.format(filepath))

			img = cv2.imread(filepath).astype(np.float)
			img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

			data_batch.append(img)
		data_batch = np.array(data_batch).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3)
		# Get labels
		if self.labels is not None:
			# subtract 1 from each class id to start from 0
			label_batch = self.labels[idx_strt:idx_end]
			labels_hot_enc = to_categorical(label_batch, self.NUM_CLASSES)

		# for i,filename in enumerate(batch_name):
		# 	img = data_batch[i,:,:,:].reshape(self.IMG_SIZE, self.IMG_SIZE, 3)
		# 	label = label_batch[i]

		# 	cv2.putText(img, text=self.class_names[label], org=(0,20), 
		# 		fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,255))
		# 	cv2.imwrite(filename, img)


		return data_batch, labels_hot_enc