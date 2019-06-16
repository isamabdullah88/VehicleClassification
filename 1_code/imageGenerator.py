import os

import random
import cv2
import numpy as np
from config import *
from keras.utils import Sequence, to_categorical

from skimage import transform
from skimage import util


class ImageGenerator(Sequence):
	def __init__(self, imgpath_list, labels=None, class_names=None, test=False):
		self.IMG_SIZE = IMG_SIZE
		self.BATCH_SIZE = BATCH_SIZE
		self.DATA_DIR = DATA_DIR
		self.IMG_DIR = IMG_DIR
		self.NUM_CLASSES = NUM_CLASSES

		self.img_filenames = imgpath_list
		self.labels = labels
		self.class_names = class_names
		self.test = test

	def __len__(self):
		return int(np.ceil(len(self.img_filenames) / self.BATCH_SIZE))


	def augment_img(self, img):
		"""
		Augment img with different methods.
		"""
		# Random rotation
		if random.random() >= 0.5:
			rand_deg = random.uniform(-25, 25)
			img = transform.rotate(img, rand_deg)

		# Random noise
		if random.random() >= 0.5:
			img = util.random_noise(img)

		# Randomly filp horizontally
		if random.random() >= 0.5:
			img = img[:,::-1]

		return img



	def __getitem__(self, idx):
		idx_strt, idx_end = idx * self.BATCH_SIZE, (idx + 1) * self.BATCH_SIZE
		batch_name = self.img_filenames[idx_strt:idx_end]

		# Get image data
		data_batch = []
		for i, filename in enumerate(batch_name):
			filepath = os.path.join(self.DATA_DIR, self.IMG_DIR, filename)
			if not os.path.exists(filepath):
				raise Exception('Image not found on the specified path {0:s}'.format(filepath))

			img = cv2.imread(filepath).astype(np.float)

			# Perfrom random image augmentation
			img = self.augment_img(img)

			img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

			data_batch.append(img)
		data_batch = np.array(data_batch).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 3) / 255.

		# Get labels
		if self.labels is not None:
			label_batch = self.labels[idx_strt:idx_end]
			labels_hot_enc = to_categorical(label_batch, self.NUM_CLASSES)

		return data_batch, labels_hot_enc
