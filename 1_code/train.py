from config import *
from data import get_imgpath_labels
from imageGenerator import ImageGenerator
from model import build_model

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard


class TrainModel(object):
	"""class for TrainModel"""

	def __init__(self, matfile_name, val_prec, custom_model=True, fine_tune_path=None):
		super(TrainModel, self).__init__()
		self.matfile_name = matfile_name
		self.val_prec = val_prec

		self.imgnames_list, self.labels_list = get_imgpath_labels(self.matfile_name)

		self.train_filenames, self.val_filenames, self.train_labels, self.val_labels = \
			train_test_split(self.imgnames_list, self.labels_list, test_size=self.val_prec)

		self.trainGenerator = ImageGenerator(self.train_filenames, labels=self.train_labels)

		self.valGenerator = ImageGenerator(self.val_filenames, labels=self.val_labels)

		if fine_tune_path is None:
			self.model = build_model(custom_model=custom_model)
		else:
			print('Loading pre-trained model!')
			self.model = load_model(fine_tune_path)
		print('Model Summary: ')
		print(self.model.summary())


	def train(self, epochs=200):

		model_names = '../2_outputs/trainedModels/model_{epoch:03d}_{val_acc:.03f}.hdf5'
		model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1,
			save_best_only=True)

		csv_logger = CSVLogger('../2_outputs/logs/train.log')

		tensor_board = TensorBoard(log_dir='../2_outputs/tensorBoard/', histogram_freq=0, 
			write_graph=True, write_images=True)

		callbacks = [model_checkpoint, csv_logger, tensor_board]

		self.model.fit_generator(
			generator = self.trainGenerator,
			validation_data = self.valGenerator,
			steps_per_epoch = len(self.train_filenames) // BATCH_SIZE,
			epochs = epochs,
			verbose = 1,
			callbacks = callbacks
		)

def main():
	matfile_name = 'devkit/cars_train_annos.mat'
	validation_perc = 0.2

	# fine_tune_path = '../2_outputs/trainedModels/model_022_0.128.hdf5'
	fine_tune_path = None

	trainModel = TrainModel(matfile_name, validation_perc, custom_model=False,
		fine_tune_path=fine_tune_path)

	trainModel.train()


if __name__ == '__main__':
	main()
