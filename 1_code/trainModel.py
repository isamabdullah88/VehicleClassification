from config import *
from data import get_imgpath_labels
from imageGenerator import ImageGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import load_model
from model import build_model
from sklearn.model_selection import train_test_split


class TrainModel(object):
    """
    TrainModel class. Handles building the model, doing data augmentation, and
    then training it.
    """

    def __init__(self, matfile_name, val_prec, model_type=None, fine_tune_path=None):
        """
        ARGS:
        matfile_name: Name of the annotation mat file. This should exactly be formatted as in
        the dataset.
        val_perc: The percentage of dataset used for validation of model.
        model_type: String specifying type of model. 'custom_model', 'VGG', or 'ResNet'.
        Default is 'ResNet'
        fine_tune_path: The path of pretrained model, if you want to transfer learn or
        fine-tune
        """
        super(TrainModel, self).__init__()
        self.matfile_name = matfile_name
        self.val_prec = val_prec

        self.imgnames_list, self.labels_list = get_imgpath_labels(self.matfile_name)

        self.train_filenames, self.val_filenames, self.train_labels, self.val_labels = \
            train_test_split(self.imgnames_list, self.labels_list, test_size=self.val_prec)

        self.trainGenerator = ImageGenerator(self.train_filenames, labels=self.train_labels)

        self.valGenerator = ImageGenerator(self.val_filenames, labels=self.val_labels)

        if fine_tune_path is None:
            self.model = build_model(model_type=model_type)
        else:
            print('Loading pre-trained model!')
            self.model = load_model(fine_tune_path)
        print('Model Summary: ')
        print(self.model.summary())

    def train(self, epochs=200):
        """
        ARGS:
        epochs: The number of epochs to train the model on.
        Other parameters are specified in the config.json file.
        """
        model_names = '../2_outputs/trainedModels/model_{epoch:03d}_{val_acc:.03f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1,
                                           save_best_only=True)

        csv_logger = CSVLogger('../2_outputs/logs/train.log')

        tensor_board = TensorBoard(log_dir='../2_outputs/tensorBoard/', histogram_freq=0,
                                   write_graph=True, write_images=True)

        callbacks = [model_checkpoint, csv_logger, tensor_board]

        self.model.fit_generator(
            generator=self.trainGenerator,
            validation_data=self.valGenerator,
            steps_per_epoch=len(self.train_filenames) // BATCH_SIZE,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks
        )


def main():
    """
    Shows sample code to train a model.
    """
    matfile_name = 'devkit/cars_train_annos.mat'
    validation_perc = 0.2

    # fine_tune_path = '../2_outputs/trainedModels/model_022_0.128.hdf5'
    fine_tune_path = None
    model_type = 'resnet152'

    trainModel = TrainModel(matfile_name, validation_perc, model_type=model_type,
                            fine_tune_path=fine_tune_path)

    trainModel.train()


if __name__ == '__main__':
    main()
