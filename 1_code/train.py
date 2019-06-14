from config import *
from data import get_imgpath_labels
from imageGenerator import ImageGenerator
from model import build_model

from sklearn.model_selection import train_test_split

matfile_name = 'devkit/cars_train_annos.mat'

imgname_list, labels = get_imgpath_labels(matfile_name)

print(len(labels))

train_filenames, val_filenames, train_labels, val_labels = train_test_split(imgname_list, labels, test_size=0.2)

trainGenerator = ImageGenerator(train_filenames, labels=train_labels)

valGenerator = ImageGenerator(val_filenames, labels=val_labels)

model = build_model()

print('model summ: ')
print(model.summary())

epochs = 200

model.fit_generator(
    generator=trainGenerator,
    steps_per_epoch=len(train_filenames) // BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    validation_data=valGenerator,
    shuffle=True,
    use_multiprocessing=False
)

model.save('model.h5')
