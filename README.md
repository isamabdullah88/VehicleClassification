# Vehicle Classification

This repository contains code and documentation for training, testing and evaluating deep learning models [of various architectures] on a [vehicle/cars dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The dataset contains 196 classes of cars which constitute different make, models, colors and other details of cars.

## Getting Started
This repository used keras with tensorflow as backend.

### Prerequisites
Following are the required libraries for repo. Omit "gpu" for tensorflow, if you want only the cpu version.
```bash
opencv-4.1.0
tensorflow-1.13.0-rc0
keras-2.2.4
numpy-1.16
mat4py (to read .mat annotation files)
scikit-learn
scikit-image (for data augmentation)
```
### Installing
A virtual environment is highly recommended to install these packages. That will make the installation process clean and isolated. 
```bash
pip install opencv-python==4.1.0
pip install tensorflow-gpu==1.13.0-rc0
pip install keras==2.2.4
pip install numpy
pip install mat4py
pip install sklearn
pip install scikit-image
```

## Usage
- To use first clone the repository.
- Please run all the commands inside the **1_code/** folder of the repository.

### Testing/Evaluation
- Download the pre-trained model from this [link](). Place the downloaded file in **2_outputs/trainedModels/** folder in the project.
- In order to test the pre-trained model, this [notebook](https://github.com/isamabdullah88/VehicleClassification/blob/master/1_code/demo.ipynb) shows simple demo code.
- In order to evaluate[get metrics], use the evaluate method in  [evaluate.py](). Please do not forget to place the images folder and annotations mat file in the **0_data** folder of the project. Remember to match the annotation matfile exactly with the given matfiles from the dataset.

### Training
To train models from scratch or fine_tune/transfer_learn, please look at the [train.py](https://github.com/isamabdullah88/VehicleClassification/blob/master/1_code/train.py) file.

Here is a short snippet, which can be used for training models.
```python
from train import TrainModel

# path relative to the folder "0_data".
matfile_name = 'devkit/cars_train_annos.mat'

# Percentage of dataset used for validation
validation_perc = 0.2

# Uncomment if you want to fine-tune from already trained model.
# fine_tune_path = '../2_outputs/trainedModels/model_022_0.128.hdf5'
fine_tune_path = None

# Instantiate training class with arguments
trainModel = TrainModel(matfile_name, validation_perc, custom_model=False,
fine_tune_path=fine_tune_path)

# Start training
trainModel.train()
```
If you wish to see the training graphs, [training/val loss, training/val acc], invoke following command to initiate tensorboard.

```bash
tensorboard --logdir ../2_ouputs/tensorBoard
```
## Experimentation
### Details
- All the images are resized to (224,224) in order to gain optimal performance with transfer learning.
- The training has been done by transfer learning/fine tuning from already existing models (e.g. VGG16), which have been pre-trained on millions of imagenet dataset.
- A small pitfall in the dataset is that labels start from 1, so it has been handled by making it start from 0.
- Since the dataset is small, in order to improve generalizability, data augmentation has been performed. Can be seen in the [imageGenerator.py](https://github.com/isamabdullah88/VehicleClassification/blob/master/1_code/imageGenerator.py).

### Results
- Training and experimenting with various architectures with given compute resources, an accuracy of 88% has been obtained on a 20% validation set from training data.
- Metrics like precision, recall and AUC curve can be obtained.

### Improvements
-  The most straightforward way of improvement is by increasing the dataset size.
- One can also experiment with more diverse set of architectures.

## Pull Requests
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[GNU](https://github.com/isamabdullah88/VehicleClassification/blob/master/LICENSE)


