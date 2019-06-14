import json

config = json.load(open('config.json'))

DATA_DIR = config['DATA_DIR']
NUM_CLASSES = config['NUM_CLASSES']
BATCH_SIZE = config['BATCH_SIZE']
IMG_SIZE = config['IMG_SIZE']
IMG_DIR = config['IMG_DIR']
