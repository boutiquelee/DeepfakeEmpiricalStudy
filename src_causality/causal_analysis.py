import os
import time
import numpy as np
import random
from tensorflow import set_random_seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)

from keras.preprocessing.image import ImageDataGenerator
from causal_inference import causal_analyzer
import utils
import subprocess
from classifiers import *

DEVICE = '0'

IMG_ROWS = 256
IMG_COLS = 256
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 2

BATCH_SIZE = 64
LR = 0.1
STEPS = 1000
NB_SAMPLE = 1920
MINI_BATCH = NB_SAMPLE // BATCH_SIZE
INIT_COST = 1e-3
print(NB_SAMPLE)

PATIENCE = 5
COST_MULTIPLIER = 2
SAVE_LAST = False

EARLY_STOP = True
EARLY_STOP_THRESHOLD = 1.0
EARLY_STOP_PATIENCE = 5 * PATIENCE

UPSAMPLE_SIZE = 1
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)

def build_data_loader(dataset_dir):
    dataGenerator = ImageDataGenerator(rescale=1. / 255)
    generator = dataGenerator.flow_from_directory(
        dataset_dir,
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary',
        subset='training')
    return generator

def trigger_analyzer(analyzer, train_gen, test_gen):
    visualize_start_time = time.time()
    analyzer.analyze(train_gen, test_gen)
    visualize_end_time = time.time()
    return

def start_analysis():
    print('loading model')
    model_file = '../../model/ShallowNet1.h5'
    model = ShallowNetV3()
    model.load(model_file)
    print(model_file)

    dataset_dirs = ['../../dataset/CELEBV2/test', '../../dataset/FS/test', '../../dataset/NT/test', '../../dataset/DF/test', '../../dataset/DFD/test']

    for dataset_dir in dataset_dirs:
        print(f'loading dataset: {dataset_dir}')
        data_generator = build_data_loader(dataset_dir)

        analyzer = causal_analyzer(
            model,
            data_generator,
            data_generator,
            input_shape=INPUT_SHAPE,
            init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
            mini_batch=MINI_BATCH,
            upsample_size=UPSAMPLE_SIZE,
            patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
            img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
            save_last=SAVE_LAST,
            early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
            early_stop_patience=EARLY_STOP_PATIENCE)

        trigger_analyzer(analyzer, data_generator, data_generator)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils.fix_gpu_memory()
    start_analysis()
    subprocess.run(["python", "neuron.py"])

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
