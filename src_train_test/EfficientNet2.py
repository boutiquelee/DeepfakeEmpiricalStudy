from sklearn.metrics import classification_report
from tensorflow.python.keras import optimizers, regularizers, Model
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Layer, \
    Reshape, Concatenate, LeakyReLU
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pprint import pprint
from Utility_functions import create_sequence, FreezeBatchNormalization
import pandas as pd
import efficientnet.tfkeras as effnet

random.seed(32)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)
sess.run(tf.global_variables_initializer())


def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height, img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle = True,
                                          seed = 32)
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height, img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle = True,
                                          seed = 32)
    i = 0
    while (True):
        if i < len(genX1.filepaths):
            X1i = genX1.next()
            yield X1i[0], X1i[1]

        if i < len(genX2.filepaths):
            X2i = genX2.next()
            yield X2i[0], X2i[1]
        i += 1

batch_size = 64
img_height = 256
img_width = 256
train_datagen = ImageDataGenerator(rescale = 1. / 255)
validation_datagen = ImageDataGenerator(rescale = 1. / 255)
train_generator = generate_generator_multiple(generator = train_datagen,
                                              dir1 = '../dataset/CELEB/train',
                                              dir2 = '../dataset/FSNT/train',
                                              batch_size = batch_size,
                                              img_height = img_height,
                                              img_width = img_width)
validation_generator = generate_generator_multiple(validation_datagen,
                                                   dir1 = '../dataset/CELEB/val',
                                                   dir2 = '../dataset/FSNT/val',
                                                   batch_size = batch_size,
                                                   img_height = img_height,
                                                   img_width = img_width)


def get_pred(predictions):
    pred = []
    for p in predictions:
        if p < 0.50:
            pred.append(0)
        else:
            pred.append(1)
    return pred


input_tensor = Input(shape=(256, 256, 3))
model=effnet.EfficientNetB0(input_tensor=input_tensor,
    include_top=True, weights=None, classes=2)
model.load_weights("../model/EfficientNet1.h5")
is_training=True
top_k_layers=6
model,df=FreezeBatchNormalization(is_training,top_k_layers,model)
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
model_checkpoint_callback = ModelCheckpoint(
    filepath="../model/EfficientNet2.h5",
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
csv_logger = CSVLogger("../model/EfficientNet2.csv", append=True, separator=',')
print_summary(model, line_length=150, positions=None, print_fn=None)


total_train = 4000
total_val =  4000
model.fit_generator(train_generator,
                    verbose = 1,
                    steps_per_epoch = int(total_train / batch_size),
                    epochs = 50,
                    validation_data = validation_generator,
                    validation_steps = int(total_val / batch_size),
                    callbacks=[model_checkpoint_callback,csv_logger])
model.save_weights("../model/EfficientNet2_50.h5")
