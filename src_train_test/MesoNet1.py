import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.metrics import classification_report
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, \
    Reshape, Concatenate, LeakyReLU
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(32)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)
sess.run(tf.global_variables_initializer())


train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
)

validation_datagen = ImageDataGenerator(
    rescale = 1. / 255,
)
train_generator = train_datagen.flow_from_directory(
    '../dataset/CELEB/train', 
    target_size = (256,256), 
    class_mode = 'binary',
    shuffle = True,
    seed = 32)
validation_generator = validation_datagen.flow_from_directory(
    '../dataset/CELEB/val',
    target_size = (256,256), 
    class_mode = 'binary',
    shuffle = True,
    seed = 32)


x = train_generator.__getitem__(0)
print(len(train_generator.labels), train_generator.labels)
print(x[1].shape)
print(x[1])
plt.imshow(x[0][7])
plt.title(x[1][7])


IMGWIDTH = 256


class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    def init_model(self):
        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))

        x1 = Conv2D(8, (3, 3), padding = 'same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x1)

        x2 = Conv2D(8, (5, 5), padding = 'same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x2)

        x3 = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x3)

        x4 = Conv2D(16, (5, 5), padding = 'same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size = (4, 4), padding = 'same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)	
        y = LeakyReLU(alpha = 0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)


model = Meso4().model
model_checkpoint_callback = ModelCheckpoint(
    filepath = "../model/MesoNet1.h5",
    save_weights_only = True,
    monitor = 'val_acc',
    mode = 'max',
    save_best_only = True)
csv_logger = CSVLogger("../model/MesoNet1.csv", append = True, separator = ',')
print_summary(model, line_length = 115, positions = None, print_fn = None)


model.fit_generator(train_generator,
                    verbose = 2,
                    epochs = 100,
                    validation_data = validation_generator,
                    callbacks = [model_checkpoint_callback, csv_logger])
model.save_weights("../model/MesoNet1_100.h5")