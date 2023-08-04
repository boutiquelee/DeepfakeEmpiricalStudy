import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from sklearn.metrics import classification_report
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.layer_utils import print_summary
from tensorflow.python.keras.models import Model as KerasModel
from tensorflow.python.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import efficientnet.tfkeras as effnet

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
    class_mode = 'categorical',
    shuffle = True,
    seed = 32)
validation_generator = validation_datagen.flow_from_directory(
    '../dataset/CELEB/val', 
    target_size = (256,256), 
    class_mode = 'categorical',
    shuffle = True,
    seed = 32)


x = train_generator.__getitem__(0)
print(len(train_generator.labels), train_generator.labels)
print(x[1].shape)
print(x[1])
plt.imshow(x[0][7])
plt.title(x[1][7])

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
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
model_checkpoint_callback = ModelCheckpoint(
    filepath="../model/EfficientNet1.h5",
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
csv_logger = CSVLogger("../model/EfficientNet1.csv", append=True, separator=',')
print_summary(model, line_length=150, positions=None, print_fn=None)

model.fit_generator(train_generator,
                    verbose=1,
                    epochs=100,
                    validation_data=validation_generator,
                    callbacks=[model_checkpoint_callback,csv_logger])
model.save_weights("../model/EfficientNet1_100.h5")