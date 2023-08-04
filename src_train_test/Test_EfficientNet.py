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


def get_pred(predictions):
    pred = []
    for p in predictions:
        if p < 0.50:
            pred.append(0)
        else:
            pred.append(1)
    return pred

print('EfficientNet1')
print('CELEB_Test')

test_datagen = ImageDataGenerator(rescale=1./255)
input_tensor = Input(shape=(256, 256, 3))
model=effnet.EfficientNetB0(input_tensor=input_tensor,
    include_top=True, weights=None, classes=2)
model.load_weights("../model/EfficientNet1.h5")
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
#print_summary(model, line_length=150, positions=None, print_fn=None)
test_generator = test_datagen.flow_from_directory(
        '../dataset/CELEB/test',
        target_size=(256, 256),
        class_mode='categorical',
        shuffle=False,
        seed=32) 
predictions=model.evaluate_generator(test_generator,verbose=2)
print(predictions)
predictions=model.predict_generator(test_generator,verbose=2)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
report = classification_report(true_classes, predicted_classes, digits=4)
print(report)
