
# coding: utf-8

# In[1]:


import keras
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras import regularizers
from keras.optimizers import SGD, Adam
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd


# In[2]:


data = np.load("./Data_6_minute_onlyweld.npz")


# In[3]:


X = data['X']
Y = data['Y']


# In[8]:


print(np.unique(Y, return_counts=True))


# In[4]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)
print(x_train.shape)

num_classes = 2

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1,  random_state=42, stratify=y_train)


np.unique(y_test, return_counts = True)

# In[5]:


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


# In[6]:


def CNN():
	model = Sequential()
	# model.add(Convolution2D(64, 3, 11, 11, border_mode='full'))
	model.add(Conv2D(64, kernel_size=(11, 11),
	                 activation = 'relu',
	                 input_shape = (224, 224, 3), padding="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

	model.add(Conv2D(128, kernel_size=(7, 7), activation = 'relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

	model.add(Conv2D(192, kernel_size= (3, 3), padding="same", activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary()
	return model

model = CNN()
print(model.summary())


# In[7]:


from keras.callbacks import ReduceLROnPlateau

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10, batch_size=32,
         validation_data = (x_valid, y_valid))

model.save('alex-cnn-weld.h5')
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

ypred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
arr= confusion_matrix(y_test.argmax(axis=1),ypred.argmax(axis=1))

from sklearn.metrics import precision_recall_fscore_support
output = precision_recall_fscore_support(y_test.argmax(axis=1), ypred.argmax(axis=1))
print("Precision is:", np.average(output[0]))
print("Recall is:", np.average(output[1]))
print("Fscore is:", np.average(output[2]))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
arr = arr / arr.astype(np.float).sum(axis=1, keepdims=True)
arr = arr.round(decimals=3)
df_cm = pd.DataFrame(arr)
plt.figure(figsize = (10,8))
plt.rcParams.update({'font.size': 16})
sn.heatmap(df_cm, annot=True, cmap = "YlGnBu", fmt='g')
plt.show()
