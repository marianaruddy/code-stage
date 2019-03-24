# -*- coding: utf-8 -*-
"""

Created on Sat Mar 23 14:35:42 2019

@author: Ju
"""

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
 

df = pd.read_csv('C:\\Users\\\cenpes\\Desktop\\young-people-survey\\datasetmusic.csv', header = 0)
X=df
X= pd.get_dummies(X,drop_first=True)
numpy_matrix = X.as_matrix()

X=numpy_matrix[:, 0:162]
y=numpy_matrix[:, 162:163]

X= pd.DataFrame(X)
y= pd.DataFrame(y)

X_train, X_test, y_train, y_test  = train_test_split( X,y, test_size=0.2, random_state=42)



X_train = X_train.values
X_test = X_test.values

X_train.shape

input_dim = X_train.shape[1]
encoding_dim = 162

input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)

decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 30
batch_size = 32

autoencoder.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
                          

autoencoder = load_model('model.h5')

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');


                    















