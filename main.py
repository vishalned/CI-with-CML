import tensorflow.keras
import matplotlib
import numpy as np
import tensorflow.keras.losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()




#parameters
BATCH_SIZE = 128
NUM_EPOCHS = 5
NUM_CLASSES = 10
INPUT_DIM = (28, 28, 1)

def create_model():
    model = Sequential()
    model.add(Conv2D(16, 3, activation='relu', input_shape = INPUT_DIM))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D((2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


data = mnist.load_data()
(X_train, y_train), (X_test, y_test) = data

n_samples_train = X_train.shape[0]
n_samples_test = X_test.shape[0]
dim = X_train.shape[1]

X_train = (X_train.reshape((n_samples_train, dim, dim, 1)).astype('float32'))/255.
X_test = (X_test.reshape((n_samples_test, dim, dim, 1)).astype('float32'))/255.
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = create_model()

history = model.fit(X_train[:5000, :], y_train[:5000, :], BATCH_SIZE, NUM_EPOCHS, validation_data=(X_test, y_test))

print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('acc.png')
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.close()


