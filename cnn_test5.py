#import tensorflow as tf

#from tensorflow.keras import datasets, layers, models

from tensorflow.keras import layers, models

import glob
import numpy

#(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#train_images = train_images.reshape((60000, 28, 28, 1))
#test_images = test_images.reshape((10000, 28, 28, 1))

files = glob.glob('csv3/*.csv')
files2 = glob.glob('csv4/*.csv') 
NUM = int(len(files)/3)
NUM2 = int(len(files2)/3)
SIZE = 28
data1 = numpy.zeros((NUM, SIZE, SIZE, 3))
data2 = numpy.zeros((NUM2, SIZE, SIZE, 3))

i = j = 0
for csv_name in files:
    data1[i, :, :, j] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
    j = j + 1
    if j == 3:
        j = 0
        i = i + 1

#data1 = data1.reshape((3, 28, 28, 3))

i = j = 0
for csv_name in files2:
    data2[i, :, :, j] = numpy.genfromtxt(csv_name, delimiter=",", skip_header=0, skip_footer=0, usecols=(range(0, SIZE)))
    j = j + 1
    if j == 3:
        j = 0
        i = i + 1

#data2 = data1.reshape((3, 28, 28, 3))

train_images = data1
test_images = data2

data3 = numpy.array([0, 1, 2])
data4 = numpy.array([0, 1])

train_labels = data3
test_labels = data4

# ピクセルの値を 0~1 の間に正規化
train_images, test_images = train_images / 255.0, test_images / 255.0


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)