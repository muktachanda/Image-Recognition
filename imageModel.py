# The dataset used in this file is CIFAR 10 and is cited as follows.
## Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
## https://www.cs.toronto.edu/~kriz/cifar.html

# Dataset format is:
## Images:
### (numImages, height=32, width=32, dimColors=3)
### [[r, g, b], [r, g, b], ... numImages times]

## Labels:
### (numImages, labelNum)
### Each label number is assigned a class in the exact order as class_names

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models

# Get data and normalize for pixel values
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

# Visualize images
## class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
## for i in range(16):
    ## plt.subplot(4, 4, i+1)
    ## plt.xticks([])
    ## plt.yticks([])
    ## plt.imshow(training_images[i], cmap=plt.cm.binary)
    ## plt.xlabel(class_names[training_labels[i][0]])
## plt.show()

# To improve performance, but risks accuracy
## training_images = training_images[:20000]
## training_labels = training_labels[:20000]
## testing_images = testing_images[:4000]
## testing_labels = testing_labels[:4000]

# Defining convolutional neural network:
## Layers: convolutional->pooling->convolutional->pooling->
## convolutional->flatten->dense(relu)->dense(softmax)
### Convolutional:
### This layer has filters that scan th einput images with a small window
### called a kernel to extract features such as edges, textures, and shapes.
### With more filters, more complex shapes can be detected.
### ReLU (rectified linear unit): reLU(x) = max(0,x), deactivates negative values,
### captures complex patterns within the images.
## MaxPooling:
### Reduces the spatial dimensions of the feature maps and preserves the most
### important features, helps in generalizing.
## Flatten:
### 2D feature maps are converted to 1D vector to connect the convolutional layers
### to the dense layers for classification.
## Dense Layer 1:
### Learns higher level representations by combining the features extracted from
### the previous layers.
### Dense Layer 2:
### Outputs a probability distribution over the 10 classes.
### Softmax: input = [x1, x2, x3], output = [exp(x1), exp(x2), exp(x3)], normalized.
### Amplifies the difference between input values and provides a way to interpret
### the output of the neural network as class probabilities.

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compiling and fitting the model:
## Optimizer:
### Adam (Adaptive Moment Estimation): adapts the learning rate for each 
### parameter individually based on the first and second moments of the gradients.
### So, converges faster and handles sparse gradients effeciently.
## Loss:
### This loss function measures the discrepancy between the predicted outputs and 
### true labels.
## Metrics:
### Accuracy is chosen as the metric to classify, which measures the proportion
### of correctly classified samples.
## Epochs: 
### Goes over the training model 10 times.
## Validation Data:
### The model's performance is evaluates using the testing set.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')
