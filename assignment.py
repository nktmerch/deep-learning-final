import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from preprocess import source_label_and_image


"""
ASSIGNMENT 2 MODEL ARCHITECTURE:

Image data is of shape [batch_size, width, heigh, channels] /
where the three channels are probably R, B, and G values from 0 to 1 / 
and the width and height are both 32.

Our first layer is a convolution of shape [5, 5, 3, 16] / 
representing 16 filters of shape [5, 5, 3] and a stride of [1, 1, 1, 1] | 
We then apply a ReLU activation and a max pooling of shape [3, 3] /
and stride [2, 2]. I am unclear what the max pooling does.

Our second layer is a convolution of shape [5, 5, 16, 20] |
We then apply the same ReLU activation and max pooling.

Our third layer is a convolution of shape [5, 5, 20, 20] /
followed by a ReLU activation.

We then flatten our filters into a single vector for the forward pass.

The forward pass consists of two dense layers with dropouts /
and ReLU activations, and a final dense layer with a softmax / 
outputting our final probabilities.
"""

"""
TODO: 
1) Download Stanford bone abnormality data set |
2) Re-implement Assignment 2 using tf.keras.Sequential |
3) Check old model performance on data set |
4) Modify model architecture based on best practice |
"""

def init_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(16, (2,2), activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(20, (2,2), activation='relu')) # The (2,2) here might be wrong
    model.add(MaxPool2D())
    model.add(Conv2D(20, (2,2), activation='relu'))
    model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(35, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    return model

def train_model(model, train_inputs, train_labels):
    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    print("ENTRY SHAPE", train_inputs[0].shape)
    model.fit(train_inputs, train_labels, batch_size=64, epochs=10)

def test_model(model, test_inputs, test_labels):
    return model.evaluate(test_inputs, test_labels, batch_size=32)

def main():
    # print("Training")
    # train_inputs, train_labels = get_data('MURA-v1.1/train_image_paths.csv','MURA-v1.1/train_labeled_studies.csv')
    # print("finished training")
    #test_inputs, test_labels = get_data('MURA-v1.1/valid_image_paths.csv', 'MURA-v1.1/valid_labeled_studies.csv')
    labels, images = source_label_and_image('MURA-v1.1/valid_image_paths.csv')
    print(labels)
    print(images)
    model = init_model()
    train_model(model, images, labels)
    #result = test_model(model, test_inputs, test_labels)
    #print(result)
    print("ass!")

if __name__ == '__main__':
	main()

