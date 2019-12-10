import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import argparse

from preprocess import get_data

parser = argparse.ArgumentParser(description='ASSIGNMENT')

parser.add_argument('--train-csv-path', type=str, default='MURA-v1.1/train_image_paths.csv',
					help='Path to the CSV list of folder sorted images')

parser.add_argument('--test-csv-path', type=str, default='MURA-v1.1/valid_image_paths.csv',
					help='Path to the CSV list of folder sorted images')

parser.add_argument('--warp-size', type=int, default=128,
                    help='Shape we warp images to')

args = parser.parse_args()


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

def init_model(img_size):
    model = tf.keras.Sequential([
        Conv2D(16, (2,2), activation='relu'),
        MaxPool2D(),
        Conv2D(20, (2,2), activation='relu'),
        MaxPool2D(),
        Conv2D(20, (2,2), activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(35, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    return model

def train_model(model, train_inputs, train_labels):
    adam = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    print("ENTRY SHAPE", train_inputs[0].shape)
    model.fit(train_inputs, train_labels, batch_size=64, epochs=10)

def test_model(model, test_inputs, test_labels):
    return model.evaluate(test_inputs, test_labels, batch_size=32)

def main():
    train_labels, train_images = get_data(args.train_csv_path, args.warp_size)
    test_labels, test_images = get_data(args.test_csv_path, args.warp_size)

    model = init_model(args.warp_size)
    train_model(model, train_images, train_labels)
    test_model(model, test_images, test_labels)

if __name__ == '__main__':
	main()

