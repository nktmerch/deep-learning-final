import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    MaxPool2D,
    BatchNormalization,
    Activation,
    Conv2D,
    Dropout,
    Concatenate,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense
)

from preprocess import get_data

parser = argparse.ArgumentParser(description='ASSIGNMENT')

parser.add_argument('--train-csv-path', type=str, default='MURA-v1.1/train_image_paths.csv',
					help='Path to the CSV list of folder sorted images')

parser.add_argument('--test-csv-path', type=str, default='MURA-v1.1/valid_image_paths.csv',
					help='Path to the CSV list of folder sorted images')

parser.add_argument('--warp-size', type=tuple, default=(128,128),
                    help='Shape we warp images to')
parser.add_argument('--growth-k', type=int, default=48,
                    help='growth of output channel')
parser.add_argument('--drop-rate', type=float, default=.2,
                    help='for dropout lol')
parser.add_argument('--batch-size', type=int, default=100,
                    help='literally the batch size')
args = parser.parse_args()

def postfix(name):
    return lambda x: '{0}_{1}'.format(name, x)

def bn_relu_conv(x0, k, drop_rate, training, name):
    pfname = postfix(name)

    x1 = BatchNormalization(name = pfname('bn0'))(
        x0, training = training)
    x1 = Activation('relu', name = pfname('rl0'))(x1)
    x1 = Conv2D(4 * k, 1, padding='SAME', use_bias=False, name = pfname('cv0'))(x1)
    x1 = Dropout(drop_rate, name = pfname('do0'))(
        x1, training = training)

    x1 = BatchNormalization(name = pfname('bn1'))(
        x1, training = training)
    x1 = Activation('relu', name = pfname('rl1'))(x1)
    x1 = Conv2D(k, 3, padding='SAME', use_bias=False, name = pfname('cv1'))(x1)
    x1 = Dropout(drop_rate, name = pfname('do1'))(
        x1, training = training)
    
    return Concatenate(axis=3, name = pfname('cc'))([x0, x1])

def dense_block(x, k, num_bn_relu_conv, drop_rate, training, name):
    pfname = postfix(name)

    for i in range(num_bn_relu_conv):
        x = bn_relu_conv(x, k, drop_rate, training, pfname('brc{}'.format(i)))
    return x

def transition_layer(x, k, drop_rate, training, name):
    pfname = postfix(name)

    x = BatchNormalization(name = pfname('bn'))(x)
    x = Activation('relu', name = pfname('rl'))(x)
    x = Conv2D(k, 1, padding='SAME', use_bias=False, name = pfname('cv'))(x)
    x = Dropout(drop_rate, name = pfname('do'))(
        x, training)
    x = AveragePooling2D(2, 2, 'VALID')(x)
    return x

def DenseNet(shape, k, drop_rate, training):
    image_input = Input(shape=shape)
    x = Conv2D(2 * k, 7, 2, padding = 'SAME', use_bias = False, name = 'cv0')(
        image_input)
    x = MaxPool2D(pool_size = 3, strides = 2, padding = 'VALID', name = 'mp0')(x)

    x = dense_block(x, k, 6, drop_rate, training, 'db0')

    x = transition_layer(x, k, drop_rate, training, 'tl1')
    x = dense_block(x, k, 12, drop_rate, training, 'db1')

    x = transition_layer(x, k, drop_rate, training, 'tl2')
    x = dense_block(x, k, 24, drop_rate, training, 'db2')

    x = transition_layer(x, k, drop_rate, training, 'tl3')
    x = dense_block(x, k, 16, drop_rate, training, 'db3')

    x = BatchNormalization(name = 'bnf')(x, training)
    x = Activation('relu', name = 'rl')(x)
    x = GlobalAveragePooling2D(name = 'gp')(x)
    x = Flatten()(x)
    # I changed this from 2 -> 1 to use binary crossentropy loss, not sure what's better
    x = Dense(1)(x) 

    return Model(inputs=image_input, outputs=x, name='densenet')

def main(): 
    # Preprocess data 
    train_labels, train_images = get_data(args.train_csv_path, args.warp_size)
    test_labels, test_images = get_data(args.test_csv_path, args.warp_size)

    # Convert to tensors
    train_labels = tf.convert_to_tensor(train_labels)
    train_images = tf.convert_to_tensor(train_images)

    test_labels = tf.convert_to_tensor(test_labels)
    test_images = tf.convert_to_tensor(test_images)

    # Instantiate, compile, and train
    input_shape = (args.warp_size, args.warp_size, )
     # TODO: How do we make this false later?
    model = DenseNet((128, 128, 1), args.growth_k, args.drop_rate, True)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(train_images, train_labels, batch_size=args.batch_size,
                validation_data = (test_images, test_labels))

if __name__ == '__main__':
	main()

