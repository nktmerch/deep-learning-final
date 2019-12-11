import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from skimage.io import imread
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
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
parser.add_argument('--growth-k', type=int, default=48,
                    help='growth of output channel')
parser.add_argument('--drop-rate', type=float, default=.2,
                    help='for dropout lol')
parser.add_argument('--batch-size', type=int, default=100,
                    help='literally the batch size')
args = parser.parse_args()

def batch_norm(x, training):
    x= layers.BatchNormalization()(x, training = training)
    return x

def leaky_relu(x):
    return tf.nn.leaky_relu(x)

def conv_layer(x, filters, kernel, stride = [1,1]):
    x = Conv2D(filters = filters, kernel_size = kernel, strides = stride, padding = 'same', use_bias = False)(x)
    return x

def dropout(x, training):
    x = Dropout(rate = args.drop_rate)(x, training = training)
    return x

def avg_pool(x, pool_size = [2,2], stride = [2,2], padding = 'valid'):
    x = AveragePooling2D(pool_size = pool_size, strides = stride, padding = padding)(x)
    return x

def concat(list):
    return tf.concat(list, axis = 3)

def max_pool(x, pool_size, strides, padding = 'valid'):
    x = MaxPool2D(pool_size = pool_size, strides = strides, padding = padding)(x)
    return x

def global_avg_pool(x):
    x = GlobalAveragePooling2D()(x)
    return x

class DenseNet(tf.keras.Model):
    def __init__(self, filters):
        super(DenseNet, self).__init__()

        # Hyperparameters
        self.batch_size = 64
        self.filters = filters
        self.learning_rate = 0.001

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def bottleneck_layer(self, x, training):
        x = batch_norm(x, training)
        x = leaky_relu(x)
        x = conv_layer(x,4 * self.filters, kernel =(1,1))
        x = dropout(x, training)

        x = batch_norm(x, training)
        x = leaky_relu(x)
        x = conv_layer(x, filters = self.filters,kernel = (3,3))
        x = dropout(x, training)
        return x

    def transition_layer(self, x, training ):
        x = batch_norm(x, training)
        x = leaky_relu(x)
        x = conv_layer(x, filters = self.filters, kernel = (1,1))
        x = dropout(x, training)
        x = avg_pool(x)
        return x

    def dense_block(self, x, num_layers, training):
        layers = []
        x = self.bottleneck_layer(x, training= training)
        layers.append(x)
        for layer in range(num_layers):
            x = concat(layers) 
            x = self.bottleneck_layer(x, training=training) 
            layers.append(x) 
        x = concat(layers)
        return x

    def call(self, x, training = True):
        x = conv_layer(x, 2 * self.filters, kernel = [7,7], stride = [2,2])
        print("conv1: " + str(x.shape))
        x = max_pool(x, pool_size=[3,3], strides = [2,2])
        print("maxpool: " + str(x.shape))
        x = self.dense_block(x, num_layers = 6, training=training) # !
        print("dense block 1: " + str(x.shape))
        x = self.transition_layer(x, training=training)
        x = self.dense_block(x, num_layers=12, training=training)
        print("dense block 2: " + str(x.shape))
        x = self.transition_layer(x, training=training)
        x = self.dense_block(x, num_layers=24, training=training)
        print("dense block 3 : " + str(x.shape))
        x = self.transition_layer(x, training=training)
        x = self.dense_block(x, num_layers=16, training=training)
        print("dense block 4: " + str(x.shape))

        x = batch_norm(x, training)
        x = leaky_relu(x)
        x = global_avg_pool(x)
        print("final: " + str(x.shape))
        x = layers.Flatten()(x)
        x = layers.Dense(2)(x)
        x = tf.nn.softmax(x)
        return x

    def loss_function(self, labels, probs):
        return self.loss(labels, probs)

    def accuracy(self, labels, probs):
        correct_predictions = tf.equal(labels, tf.argmax(probs, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_data, train_labels):
    for i in range(0, len(train_data), model.batch_size):
        print("Training Batch", i // model.batch_size)
        with tf.GradientTape() as tape:
            batch_data = train_data[i:i+model.batch_size]
            batch_labels = train_labels[i:i + model.batch_size]

            probs = model.call(batch_data)
            loss = model.loss_function(batch_labels, probs)
            print("Loss: ", loss)

        print("Trainable Variables", len(model.trainable_variables))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    

def test(model, test_data, test_labels):

    accuracies = []
    for i in range(0, len(test_data), model.batch_size):
        print("Training Batch", i // model.batch_size)
        batch_data = test_data[i:i + model.batch_size]
        batch_labels = test_labels[i:i + model.batch_size]

        batch_probs = model.call(batch_data)
        accuracies.append(model.accuracy(batch_labels, batch_probs))

    return tf.reduce_mean(accuracies)


def main():
    # Preprocess data
    train_labels, train_images = get_data(args.train_csv_path, args.warp_size)
    test_labels, test_images = get_data(args.test_csv_path, args.warp_size)

    # Convert to tensors
    train_images = tf.convert_to_tensor(train_images)
    test_images = tf.convert_to_tensor(test_images)
    

    # Instantiate model
    model = DenseNet(args.growth_k)

    # Train and test
    train(model, test_images, test_labels)
    accuracy = test(model, test_images, test_labels)
    print("Accuracy: ", accuracy)

if __name__ == '__main__':
	main()

