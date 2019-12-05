import pickle
import numpy as np
import tensorflow as tf
import os
import csv
try:
    import Image
except ImportError:
    from PIL import Image

def unpickle(file):
	"""
	CIFAR data contains the files data_batch_1, data_batch_2, ..., 
	as well as test_batch. We have combined all train batches into one
	batch for you. Each of these files is a Python "pickled" 
	object produced with cPickle. The code below will open up a 
	"pickled" object (each file) and return a dictionary.

	NOTE: DO NOT EDIT

	:param file: the file to unpickle
	:return: dictionary of unpickled data
	"""
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


def get_data(image_paths, label_paths):
	"""
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param first_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param first_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	"""
	labels = []
	with open(label_paths) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		for row in csv_reader:
			labels.append(int(row[1]))
	labels_one_hot = tf.one_hot(labels, depth =2 )
	images = []
	with open(image_paths) as csv_file:
		csv_reader = csv.reader(csv_file)
		for path in csv_reader:
			img = Image.open(path[0])
			resized_img = np.resize(img, (32,32,1)).astype(np.float32)
			images.append(resized_img)
	images = tf.convert_to_tensor(images)
	print(images.shape)
	print(labels_one_hot.shape)
	return images, labels_one_hot
