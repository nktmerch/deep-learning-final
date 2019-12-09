import pickle
import numpy as np
import tensorflow as tf
import os
import csv
import imageio as io

def get_label(file_path):
	return "positive" in file_path

def get_image(file_path):
	gray = io.imread(file_path, as_gray=True)
	image = np.zeros([gray.shape[0], gray.shape[1], 1])
	image[:,:,0] = gray
	print(image.shape)
	return image

def source_label_and_image(path_to_csv):
	labels = []
	images = []
	with open(path_to_csv) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		i = 0
		for row in csv_reader:
			file_path = row[0]
			labels.append(get_label(file_path))
			images.append(get_image(file_path))
			i += 1
			if i == 100: break
	
	labels = np.asarray(labels).astype(np.int)
	images = np.asarray(images)
	return labels, images



