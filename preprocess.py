import pickle
import numpy as np
import tensorflow as tf
import os
import csv
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='PREPROCESS')

parser.add_argument('--csv-path', type=str, default='MURA-v1.1/valid_image_paths.csv',
					help='Path to the CSV list of folder sorted images')

parser.add_argument('--warp-size', type=int, default=128,
                    help='Shape we warp images to')

parser.add_argument('--img-num', type=int, default=0,
					help='Which image to visualize')

parser.add_argument('--notify-every', type=int, default=1,
					help='How often to update on how many images we have loaded')

args = parser.parse_args()

def get_label(file_path):
	# Janky, but gets the job done
	return int("positive" in file_path)

def get_image(file_path, warp_size):
	image = imread(file_path, as_gray=True)
	# Alternate take: The list of images should not be a
	# numpy array, we should do no warping, and our 
	# network should (somehow) handle images of different
	# sizes.
	warp_width, warp_height = warp_size
	image = resize(image, (warp_width, warp_height))
	image = np.asarray(image, dtype = np.float32)
	image = np.expand_dims(image, axis = 2)
	return image

def get_data(csv_path, warp_size):
	labels = []
	images = []
	with open(csv_path) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		rows = [row for row in csv_reader]
		imgs_read = 0
		for iteration, row in enumerate(rows):
			file_path = row[0]
			labels.append(get_label(file_path))
			images.append(get_image(file_path, warp_size))

			if iteration % args.notify_every == 0:
				print("\rReading in ({0} / {1}) images from {2}"
						.format(imgs_read, len(rows), csv_path), end='\r')
				imgs_read += args.notify_every
	
	print("Succesfully read in {0} images from {1}"
		.format(imgs_read, csv_path))

	return labels, images

def main():
	_, images = get_data(args.csv_path, args.warp_size)
	if args.img_num >= len(images):
		raise Exception("The folder ./valid only contains {0} images, but the index passed was {1}."
							.format(len(images), args.image_num))
	image = images[args.img_num][:,:,0]
	plt.imshow(image, cmap="gray")
	plt.show()

if __name__ == "__main__":
	main()



