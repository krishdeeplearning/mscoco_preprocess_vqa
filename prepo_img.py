import numpy as np
import json
import h5py
import os
from scipy.misc import imread



def prepare_vgg19_features(limit=1000):
	print "starting vgg19 image preprocessing"
	from keras.applications.vgg19 import VGG19
	from keras.preprocessing import image
	from keras.applications.vgg19 import preprocess_input
	from keras.models import Model

	model = VGG19(weights='imagenet', include_top=False)

	data_ques_json = 'data/vqa_data_prepro.json'
	with open(data_ques_json, 'r') as an_file:
	    ques_json_data = json.loads(an_file.read())

	hf = h5py.File('img_prepro.h5', 'w')

	total_count = 0
	count = 0
	imgs = []
	final_features = np.empty((0,512,14,14))
	while total_count < len(ques_json_data['unique_img_train']):
		img_path = ques_json_data['unique_img_train'][count]
		img = image.load_img(img_path, target_size=(448, 448))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		imgs.append(x)
		count = count + 1
		total_count = total_count + 1
		if(count == limit):
			imgs = np.vstack(tuple(imgs))
			features = model.predict(imgs)
			final_features = np.vstack((final_features,features))
			count = 0 
			imgs = []
	print final_features.shape
	hf.create_dataset('train_images', data=final_features)
	print "Finished preprocessing"


prepare_vgg19_features()