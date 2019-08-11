from utilities import *
import argparse

import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
#from pathlib import Path

def predict(image_path, model = None, topk=3, cat_to_name = None, gpu = False):
	''' Predict the class (or classes) of an image using a trained deep learning model.
	'''
	if gpu:
		#  Stating the current device 
		if gpu and torch.cuda.is_available():
			print ('using gpu')
		elif gpu and not torch.cuda.is_available():
			print ('gpu is not avaliable,  using cpu')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
		print ('using cpu')
		device = torch.device('cpu')
	
	
	# changing image to numpy tensor
	image = process_image(image_path)
	image_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
	model = model.to(device)
	with torch.no_grad():
		output = model.forward(image_tensor)
		
	probs = F.softmax(output,dim=1).topk(topk)
	
	#switching key-value pairs from class_to_idx
	index_to_class = {val: key for key, val in model.class_to_idx.items()}
	probabilities = probs[0].tolist()[0]
	classes = probs[1].tolist()[0]
	flower_type = [cat_to_name[index_to_class[i]] for i in classes]

	return round(probabilities,2) ,flower_type



if __name__ == '__main__':
	# adapting argparse 
	parser = argparse.ArgumentParser()
	parser.add_argument('image_path', type = str, default = '/aipnd-project/flowers/valid/1/image_06739.jpg', help = 'Specify image path')
	parser.add_argument('modelpath', type = str, default = 'vgg16', help = 'Specify model path')
	parser.add_argument('--topk', type= int, default = 1)
	parser.add_argument('--category_names', type = str, default = 'cat_to_name.json')
	parser.add_argument('--gpu', action='store_true', default= False)
	
	args = parser.parse_args()
	top_k = args.top_k
	gpu = args.gpu
	
	device, criterion, optimizer, model = load_checkpoint(filepath)
	if args.category_names:
		with open(args.category_names, 'r') as f:
			model.class_to_idx = json.load(f)
	
	probabilities, flower_type = predict(image_path, model, topk, cat_to_name, gpu)
