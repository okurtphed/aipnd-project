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

def loading_data(data_dir = 'flowers'):
		train_dir = data_dir + '/train'
		valid_dir = data_dir + '/valid'
		test_dir = data_dir + '/test'
		
		# Defining transforms for the training, validation, and testing sets
		train_transforms = transforms.Compose([transforms.RandomRotation(30),
																			transforms.RandomResizedCrop(224),
																			transforms.RandomHorizontalFlip(),
																			transforms.ToTensor(),
																			transforms.Normalize([0.485, 0.456, 0.406],
																													 [0.229, 0.224, 0.225])])
		valid_transforms = transforms.Compose([transforms.Resize(255),
																			transforms.CenterCrop(224),
																			transforms.ToTensor(),
																			transforms.Normalize([0.485, 0.456, 0.406],
																													 [0.229, 0.224, 0.225])])
		test_transforms = transforms.Compose([transforms.Resize(255),
																			transforms.CenterCrop(224),
																			transforms.ToTensor(),
																			transforms.Normalize([0.485, 0.456, 0.406],
																													 [0.229, 0.224, 0.225])])
		# Loading the datasets with ImageFolder
		train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
		valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
		test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
		
		# Defining the dataloaders
		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
		validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
		testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
		
		return trainloader, validloader, testloader


def save_checkpoint(save_dir =''):
		
		model.class_to_idx = train_dataset.class_to_idx
		
		checkpoint = {'architecture': model.arch,
									'learning_rate':model.learning_rate,
									'epochs': model.epochs,
									'classifier': model.classifier,
									'optimizer' : model.optimizer,
									'criterion': model.criterion,
									'state_dict': model.state_dict(),
									'class_to_idx': model.class_to_idx}
		s_dir = os.path.join(save_dir,model.arch + '_checkpoint.pth')
		# saving as 'folder\modelname_checkpoint.pth'
		return torch.save(checkpoint, s_dir)

def load_checkpoint(modelpath):
		checkpoint = torch.load(modelpath)
		for k,v in checkpoint.items():
				v = checkpoint[k]
		
		model.load_state_dict(checkpoint['state_dict'])
		model.class_to_idx = checkpoint['class_to_idx']
		
		
		return device, criterion, optimizer, model.eval()

def process_image(image_path):
		''' Scales, crops, and normalizes a PIL image for a PyTorch model,
				returns an Numpy array
		'''
		
		from PIL import Image
		
		# opening image
		image = Image.open(image_path)
		
		#checking rgb
		if not image.mode == 'RGB':
				image = image.convert('RGB')
		
		# apply the same transformations with resize 256
		image_transforms = transforms.Compose([transforms.Resize(256),
																			transforms.CenterCrop(224),
																			transforms.ToTensor(),
																			transforms.Normalize([0.485, 0.456, 0.406],
																													 [0.229, 0.224, 0.225])])
		image = image_transforms(image)
		
		return image.numpy() 

def imshow_with_title(image_path):
		fig, ax = plt.subplots()
		image = process_image(image_path)

		flower_type = image_path.split('/')[-2]
		flower_name = cat_to_name[flower_type]
		
		
		if ax is None:
				fig, ax = plt.subplots()
				
		# PyTorch tensors assume the color channel is the first dimension
		# but matplotlib assumes is the third dimension
		image = image.transpose((1, 2, 0))
		
		# Undo preprocessing
		mean = np.array([0.485, 0.456, 0.406])
		std = np.array([0.229, 0.224, 0.225])
		image = std * image + mean
		
		# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
		image = np.clip(image, 0, 1)

		ax.imshow(image)
		ax.set_title(flower_name)
		
		return ax


def build_network(architecture = 'vgg16', learning_rate = 0.002, hidden_units = 1024 ):
		# #choosing vgg16 or alexnet
		if architecture == 'vgg16':
				model = models.vgg16(pretrained = True)
				#replacing classifier
				classifier = nn.Sequential(OrderedDict([
													('fc1', nn.Linear(25088, hidden_units)),
													('relu', nn.ReLU()),
													('dropout', nn.Dropout(0.2)), 
													('fc2', nn.Linear(hidden_units, 102)),
													('output', nn.LogSoftmax(dim=1))
													]))
		elif architecture == 'alexnet':
				model = models.alexnet(pretrained=True)
				#replacing classifier, Classifier with more layer could provide more accuracy but takes too long to train
				classifier = nn.Sequential(OrderedDict([
									('fc1', nn.Linear(9216, hidden_units)),
									('relu', nn.ReLU()),
									('dropout', nn.Dropout(0.2)), 
									('fc2', nn.Linear(hidden_units, 102)),
									('output', nn.LogSoftmax(dim=1))
									]))
		else:
				print('Please choose either vgg16 or alexnet')

		# Using gpu if avaliable
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# Freezing parameters
		for param in model.parameters():
				param.requires_grad = False

		
		
		# initilazing model variables to save and load later
		model.arch = architecture
		model.learning_rate = learning_rate
		model.epochs = 0
		model.classifier = classifier
		
		# defining optimizer with learning rate input
		optimizer = optim.Adam(model.classifier.parameters(), lr=model.learning_rate)
		model.optimizer = optimizer

		#defining criterion
		criterion = nn.NLLLoss()
		
		#saving criterion
		model.criterion = criterion
		
		return device, criterion, optimizer, model.to(device)

def display_results(image_path, model):
		probs, classes, flower_type = predict(image_path, model)
		
		#return plt.barh(flower_type, probs), imshow_with_title(image_path)
		fig, ax = plt.subplots(figsize=(10,4))
		plt.ylabel('Probabiliy %')
		#plt.xlabel('Flower Type')
		# why orient='h' not working
		return sns.barplot(flower_type, probs, color=sns.color_palette()[0]),imshow_with_title(image_path)
