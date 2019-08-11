from utilities import *
import argparse

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

#import os
from PIL import Image
#from pathlib import Path

def train_network(device, criterion, optimizer, model,  epochs = 5, gpu = True, save_dir = '' ):
		
		
		#device, criterion, optimizer, model = build_network(architecture, learning_rate, hidden_units)
		
		if gpu == 'True':
				#  Stating the current device 
				if gpu and torch.cuda.is_available():
						print ('using gpu')
				elif gpu and not torch.cuda.is_available():
						print ('gpu is not avaliable,  using cpu')
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
				print ('using cpu')
				device = torch.device('cpu')
				
		model.to(device)
		
#     from workspace_utils import active_session
#     with active_session():
		trainloader, validloader, testloader = loading_data(data_dir = 'flowers')
		running_loss = 0
		steps =0
		print_every = 5

		for epoch in range(epochs):

				model.epochs +=1

				for inputs, labels in trainloader:
						steps+=1

						# moving tensors to the available device
						inputs, labels = inputs.to(device), labels.to(device)


						# reseting the optimizer for every turn
						optimizer.zero_grad()

						# receiving the result and educating the network
						logps = model.forward(inputs)
						loss = criterion(logps, labels)
						loss.backward()
						optimizer.step()

						running_loss += loss.item()

						if steps % print_every == 0:
								test_loss = 0
								accuracy = 0
								model.eval()
								with torch.no_grad():
										for inputs, labels in validloader:
												inputs, labels = inputs.to(device), labels.to(device)
												logps = model.forward(inputs)
												batch_loss = criterion(logps, labels)

												test_loss += batch_loss.item()

												# Calculate accuracy
												ps = torch.exp(logps)
												top_p, top_class = ps.topk(1, dim=1)
												equals = top_class == labels.view(*top_class.shape)
												accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

												print(f"Epoch {epoch+1}/{epochs}.. "
															f"Train loss: {running_loss/print_every:.3f}.. "
															f"Validation loss: {test_loss/len(testloader):.3f}.. "
															f"Validation accuracy: {accuracy/len(testloader):.3f}")
 
		save_checkpoint(save_dir)    
		print('Model has been saved successfully')
		return model

if __name__ == '__main__':
		# adapting argparse 
		parser = argparse.ArgumentParser()
		parser.add_argument('--architecture', default = 'vgg16', type = str, help = 'vgg16 or alexnet')
		parser.add_argument('--learning_rate', default= 0.001, type = float, help = 'Choose learning rate for your model')
		parser.add_argument('--hidden_units', default = 512, type = int, help = 'Choose learning rate for your model')
		parser.add_argument('--epochs', default = 5, type= int, help = 'Choose number of epochs to train your model')
		parser.add_argument('--gpu', default = True, type = bool, help = 'Do you like to use gpu if avaliable?')
		parser.add_argument('--save_dir', default = '', type = str, help = 'Specify directory to save your network')

		args = parser.parse_args()

		architecture = args.architecture
		learning_rate = args.learning_rate
		hidden_units = args.hidden_units
		epochs = args.epochs
		gpu = args.gpu
		save_dir = args.save_dir
#train_network(device, criterion, optimizer, model,  epochs = 5, gpu = True, save = False, save_dir = '' )
		device, criterion, optimizer, model = build_network(architecture, learning_rate, hidden_units)

		train_network(device, criterion, optimizer, model, epochs, gpu, save_dir)

