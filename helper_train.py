import os
import torch
from torch import optim
from  torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
from collections import OrderedDict
import numpy as np

def create_path(data_dir):
    """

    :rtype: tuple(str)
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return (data_dir, train_dir, valid_dir, test_dir)
    
def create_data_loaders(train_dir, valid_dir, test_dir):
	# number of subprocesses to use for data loading
	num_workers = 0
	
	# number of samples per batch to load
	batch_size = 8
	
	train_transforms = transforms.Compose([transforms.Resize(255),
	                                       transforms.RandomCrop(224),
	                                       transforms.RandomHorizontalFlip(),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize([0.485, 0.456, 0.406],
	                                                            [0.229, 0.224, 0.225])
	                                       ])
	
	test_transforms = test_transforms = transforms.Compose([transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                                 [0.229, 0.224, 0.225])
	                                                        ])
	
	# Load the datasets with ImageFolder
	train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
	test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
	
	# Using the image datasets and the transforms, define the dataloaders
	train_loader = DataLoader(train_datasets, batch_size=batch_size, num_workers=num_workers)
	valid_loader = DataLoader(valid_datasets, batch_size=batch_size, num_workers=num_workers)
	test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

	# class to idx mapping
	class_to_idx = train_datasets.class_to_idx
	
	return(train_loader, valid_loader, test_loader, class_to_idx)

def create_cat_to_name_dict(filepath):
	with open(filepath, 'r') as f:
		cat_to_name = json.load(f)
	
	return cat_to_name

def create_model(model_arch, hidden_units):
	
	if model_arch == 'vgg':
		model = models.vgg11(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
			
		# Defining the feed forward Classifier
		classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
		                                        ('relu1', nn.ReLU()),
		                                        ('dropout1', nn.Dropout(p=0.25)),
		                                        ('fc2', nn.Linear(hidden_units, 102))
		                                        ])
		                           )
		model.classifier = classifier
		return  model
	
	else:
		model = models.resnet152(pretrained=True)
		for param in model.parameters():
			param.requires_grad = False
			
		# Defining the feed forward Classifier
		classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, hidden_units)),
		                                        ('relu1', nn.ReLU()),
		                                        ('dropout1', nn.Dropout(p=0.25)),
		                                        ('fc2', nn.Linear(hidden_units, 102))
		                                        ])
		                           )
		
		model.fc = classifier
		return model
	
	

def create_fc():
	# Defining the feed forward Classifier
	classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 512)),
	                                        ('relu1', nn.ReLU()),
	                                        ('dropout1', nn.Dropout(p=0.25)),
	                                        ('fc2', nn.Linear(512, 102))
	                                        ])
	                           )
	return classifier


def train(model, epochs, lr, train_loader, valid_loader, class_to_idx, cat_to_name, save_dir, on_gpu, model_arch):
	# Setting up training device to gpu or cpu
	device = 'cuda' if torch.cuda.is_available() and on_gpu else 'cpu'
	device = torch.device(device)
	
	# Defining the loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
	

	# converting idx back to class so as to map output back to classes
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	
	# track validation accuracy
	class_correct = {x: y for x, y in zip(cat_to_name.keys(), [0 for i in range(len(cat_to_name))])}
	class_total = {x: y for x, y in zip(cat_to_name.keys(), [0 for i in range(len(cat_to_name))])}
	
	overall_valid_loss, overall_train_loss = [], []
	
	# tracking changes in validation loss
	valid_loss_min = np.inf
	
	# Moving the model to available device
	model.to(device)
	
	for e in range(1, epochs + 1):
		
		# keep track of training and validation loss
		train_loss = 0.0
		valid_loss = 0.0
		
		###################
		# train the model #
		###################
		model.train()
		for data, target in train_loader:
			# move tensors to GPU if CUDA is available
			data, target = data.to(device), target.to(device)
			
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(data)
			
			# calculate the batch loss
			loss = criterion(output, target)
			
			# backward pass: compute gradient of the loss with respect to model parameters
			loss.backward()
			
			# perform a single optimization step (parameter update)
			optimizer.step()
			
			# update training loss
			train_loss += loss.item() * data.size(0)
		
		######################
		# validate the model #
		######################
		model.eval()
		for data, target in valid_loader:
			
			# move tensors to GPU if CUDA is available
			data, target = data.to(device), target.to(device)
			
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(data)
			
			# calculate the batch loss
			loss = criterion(output, target)
			
			# update validation loss
			valid_loss += loss.item() * data.size(0)
			
			## To calculate validation accuracy
			# convert output probabilities to predicted class
			_, pred = torch.max(output, 1)
			
			# compare predictions to true label
			correct_tensor = pred.eq(target.data.view_as(pred))
			correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(
				correct_tensor.cpu().numpy())
			
			# calculate validation accuracy for each object class
			for i in range(len(target.data)):
			    label = target.data[i]
			    class_correct[idx_to_class[label.item()]] += correct[i].item()
			    class_total[idx_to_class[label.item()]] += 1
		
		# calculate average losses
		train_loss = train_loss / len(train_loader.dataset)
		valid_loss = valid_loss / len(valid_loader.dataset)
		
		overall_train_loss.append(train_loss)
		overall_valid_loss.append(valid_loss)
		
		valid_accuracy = round(100 * (np.sum([x for x in class_correct.values()]) / np.sum([x for x in class_total.values()])), 2)
		
		print("Training loss: {}  Validation loss: {}    Validation accuracy: {}".format(train_loss,
		                                                                                 valid_loss,
		                                                                                 valid_accuracy)
		      )
		
		# save model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print(f'Epoch {e}/{epochs+1} Validation loss decreased {round(valid_loss_min, 6)} --> ' +
			      f'{round(valid_loss, 6)}.\nSaving model ...')
			model.cpu()
			
			try:
				torch.save(save_model(model, model_arch, class_to_idx), save_dir + '/'+'model.pt')
			except:
				torch.save(save_model(model, model_arch, class_to_idx), 'model.pt')
			
			model.to(device)
			valid_loss_min = valid_loss
		
		model.train()



def save_model(model, model_arch, class_to_idx):
	if model_arch == 'vgg':
		checkpoint = {'base_model': models.vgg11(),
		              'classifier': model.classifier,
		              'state_dict': model.state_dict(),
		              'class_to_idx': model.class_to_idx
		              }
		return checkpoint
	
	else:
		checkpoint = {'base_model': models.resnet152(),
		              'classifier': model.fc,
		              'state_dict': model.state_dict(),
		              'class_to_idx': class_to_idx
		              }
		return checkpoint
