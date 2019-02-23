import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

# path = "C:/Users/eva4good/Google Drive/dlnd 2/flowers/mpagi.pt"
# torch.save(models.vgg11(), path)

def file_exist(file_path):
	"""Return if a path exists or not
	
	Args:
		file_path (str): A valid directory

	Returns:
		bool: Returns True if path exists or False if not.

	"""
	return os.path.isfile(file_path)


def load_checkpoint(checkpoint_path):
	"""Return a model from a checkpoint
	
	Args:
		checkpoint_path (str): A path to a valid PyTorch checkpoint file.

	Returns:
		tuple: Return a tuple of the following
			`checkpoint_path` - A path to a valid PyTorch checkpoint file
			`model` - A valid `torch.Module` object
			`model._get_name()` - Name of the model
			`class_to_idx` - Class to id mapping of the train data

	"""
	
	checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	model = checkpoint['base_model']
	
	try:
		model.fc = checkpoint['classifier']
	except:
		model.classifier = checkpoint['classifier']
		
	model.load_state_dict(checkpoint['state_dict'], strict=False)
	class_to_idx = checkpoint['class_to_idx']
	
	return checkpoint_path, model, model._get_name(), class_to_idx


def process_image(image_path, asNumpy=False):
	"""Return a processed Tensor or Numpy Image
	
	Args:
		image_path (str): Path to the image to process
		asNumpy (`str`, optional): Specifies if the image should be returned as a Numpy image or a Tensor image.
			Default image is Tensor

	Returns:

	"""
	process_transform = transforms.Compose([transforms.Resize(256),
	                                        transforms.CenterCrop(224),
	                                        transforms.ToTensor(),
	                                        transforms.Normalize([0.485, 0.456, 0.406],
	                                                             [0.229, 0.224, 0.225])
	                                        ])
	
	''' Scales, crops, and normalizes a PIL image for a PyTorch model,
	        returns a PyTorch Tensor Image or a Numpy Image
	    '''
	# Reference: https://pillow.readthedocs.io/en/latest/reference/Image.html
	
	im = Image.open(image_path)
	im = im.convert("RGB")
	im = process_transform(im)
	return im  if not asNumpy else im.numpy()


def predict(image_path, model, top_k, onGPU):
	"""Return predicted classes and probabilities
	
	Args:
		image_path (str): Path to image to predict
		model (obj): A valid `torch.nn.Module` object
		top_k (int): Maximum number of classes and probabilities to return
		onGPU (bool): Specifies if prediction should be done on GPU. Default is True

	Returns:
		obj: Returns a `torch.tensor` object containing `topk` number of classes and probabilities.

	"""
	# Setting up training device to gpu or cpu
	device = 'cuda' if torch.cuda.is_available() and onGPU else 'cpu'
	device = torch.device(device)
	
	model.to(device)
	model.eval()
	image = process_image(image_path).unsqueeze_(0)
	output = model(image.to(device)) / 100
	output = output.float().cpu()
	return torch.topk(output, top_k)
	
	
def get_names_probs(predict_output, cat_to_name, class_to_idx):
	"""Return the names and probabilities of the predicted classes
	
	Args:
		predict_output (obj): A `torch.tensor` object from typically with 2 tensors
		cat_to_name (dict): A dictionary containing key to category  mapping of each category in the cat_to_name file
		class_to_idx (dict): Class to id mapping of the train data

	Returns:
		tuple: Returns a tuple of
			`flower_names` - Name of the flowers predicted
			`flower_probs` - Probabilities associated with each predicted flower

	"""
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	flower_names = [cat_to_name[idx_to_class[x.item()]] for x in predict_output[1][0]]
	flower_probs = [x.item() for x in predict_output[0][0]]
	return flower_names, flower_probs

