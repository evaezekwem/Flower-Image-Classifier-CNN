import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

path = "C:/Users/eva4good/Google Drive/dlnd 2/flowers/mpagi.pt"
torch.save(models.vgg11(), path)

def file_exist(file_path):
	return os.path.isfile(file_path)


def load_checkpoint(checkpoint_path):
	
	class_to_idx = None
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
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	flower_names = [cat_to_name[idx_to_class[x.item()]] for x in predict_output[1][0]]
	flower_probs = [x.item() for x in predict_output[0][0]]
	return flower_names, flower_probs

