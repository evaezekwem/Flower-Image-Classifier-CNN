# Imports here
import os
import argparse
import helper_train as h
# import matplotlib.pyplot as plt
# import json
# from PIL import Image

import numpy as np
import torch
from torch import nn

from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# main(args.data_dir, args.save_dir, args.model_arch, args.lr, args.hidden_units, args.epoch, args.on_gpu)
def main(data_dir, save_dir, model_arch, lr, hidden_units, epoch, on_gpu):
    # print("Data directory: {} \nSave directory: {}".format(data_dir, save_dir))
    
    
    all_dir = h.create_path(data_dir)
    train_dir = all_dir[1]
    valid_dir = all_dir[2]
    test_dir = all_dir[3]
    
    all_loaders = h.create_data_loaders(train_dir, valid_dir, test_dir)
    train_loader = all_loaders[0]
    valid_loader = all_loaders[1]
    test_loader = all_loaders[2]
    class_to_idx = all_loaders[3]
    
    # Label Mapping
    cat_to_name = h.create_cat_to_name_dict('cat_to_name.json')
    
    # Creating model
    model = h.create_model(model_arch, hidden_units)
    
    h.train(model, epoch, lr, train_loader, valid_loader, class_to_idx, cat_to_name, save_dir, on_gpu, model_arch)
    
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A \
             simple app to train a neural net')

    parser.add_argument('data_dir', action='store',
                        help="Specifies the folder containing the training and validation data")
    parser.add_argument('--save_dir', action='store', dest='save_dir',
                        help="Specifies the folder to save the trained model")
    parser.add_argument('--arch', default='resnet', choices=['vgg', 'resnet'], dest='model_arch',
                        help="Specifies the architecture of the model. 'resnet' or 'vgg'")
    parser.add_argument('--learning_rate', action='store', type=float, default=0.01, dest='lr',
                        help="Specifies the learning rate of the model")
    parser.add_argument('--hidden_units', action='store', type=int, default=512, dest='hidden_units',
                        help="Specifies the number of hidden units in the network")
    parser.add_argument('--epochs',action='store', type=int, default=3, dest='epochs',
                        help="Specifies the number of epochs for the training")
    parser.add_argument('--gpu', action='store_false', default=True, dest='on_gpu',
                        help="Specifies if the model training will be done on GPU or CPU")

    args = parser.parse_args()
    
    print("#################....Parameters....##################")
    print("save_dir.........................{}".format(args.save_dir))
    print("arch.............................{}".format(args.model_arch))
    print("learning_rate....................{}".format(args.lr))
    print("hidden_units.....................{}".format(args.hidden_units))
    print("epochs...........................{}".format(args.epochs))
    print("gpu..............................{}".format(args.on_gpu))
    print("#################....Parameters....##################")
    main(args.data_dir, args.save_dir, args.model_arch, args.lr, args.hidden_units, args.epochs, args.on_gpu)
