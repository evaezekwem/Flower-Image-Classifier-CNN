
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[62]:


# Imports here
import os
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


# Check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available! Training on GPU...')
    

# Setup the training device
device = 'cuda' if train_on_gpu else 'cpu'
device = torch.device(device)

device


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). You can [download the data here](https://s3.amazonaws.com/content.udacity-data.com/courses/nd188/flower_data.zip). The dataset is split into two parts, training and validation. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. If you use a pre-trained network, you'll also need to make sure the input data is resized to 224x224 pixels as required by the networks.
# 
# The validation set is used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks available from `torchvision` were trained on the ImageNet dataset where each color channel was normalized separately. For both sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.

# In[63]:


#raw_data_path = os.path.join(os.path.pardir, data)

data_dir = 'flower_data'
train_dir = os.path.join(data_dir, data_dir, 'train')
valid_dir = os.path.join(data_dir, data_dir, 'valid')


# In[64]:


# TODO: Define your transforms for the training and validation sets

#number of subprocesses to use for data loading
num_workers = 0

# number of samples per batch to load
batch_size = 8

# percentage of training set to use as validation
valid_size = 0.2

# Define data transformations
train_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Scale(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
test_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)


# obtain training indicies that will be used for validation
num_train = len(train_datasets)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)



# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,
                                                sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,
                                                sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[65]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
len(cat_to_name)

len(test_loader)

# next(iter(test_loader))


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[66]:


# TODO: Build and train your network
model_rsn152_2 = models.resnet152(pretrained=True)
model_rsn152_2


# In[67]:


# Freezing pretrained model parameters 
# for param in model_rsn152_2.parameters():
#     param.requires_grad = False
    

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 512)),
                                              ('relu1', nn.ReLU()),
                                              ('dropout1', nn.Dropout(p=0.25)),
                                              ('fc2', nn.Linear(512, 102))
                                             ]))

# class Classifier(nn.Module):
#     def __init__:
#         super().__init__
        
        


model_rsn152_2.fc = classifier

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model_rsn152_2.parameters(), lr=0.00001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_rsn152_2.to(device)


# In[ ]:


state_dict = torch.load('model_rsn152_2.pt')
model_rsn152_2 = models.resnet152()
model_rsn152_2.fc = classifier
model_rsn152_2.load_state_dict(state_dict, strict = False)
model_rsn152_2.to(device)
# # torch.save(model_rsn152_2.state_dict(), 'model_rsn152_2.pt')


# In[61]:


get_ipython().run_cell_magic('time', '', '\n# track validation accuracy\nclass_correct = list(0. for i in range(len(cat_to_name)))\nclass_total = list(0. for i in range(len(cat_to_name)))\n\n\n# number of epochs to train the model\nepochs = 100\n\noverall_valid_loss, overall_train_loss = [], []\n\nvalid_loss_min = 0.00212 # tracking changes in validation loss\n\nfor e in range(1, epochs+1):\n    \n    # keep track of training and validation loss\n    train_loss = 0.0\n    valid_loss = 0.0\n    \n    ###################\n    # train the model #\n    ###################\n    model_rsn152_2.train()\n    for data, target in train_loader:\n        # move tensors to GPU if CUDA is available\n        data, target = data.to(device), target.to(device)\n#         print(f"Data size {len(data)}")\n#         print(f"Data size {target.data[0]}")\n#         break\n#     break\n        #clear the gradients of all optimized variables\n        optimizer.zero_grad()\n        #forward pass: compute predicted outputs by passing inputs to the model\n        output = model_rsn152_2(data)\n        # calculate the batch loss\n        loss = criterion(output, target)\n        # backward pass: compute gradient of the loss with respect to model parameters\n        loss.backward()\n        # perform a single optimization step (parameter update)\n        optimizer.step()\n        # update training loss\n        train_loss += loss.item() * data.size(0)\n        #print(f\'{e} trainings completed\')\n        \n    ######################\n    # validate the model #\n    ######################\n    model_rsn152_2.eval()\n    for data, target in valid_loader:\n        # move tensors to GPU if CUDA is available\n        data, target = data.to(device), target.to(device)\n        # forward pass: compute predicted outputs by passing inputs to the model\n        output = model_rsn152_2(data)\n        # calculate the batch loss\n        loss = criterion(output, target)\n        # update average validation loss\n        valid_loss += loss.item() * data.size(0)\n        #print(f\'{e} validation loss calculation  completed\')\n\n        \n        ## To calculate validation accuracy\n        # convert output probabilities to predicted class\n        _, pred = torch.max(output, 1)\n        # compare predictions to true label\n        correct_tensor = pred.eq(target.data.view_as(pred))\n        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())\n        # calculate validation accuracy for each object class\n#         for i in range(batch_size):\n#             label = target.data[i]\n#             class_correct[label] += correct[i].item()\n#             class_total[label] += 1\n    \n    \n    \n    # calculate average losses\n    train_loss = train_loss/len(train_loader.dataset)\n    valid_loss = valid_loss/len(valid_loader.dataset)\n\n    overall_train_loss.append(train_loss)\n    overall_valid_loss.append(valid_loss)\n    \n    # save model if validation loss has decreased\n    if valid_loss <= valid_loss_min:\n        print(f\'Epoch {e}/{epochs+1} Validation loss decreased {round(valid_loss_min, 6)} --> \'+\n              f\'{round(valid_loss, 6)}.\\nSaving model ...\')\n        model_rsn152_2.cpu()\n        torch.save(model_rsn152_2.state_dict(), \'model_rsn152_2.pt\')\n        model_rsn152_2.to(device)\n        valid_loss_min = valid_loss\n        \n    if e % 3 == 0:\n#         valid_accuracy = 100 * np.sum(class_correct) / np.sum(class_total)\n        # Plot the error and accuracy graph\n#         plt.figure(1)\n        \n        \n        # error plot\n#         plt.subplot(221)\n        plt.plot([x for x in range(len(overall_train_loss))], overall_train_loss, label=\'train\', color=\'r\')\n        plt.plot([x for x in range(len(overall_train_loss))], overall_valid_loss, label=\'validation\', color=\'g\')\n        plt.xlabel(\'Epoch\')\n        plt.ylabel(\'Loss\')\n        plt.title(\'Train vs Validation Loss\')\n        plt.legend()\n        \n        # Accuracy plot\n#         plt.subplot(222)\n#         plt.plot(np.sum(class_total), valid_accuracy)\n#         plt.xlabel(\'No of observations\')\n#         plt.ylabel(\'Accuracy\')\n#         plt.title(\'Validation Accuracy\')\n        \n        plt.show()\n    model_rsn152_2.train()')


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[24]:


classifier


# In[34]:


# TODO: Save the checkpoint 
model_rsn152_2.class_to_idx = train_datasets.class_to_idx
torch.save(optimizer.state_dict(), 'optimizer.pt')

checkpoint = {'base_model': models.resnet152(),
              'classifier': classifier,
              'minimum_loss': valid_loss_min,
              'optim_state_dict': optimizer.state_dict(),
              'state_dict': model_rsn152_2.state_dict()
             }
torch.save(checkpoint, 'custom_rsn152_2_trained.pt')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[35]:


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['base_model']
    model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    return model

model2 = load_checkpoint('custom_rsn152_2_trained.pt')

model2.to(device)

# model2 = model_rsn152_2
# model2 = models.densenet201()
# model2.classifier = classifier
# model2.load_state_dict('model_dsn201_1.pt')


# ### Testing The Model

# In[36]:




correct = 0
total = 0

class_correct = {x:y for x,y in zip(cat_to_name.keys(), [0 for i in range(len(cat_to_name))])}
class_total   = {x:y for x,y in zip(cat_to_name.keys(), [0 for i in range(len(cat_to_name))])}

model2.eval()
with torch.no_grad():
    for images, labels in test_loader:
#         break
        images, labels = images.to(device), labels.to(device)
        
        outputs = model2(images)
        
        _, predictions = torch.max(outputs, 1)
        
        # compare predictions to true label
        correct_tensors = predictions.eq(labels.view_as(predictions))
        correct = np.squeeze(correct_tensors.numpy()) if not train_on_gpu else np.squeeze(correct_tensors.cpu().numpy())
        
        # calculate test accuracy for each object class
        for i in range((len(labels.data))):

            label = labels.data[i]
            
            try:
                class_correct[str(label.item())] += correct[i].item()
                class_total[str(label.item())] += 1
                
            except KeyError:
                pass
            
            except IndexError:
                pass
            
#            

        
for i in cat_to_name.keys():
    if class_total[i] > 0:
        print(f'Test Accuracy of {cat_to_name[i]}: {round(100 * class_correct[i] / class_total[i], 2)}% '+
              f'{np.sum(class_correct[i])}/{np.sum(class_total[i])}')
    else:
        print(f'Accuracy of {cat_to_name[i]}: N/A (no training examples)')


print('\nTest Accuracy (Overall): {}%'
      .format(round(100 * np.sum([x for x in class_correct.values()]) / np.sum([x for x in class_total.values()]), 2)))




# In[ ]:


train_on_gpu


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[ ]:



test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img.resize(224, 224)
    transforms.Normalize()
    transforms.ToTensor(img.resize(224, 224))
    
image
transforms.ToTensor(img.resize(224, 224))


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[ ]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[ ]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the validation accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:


# TODO: Display an image along with the top 5 classes

