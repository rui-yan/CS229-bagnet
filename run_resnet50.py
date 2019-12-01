'''
CREDITS: Much of this code and some comments were adapted from the tutorial at: 
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
with modifications to fit our dataset and standardization to be comparable with BagNet-33
'''

from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import matplotlib.pyplot as plt

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Top level data directory
data_dir = './CS229-final-project/'

# Number of classes in the dataset
num_classes = 5

# Batch size for training (standardized to BagNet baseline)
batch_size = 256 

# Number of epochs to train for (doesn't matter too much... should technically stop running after no more improvement, could be different for ResNet and BagNet)
num_epochs = 100

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

##------------------------------------- Model -------------------------------------##
# Some useful functions for model training
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):

    """
    This function trains for a specified number of epochs, running validation step
    after each epoch. Keeps track of best performing model and returns it at end
    of training. Training and validation accuracies printed after each epoch.
    """
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history

# Sets model parameters so that we don't fine tune all parameters but only feature extract and
# compute gradients for newly initialized layer
def set_parameter_requires_grad(model, feature_extracting):
    """
    This function sets all parameters of model to False, which means we don't fine
    tune all parameters but only feature extract and compute gradients
    for newly initialized layer.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    This function initializes these variables which will be set in this
    if statement. Each of these variables is model specific.
    """
    model_ft = None

    if model_name == "bagnet":
        model_ft = bagnets.pytorchnet.bagnet33(pretrained=use_pretrained)
    if model_name == "resnet":
        model_ft = models.resnet50(pretrained=use_pretrained)

    set_parameter_requires_grad(model_ft, feature_extract)

    # Change the last layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[Using", device , "...]")

##---------------------------- Training and validation dataset------------------------##
print('==> [Preparing data ....]')

# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256),  # resize the image to 256*256 pixels
        transforms.CenterCrop(256),  # crop the image to 256*256 pixels about the center
        transforms.RandomHorizontalFlip(),  # convert the image to PyTorch Tensor data type
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")


# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder('./flowers', data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

###########  RESNET-50 ###########
print('==> Resnet-50 model')

##---- Load and modify model ----##
model_name = 'resnet'
model_ft_resnet = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Check output layer matches number of output categories of our dataset.
print(model_ft_resnet.fc)

##---- Create the optimizer ----##
# Create the optimizer to update only desired parameters (only output layer in our case).
# Send the model to CPU/GPU
model_ft_resnet = model_ft_resnet.to(device)

#  Gather the parameters to be optimized/updated in this run. If we are finetuning we will be
#  updating all parameters. However, if we are doing feature extract method, we will only update
#  the parameters that we have just initialized, i.e. the parameters with requires_grad is True.
params_to_update = model_ft_resnet.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft_resnet.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft_resnet.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft_resnet = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

##---- Train and evaluate -----#
# Setup the loss fxn
print("[Using CrossEntropyLoss ...]")
criterion = nn.CrossEntropyLoss()

print("[Training the model begun ...]")
model_ft_resnet, rhist = train_model(model_ft_resnet, dataloaders_dict, criterion,
    optimizer_ft_resnet, num_epochs=num_epochs)

#Save trained model parameters
#Adapted from: https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save({'model_resnet_state_dict': model_ft_resnet.state_dict(), 
            'optimizer_resnet_state_dict': optimizer_ft_resnet.state_dict(), 
            'rhist' : rhist
           }, './saved_resnet.pth')