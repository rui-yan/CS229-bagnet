'''
THIS FILE CONTAINS CODE TO RUN PATCH BLACKOUT EXPERIMENTS ON BAGNET-33.

CREDITS: Much of the beginning of this code was adapted code from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2
import copy
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import tqdm
import random
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
try:
    import torchbearer
except:
    !pip install torchbearer
    import torchbearer
from torchbearer import Trial
import scipy
import scipy.special
import bagnets.pytorchnet
print("[libraries successfully installed...]")


#--------------------Some Helper Functions---------------------------

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

    if model_name == "bagnet9":
        model_ft = bagnets.pytorchnet.bagnet9(pretrained=use_pretrained)
    if model_name == "bagnet17":
        model_ft = bagnets.pytorchnet.bagnet17(pretrained=use_pretrained)
    if model_name == "bagnet33":
        model_ft = bagnets.pytorchnet.bagnet33(pretrained=use_pretrained)

    set_parameter_requires_grad(model_ft, feature_extract)

    # Change the last layer
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft
print("[Helper functions loaded...]")


#--------------------- Load test datasets ------------------------------
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),  # resize the image to 224*224 pixels
        transforms.CenterCrop(224),  # crop the image to 224*224 pixels about the center
        transforms.RandomHorizontalFlip(),  # convert the image to PyTorch Tensor data type
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation
    "val": transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("[Initializing test datasets and dataloaders...]")

# Create test datasets
image_datasets = {x: datasets.ImageFolder("./data/test", data_transforms[x])
                  for x in ["train", "test", "val"]}

# Create test dataloaders
batch_size = 32
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
                    for x in ["train", "test", "val"]}
train_loader = dataloaders_dict["train"]
val_loader = dataloaders_dict["val"]
test_loader = dataloaders_dict["test"]

print("[Datasets loaded...]")

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[Using", device , "...]")


#------------------- Initialize Bagnet-33 model --------------------

print('==> Bagnet-33 model')
model_name = "bagnet33"
feature_extract = True
num_classes = 5
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Send the model to CPU
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
print("[Using CrossEntropyLoss ...]")
criterion = nn.CrossEntropyLoss()
print("[Bagnet33 model Initialized...]")


#--------------- Load saved weights --------------------------------

checkpoint = torch.load("./bagnet33_baseline_model.pth")
model_ft.load_state_dict(checkpoint['model_bagnet33_state_dict'])
optimizer_ft.load_state_dict(checkpoint['optimizer_bagnet33_state_dict'])
print("--------Saved Bagnet33 weights loaded--------------------")

model_ft.fc.weight


#---------------------- Experiments --------------------------------

##----- EXPERIMENT 1: Drop out every other patch -----##

exp1_model = copy.deepcopy(model_ft) #make a copy of model so we don't mess up original weights

#Changing aggregation layer weights
num_patches = exp1_model.fc.weight.shape[1]

ones = np.ones(num_patches) #Create selective dropout Tensor with alternating 1-0 pattern
for i in range(ones.shape[0]):
    if i % 2 == 1:
        ones[i] = ones[i] % 1
ones_tensor = torch.from_numpy(ones)
ones_tensor = ones_tensor.float() #cast to float32 to match model
ones_tensor = ones_tensor.to(device)

exp1_model.fc.weight = nn.Parameter(exp1_model.fc.weight * ones_tensor) #for every class, every other patch is deleted

print("--------Investigate performance on test datasets: Alternating patch experiment---------")
trial = Trial(exp1_model, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


##----- EXPERIMENT 2: Drop out 25% of patches randomly -----##

#Create vector with 1/4 0's and 3/4 1's
fours = np.ones(num_patches) #Create selective dropout Tensor with alternating 1-0 pattern
for i in range(fours.shape[0]):
    if i < (num_patches / 4):
        fours[i] = 0


#Random dropout 1
exp3_modelv1 = copy.deepcopy(model_ft)

random_patch_ind_1 = np.random.permutation(fours)

random_patch_ind_1_tensor = torch.from_numpy(random_patch_ind_1)
random_patch_ind_1_tensor = random_patch_ind_1_tensor.float() #cast to float32 to match model
random_patch_ind_1_tensor = random_patch_ind_1_tensor.to(device)

exp3_modelv1.fc.weight = nn.Parameter(exp3_modelv1.fc.weight * random_patch_ind_1_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 25% patches experiment v1---------")
trial = Trial(exp3_modelv1, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 2
exp3_modelv2 = copy.deepcopy(model_ft)

random_patch_ind_2 = np.random.permutation(fours)

random_patch_ind_2_tensor = torch.from_numpy(random_patch_ind_2)
random_patch_ind_2_tensor = random_patch_ind_2_tensor.float() #cast to float32 to match model
random_patch_ind_2_tensor = random_patch_ind_2_tensor.to(device)

exp3_modelv2.fc.weight = nn.Parameter(exp3_modelv2.fc.weight * random_patch_ind_2_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 25% patches experiment v2---------")
trial = Trial(exp3_modelv2, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 3
exp3_modelv3 = copy.deepcopy(model_ft)

random_patch_ind_3 = np.random.permutation(fours)

random_patch_ind_3_tensor = torch.from_numpy(random_patch_ind_3)
random_patch_ind_3_tensor = random_patch_ind_3_tensor.float() #cast to float32 to match model
random_patch_ind_3_tensor = random_patch_ind_3_tensor.to(device)

exp3_modelv3.fc.weight = nn.Parameter(exp3_modelv3.fc.weight * random_patch_ind_3_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 25% patches experiment v3---------")
trial = Trial(exp3_modelv3, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


##----- EXPERIMENT 3: Drop out 50% of patches randomly -----##

#Random dropout 1
exp2_modelv1 = copy.deepcopy(model_ft)

random_patch_ind1 = np.random.permutation(ones)
random_patch_ind1

random_patch_ind1_tensor = torch.from_numpy(random_patch_ind1)
random_patch_ind1_tensor = random_patch_ind1_tensor.float() #cast to float32 to match model
random_patch_ind1_tensor = random_patch_ind1_tensor.to(device)

exp2_modelv1.fc.weight = nn.Parameter(exp2_modelv1.fc.weight * random_patch_ind1_tensor)

print("--------Investigate performance on test datasets: Random dropout 50% patches experiment v1---------")
trial = Trial(exp2_modelv1, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 2
exp2_modelv2 = copy.deepcopy(model_ft)

random_patch_ind2 = np.random.permutation(ones)
random_patch_ind2

random_patch_ind2_tensor = torch.from_numpy(random_patch_ind2)
random_patch_ind2_tensor = random_patch_ind2_tensor.float() #cast to float32 to match model
random_patch_ind2_tensor = random_patch_ind2_tensor.to(device)

exp2_modelv2.fc.weight = nn.Parameter(exp2_modelv2.fc.weight * random_patch_ind2_tensor) #for every class, every other patch is deleted

print("--------Investigate performance on test datasets: Random dropout 50% patches experiment v2---------")
trial = Trial(exp2_modelv2, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 3
exp2_modelv3 = copy.deepcopy(model_ft)

random_patch_ind3 = np.random.permutation(ones)
random_patch_ind3

random_patch_ind3_tensor = torch.from_numpy(random_patch_ind3)
random_patch_ind3_tensor = random_patch_ind3_tensor.float() #cast to float32 to match model
random_patch_ind3_tensor = random_patch_ind3_tensor.to(device)

exp2_modelv3.fc.weight = nn.Parameter(exp2_modelv3.fc.weight * random_patch_ind3_tensor) #for every class, every other patch is deleted

print("--------Investigate performance on test datasets: Random dropout 50% patches experiment v3---------")
trial = Trial(exp2_modelv3, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


##----- EXPERIMENT 4: Drop out 75% of patches randomly -----##

#Create vector with 3/4 0's and 1/4 1's
twentyfiveperc = np.ones(num_patches) #Create selective dropout Tensor with alternating 1-0 pattern
for i in range(twentyfiveperc.shape[0]):
    if i < (num_patches * 3 / 4):
        twentyfiveperc[i] = 0


#Random dropout 1
exp4_modelv1 = copy.deepcopy(model_ft)

exp4_random_patch_ind1 = np.random.permutation(twentyfiveperc)

exp4_random_patch_ind1_tensor = torch.from_numpy(exp4_random_patch_ind1)
exp4_random_patch_ind1_tensor = exp4_random_patch_ind1_tensor.float() #cast to float32 to match model
exp4_random_patch_ind1_tensor = exp4_random_patch_ind1_tensor.to(device)

exp4_modelv1.fc.weight = nn.Parameter(exp4_modelv1.fc.weight * exp4_random_patch_ind1_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 75% patches experiment v1---------")
trial = Trial(exp4_modelv1, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 2
exp4_modelv2 = copy.deepcopy(model_ft)

exp4_random_patch_ind2 = np.random.permutation(twentyfiveperc)

exp4_random_patch_ind2_tensor = torch.from_numpy(exp4_random_patch_ind2)
exp4_random_patch_ind2_tensor = exp4_random_patch_ind2_tensor.float() #cast to float32 to match model
exp4_random_patch_ind2_tensor = exp4_random_patch_ind2_tensor.to(device)

exp4_modelv2.fc.weight = nn.Parameter(exp4_modelv2.fc.weight * exp4_random_patch_ind2_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 75% patches experiment v2---------")
trial = Trial(exp4_modelv2, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()


#Random dropout 3

exp4_modelv3 = copy.deepcopy(model_ft)

exp4_random_patch_ind3 = np.random.permutation(twentyfiveperc)

exp4_random_patch_ind3_tensor = torch.from_numpy(exp4_random_patch_ind3)
exp4_random_patch_ind3_tensor = exp4_random_patch_ind3_tensor.float() #cast to float32 to match model
exp4_random_patch_ind3_tensor = exp4_random_patch_ind3_tensor.to(device)

exp4_modelv3.fc.weight = nn.Parameter(exp4_modelv3.fc.weight * exp4_random_patch_ind3_tensor) #every class has random patches deleted in the same way

print("--------Investigate performance on test datasets: Random dropout 75% patches experiment v2---------")
trial = Trial(exp4_modelv3, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()
