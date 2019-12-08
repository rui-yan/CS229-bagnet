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

model_name = "resnet50"

# Number of classes in  the dataset
num_classes = 5

# Batch size for training (standardized to BagNet baseline)
batch_size = 32

# Number of epochs to train for (doesn't matter too much... should technically stop running after no more improvement, could be different for ResNet and BagNet)
#TODO: CHANGE TO SOMETHING LARGER
num_epochs = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

##------------------------------------- Model -------------------------------------##
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

    if model_name == "resnet50":
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
print("==> [Preparing data ....]")

# Data augmentation and normalization for training
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(256),  # resize the image to 256*256 pixels
        transforms.CenterCrop(256),  # crop the image to 256*256 pixels about the center
        transforms.RandomHorizontalFlip(),  # convert the image to PyTorch Tensor data type
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Just normalization for validation
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
train_data = torchvision.datasets.ImageFolder("./flowers/train/", data_transforms["train"])
val_data = torchvision.datasets.ImageFolder("./flowers/val/", data_transforms["val"])
test_data = torchvision.datasets.ImageFolder("./flowers/test/", data_transforms["test"])


# Create training and validation dataloaders
train_loader = torch.utils.data.DataLoader(
  train_data,
  batch_size=batch_size,
  shuffle=True,
  num_workers=4)

val_loader = torch.utils.data.DataLoader(
  val_data,
  batch_size=batch_size,
  shuffle=False,
  num_workers=4)

test_loader = torch.utils.data.DataLoader(
  test_data,
  batch_size=batch_size,
  shuffle=False,
  num_workers=4)


###########  RESNET-50 ###########
print('==> Resnet-50 model')

##---- Load and modify model ----##
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Check output layer matches number of output categories of our dataset.
print(model_ft.fc)

##---- Create the optimizer ----##
# Create the optimizer to update only desired parameters (only output layer in our case).
# Send the model to CPU
model_ft = model_ft.to(device)

#  Gather the parameters to be optimized/updated in this run. If we are finetuning we will be
#  updating all parameters. However, if we are doing feature extract method, we will only update
#  the parameters that we have just initialized, i.e. the parameters with requires_grad is True.
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

#---- Train and evaluate -----#
# Setup the loss fxn
print("[Using CrossEntropyLoss ...]")
criterion = nn.CrossEntropyLoss()

print("[Training the model begun ...]")
# model_ft, train_acc, train_loss, val_acc, val_loss = train_model(model_ft, dataloaders_dict, criterion,
#     optimizer_ft, num_epochs=num_epochs)

import torchbearer
from torchbearer import Trial

start_epoch = 0
TOTAL_EPOCH = num_epochs #40
INITIAL_LR = 0.001

train_loss = []
train_acc = []
val_loss = []
val_acc = []
test_acc = []
test_loss = []

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=INITIAL_LR)

tolerate = 0
best_val_acc = 0
patiance = 0

for i in range(start_epoch, TOTAL_EPOCH):
  print("Training starts: Epoch ", i)

  trial = Trial(model_ft, optimizer_ft, criterion, metrics=['loss', 'accuracy']).to(device)
  trial.with_generators(train_loader, val_generator=val_loader, test_generator=test_loader)
  trial.run(epochs=1)

  results_train = trial.evaluate(data_key=torchbearer.TRAIN_DATA)
  train_loss.append(results_train['train_loss'])
  train_acc.append(results_train['train_acc'])

  results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
  val_loss.append(results['val_loss'])
  val_acc.append(results['val_acc'])

  results_test = trial.evaluate(data_key=torchbearer.TEST_DATA)
  test_loss.append(results_test['test_loss'])
  test_acc.append(results_test['test_acc'])

  if results['val_acc'] >= best_val_acc:
    best_val_acc = results['val_acc']
    torch.save(model_ft.state_dict(), 'model_val%d.pt'%(best_val_acc*10000))
    print(" --------- New best validation acc --------- ", best_val_acc)
    patiance = 0

  else:
    tolerate += 1
    patiance += 1
    print("tolerate: ", tolerate, ", patiance: ", patiance)

  if tolerate >= 3:
    INITIAL_LR /= 2
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INITIAL_LR)
    tolerate = 0
    print("New LR: ", INITIAL_LR)

  if patiance > 15:
    break
    print("Early stopping")

##----- Save model ------##
print("==> Saving model...")
print('model_val%d.pt'%(best_val_acc*10000))
model_ft.load_state_dict(torch.load('model_val%d.pt'%(best_val_acc*10000)))

print("==> Saving loss and accuracy data...")
out=open('model_val%d.txt'%(best_val_acc*10000), 'w')
out.write(str(train_loss) + "\n")
out.write(str(val_loss) + "\n")
out.write(str(test_loss) + "\n")
out.write(str(train_acc) + "\n")
out.write(str(val_acc) + "\n")
out.write(str(test_acc) + "\n")
out.close()


##------- Evaluation ------##
results = trial.evaluate(data_key=torchbearer.VALIDATION_DATA)
print(results)

# Investigate the performance on the test data
predictions = trial.predict()
predicted_classes = predictions.argmax(1).cpu()
true_classes = [x for (_,x) in test_data.samples]
Index = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

from sklearn import metrics
CLASS = train_data.classes
print(metrics.classification_report(true_classes, predicted_classes, target_names=CLASS))


##------- Plotting ------##
# Plot loss and accuracy for bagnet33
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,4))
ax[0].plot(train_loss, label= "Train loss")
ax[0].plot(val_loss, label= "Val loss")
ax[0].plot(test_loss, label= "Test loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(train_acc, label= "Train acc")
ax[1].plot(val_acc, label= "Val acc")
ax[1].plot(test_acc, label= "Test acc")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()

plt.savefig("loss_acc_plot_bagnet33.png")
