from __future__ import print_function, division
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
# Some useful functions for model training
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    This function trains for a specified number of epochs, running validation step
    after each epoch. Keeps track of best performing model and returns it at end
    of training. Training and validation accuracies printed after each epoch.
    """
    since = time.time()

    train_acc_history =[]
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

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
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history


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
    # Just normalization for validation
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
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
dataloaders_dict = {"train": torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                    shuffle=True, num_workers=2),
                    "val": torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                    shuffle=False, num_workers=2),
                    "test": torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                    shuffle=False, num_workers=2)}


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

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#---- Train model -----#
# Setup the loss fxn
print("[Using CrossEntropyLoss ...]")
criterion = nn.CrossEntropyLoss()

print("[Training the model begun ...]")
model_ft, train_acc, train_loss, val_acc, val_loss = train_model(model_ft, dataloaders_dict, criterion,
    optimizer_ft, num_epochs=num_epochs)

##----- Save model ------##
print("==> Saving model...")
torch.save({'model_resnet50_state_dict': model_ft.state_dict(),
            'optimizer_resnet50_state_dict': optimizer_ft.state_dict()
            }, './resnet50/resnet50_baseline_model.pth')

print("==> Saving loss and accuracy data...")
out=open('./resnet50/resnet50_loss_acc_data.txt', 'w')
out.write(str(train_acc) + "\n")
out.write(str(train_loss) + "\n")
out.write(str(val_acc) + "\n")
out.write(str(val_loss) + "\n")
out.close()


##------- Plot acc_loss ------##
# Plot loss and accuracy for resnet50
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(9,4))
ax[0].plot(train_loss, label= "Train loss")
ax[0].plot(val_loss, label= "Val loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(train_acc, label= "Train acc")
ax[1].plot(val_acc, label= "Val acc")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()

plt.savefig("./resnet50/resnet50_loss_acc_plot.png")


##------- Test model ------##
print("==> Testing model...")
checkpoint = torch.load('./resnet50/resnet50_baseline_model.pth')
model_ft.load_state_dict(checkpoint['model_resnet50_state_dict'])

def test(model, dataloaders, criterion):
    # monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    for batch_idx, (data, target) in enumerate(dataloaders['test']):
        # move to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # calculate the loss
        loss = criterion(output, target)

        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        print('\nTest Loss: {:.6f}\n'.format(test_loss))
        print('Test Accuracy: %2d%% (%2d/%2d)\n' % (100. * correct / total, correct, total))

test(model_ft, dataloaders_dict, criterion)
