from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import matplotlib.pyplot as plt
import time
import os
import copy

from dataset.thumbnail import ThumbnailDataset
from utils.utils import imshow
from model.train import train_model
from model.visualize import visualize_model


plt.ion()   # interactive mode


# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transform.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#         # transforms.RandomResizedCrop(224),
#         # transforms.RandomHorizontalFlip(),
#     'val': transforms.Compose([
#         transform.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#         # transforms.Resize(256),
#         # transforms.CenterCrop(224),
# }

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

data_dir = 'data/thumbnail'
seed = 0
batch_size = 4
split_ratio = [0.8, 0.1, 0.1]

thumbnail_dataset = ThumbnailDataset(data_dir, extensions=IMG_EXTENSIONS, transform=data_transform)

dataset_idx = thumbnail_dataset.split_idx(split_ratio, seed)
image_datasets = {key: torch.utils.data.Subset(thumbnail_dataset, dataset_idx[key])
                    for key in ['train', 'val', 'test']}
 
# = {x: ThumbnailDataset(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = thumbnail_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('thumbnail_dataset.samples[5]', thumbnail_dataset.samples[5])
print('thumbnail_dataset.targets[5]', thumbnail_dataset.targets[5])
print('thumbnail_dataset.tags[5]', thumbnail_dataset.tags[5])

print("image_datasets['train'][0]", image_datasets['train'][0])
print("image_datasets['val'][0]", image_datasets['val'][0])
print("image_datasets['test'][0]", image_datasets['test'][0])
print('dataset_sizes', dataset_sizes)
print('class_names', class_names)
print('device', device)

# Get a batch of training data
inputs, classes, tags = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
# visualize
imshow(out, title=str([class_names[x] for x in classes])+'\n'+str(tags))

model_ft = models.resnet50(pretrained=True, progress=True)
# fix model weights
for param in model_ft.parameters():
    param.requires_grad = False

# reset final fully connected layer
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                       dataset_sizes, device, num_epochs=25)

visualize_model(model_ft, dataloaders, class_names, device, num_images=10)