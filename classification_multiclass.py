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
import argparse
import wandb
import sklearn.model_selection
import PIL

from utils.utils import imshow
from model.train import train_model
from model.visualize import visualize_model
from model.resnet50 import resnet50

def is_valid_image(filepath):
    try:
        im = PIL.Image.open(filepath)
        # im.load()
    except:
        return False
    return True

class TargetTransform():
    def __init__(self, classes, class_to_idx):
        self.classes = list()
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        for c in classes:
            for word in c.split(' '):
                self.classes.append(word)
        self.classes = list(set(self.classes))
        self.classes.sort()
        self.class_to_target = {_class: idx for idx, _class in enumerate(self.classes)}

    def target_transformation(self, target):
        _class = self.idx_to_class[target]
        target = np.zeros(len(self.classes))
        for c in _class.split(' '):
            target[self.class_to_target[c]] = 1

        return target.astype(np.float32)


def main(args):
    if args.wandb_log:
        wandb.init(project= 'thumbnail_crawl', config=args)
        args = wandb.config

    # plt.ion()   # interactive mode

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

    IMG_EXTENSIONS = tuple(args.image_extensions)

    # data_transform = {
    #     'train': transforms.Compose([
    #     transforms.RandomResizedCrop((224, 224)),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_dir = args.data_dir
    seed = args.seed
    batch_size = args.batch_size
    split_ratio = args.split_ratio

    thumbnail_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transform,
                                                         is_valid_file=is_valid_image) ### jpg PIL.UnidentifiedImageError
                                                         # extensions=IMG_EXTENSIONS)
    target_transform = TargetTransform(thumbnail_dataset.classes, thumbnail_dataset.class_to_idx)
    thumbnail_dataset.target_transform = target_transform.target_transformation


    dataset_name = ['train', 'val']
    #dataset_idx = thumbnail_dataset.split_idx(split_ratio, seed)
    dataset_idx = {}
    dataset_idx['train'], dataset_idx['val'] = sklearn.model_selection.train_test_split(
                                list(range(len(thumbnail_dataset.targets))),
                                test_size=split_ratio[1], stratify=thumbnail_dataset.targets)

    image_datasets = {key: torch.utils.data.Subset(thumbnail_dataset, dataset_idx[key])
                        for key in dataset_name}

    # = {x: ThumbnailDataset(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=8)
                  for x in dataset_name}
    dataset_sizes = {x: len(image_datasets[x]) for x in dataset_name}
    class_names = target_transform.classes # thumbnail_dataset.classes
    print('Classification task: ', class_names)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    if not torch.cuda.is_available():
        print('Not using cuda!!!!!!!!!!!!!!!!!!!')

    # print('thumbnail_dataset.samples[5]', thumbnail_dataset.samples[5])
    # print('thumbnail_dataset.targets[5]', thumbnail_dataset.targets[5])
    #
    # print("image_datasets['train'][0]", image_datasets['train'][0])
    # print("image_datasets['val'][0]", image_datasets['val'][0])
    # print("image_datasets['test'][0]", image_datasets['test'][0]) if len(dataset_name) == 3 else None
    # print('dataset_sizes', dataset_sizes)
    # print('class_names', class_names)
    # print('device', device)
    # print('learning type', args.learning_type)

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    # visualize
    # imshow(out, title=str([class_names[x] for x in classes]))

    # 'og', 'transfer', 'finetune'
    model_ft = resnet50(args.learning_type, class_names, device)

    criterion = nn.BCELoss() # nn.CrossEntropyLoss() nn.MultiLabelSoftMarginLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size, gamma=0.1)

    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                           dataset_sizes, device, num_epochs=args.n_epochs, wandb_log=args.wandb_log)


    fname = ['['+data_dir.split('/')[-1]+']', 'ckpt', args.learning_type, 'batch', str(args.batch_size), 'n_epochs', str(args.n_epochs),
             'lr', str(args.lr), 'step_size', str(args.step_size), 'seed', str(args.seed)]
    fname = '_'.join(fname) + '.pt'
    torch.save(model_ft.state_dict(), os.path.join('model/weights/crawl/', fname))
    if args.wandb_log:
        wandb.save(os.path.join('model/weights/crawl/', 'ckpt*'))

    visualize_model(model_ft, dataloaders, class_names, args, device, data_dir, num_images=20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thumbnail dataset image classification.')

    parser.add_argument('-d', '--data_dir', default='/media/daftpunk2/home/seungheon/thumbnailDataset_google',
                        type=str,
                        help='path to the dataset.')
    parser.add_argument('-l', '--learning_type', default='finetune', choices=['og', 'transfer', 'finetune'],
                        help='learning type: og (learning from scratch. original), transfer (transfer learning), finetune (fine tuning).')
    parser.add_argument('-r', '--split_ratio', default=(0.8, 0.2), type=lambda x: len(x) in [2],# [2, 3],
                        help='ratio for train/validation(/test) dataset splitting.')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help='batch size for training.')
    parser.add_argument('-e', '--n_epochs', default=25
                        , type=int,
                        help='number of epochs for training.')
    parser.add_argument('-lr', '--lr', default=0.01, type=float,
                        help='learning rate for training optimizer.')
    parser.add_argument('-step', '--step_size', default=7, type=int,
                        help='step size for training scheduler.')
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed number.')
    parser.add_argument('-ext', '--image_extensions',
                        default=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'),
                        type=tuple, help='tuple of image extensions.')

    parser.add_argument('-w', '--wandb_log', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to log in wandb.')

    main(parser.parse_args())