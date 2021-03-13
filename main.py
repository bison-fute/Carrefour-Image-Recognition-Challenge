import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import models,transforms,datasets
from torchvision.utils import make_grid
import time
import sys
import glob
import pickle as pk

from images_processing import*

## Set up de PyTorch
print("Versions:")
print("torch version:", torch.__version__)
print("system version: ", sys.version)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

## Dataset Loading
data_path = "data/"

# Note: commencer par exécuter metadata_main
barcode_to_cat = pk.load(open('barcode_to_cat', 'rb'))
batch_size = 64

# Voir si on a besoin de normaliser (ça dépend du model utilisé)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

vgg_transform = transforms.Compose([
                transforms.RandomAffine(180, fillcolor=(255, 255, 255)),
                transforms.RandomGrayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.RandomPerspective(fill=(255, 255, 255)),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

dataset = ImageFolderWithPaths(data_path, vgg_transform, barcode_to_cat) # our custom dataset
train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train_dsets, test_dsets = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
    train_dsets,
    batch_size,
    num_workers=0
)
tets_loader = torch.utils.data.DataLoader(
    test_dsets,
    batch_size,
    num_workers=0
)

# iterate over data
for x, y in train_loader:
    break

# visualisation des premières images, avec leurs catégories:
imshow(make_grid(x[:32]))
print(y[:32])

##

