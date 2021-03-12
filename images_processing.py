import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import models,transforms,datasets
import time
import sys
import glob

## Set up
print("Versions:")
print("torch version:", torch.__version__)
print("system version: ", sys.version)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())
## Dataset Loading

# We need to normalize images following a defined mean / std if we use transfert learning:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

vgg_format = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

##
batch_size = 10

dsets = datasets.ImageFolder('../data/', vgg_format)

data_loader = torch.utils.data.DataLoader(
    dsets,
    batch_size=batch_size,
    num_workers=0
)

def get_ids(data_path):
    jpeg_filepaths = glob.glob(os.path.join(data_path, "**/*.jpeg"), recursive=True)

    # list all jpeg files in data
    def get_label(file_name):
        return file_name.split('\\')[-1]  #.split('_')[1]

    ids = []

    for f_name in jpeg_filepaths:
        ids.append(get_label(f_name))

    return ids

img_ids = get_ids('../data/')

##

count = 0
for i, data_t in enumerate(data_loader):
    if count == 0:
        x, ids = data_t
        ids = img_ids[i*batch_size: (i+1)*batch_size]
    count += 1
    break

##

def imshow(inp, title=None):
#   Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = np.clip(std * inp + mean, 0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


num = 7
imshow(x[num])
print(ids[num])

##
