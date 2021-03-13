import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import models,transforms,datasets
import time
import sys
import glob
import pickle as pk

from metadata_processing import*

## Metadata Processing
min_cat_size = 50

# set up eda object
eda = ExploratoryDataAnalysis('metadata_images.json', min_cat_size)

# get barcode to cat dictionnary
barcode_to_cat = eda.get_barcode_to_cat()
cat_to_name = eda.get_cat_to_name()
eda.plot_distrib()
# écriture du dictionnaire en mémoire:
pk.dump(barcode_to_cat, open('barcode_to_cat', 'wb'))
pk.dump(cat_to_name, open('cat_to_name', 'wb'))

