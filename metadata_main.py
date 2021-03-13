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
data_path = "data/"
min_cat_size = 50

# Part 1
# set metadata into a class
metadata = MetadataPreprocessing('metadata_images.json')

# convert json to a list of dict
metadata_list = metadata.json_to_list_of_dict()
metadata_nb_of_product = metadata.nb_of_products()

# show what describes the key 'arbonodes'
metadata.a_random_arbonodes(metadata_list)

# focus on the branch of is_primary_link of all products. Modify the json file removing is_primary_link = False fields.
metadata_list_copy = metadata.primary_link_branch()

# Part 2

jpeg_filepaths = glob.glob(os.path.join(data_path, "**/*.jpeg"), recursive=True)  # list all jpeg files in data
png_filepaths = glob.glob(os.path.join(data_path, "**/*.png"), recursive=True)
filepaths = jpeg_filepaths + png_filepaths
all_paths = glob.glob(os.path.join(data_path, "**/*.*"), recursive=True)
try:            # Vérification que tous les fichiers ont bien été traités
    assert len(filepaths) == len(all_paths)
except:
    print("Formats de fichiers non pris en compte...")
    print("Fichiers concernés:")
    print(set(all_paths) - set(filepaths))
    raise AssertionError

# set up an eda object
eda = ExploratoryDataAnalysis(filepaths, metadata_list_copy, min_cat_size)

# create final {barcode, category} dictionary used for the supervised learning tasK
barcode_to_cat = eda.get_barcode_to_cat()

# écriture du dictionnaire en mémoire:
pk.dump(barcode_to_cat, open('barcode_to_cat', 'wb'))
