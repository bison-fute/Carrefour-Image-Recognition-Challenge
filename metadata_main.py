import pickle as pk
from metadata_processing import *

# Metadata Processing
min_cat_size = 50

# set up eda object
eda = ExploratoryDataAnalysis('metadata_images.json', min_cat_size)

# get barcode to cat dictionary
barcode_to_cat = eda.get_barcode_to_cat()
cat_to_name = eda.get_cat_to_name()
eda.plot_distrib()

# writing the dictionary in memory:
pk.dump(barcode_to_cat, open('barcode_to_cat99', 'wb'))
pk.dump(cat_to_name, open('cat_to_name99', 'wb'))

