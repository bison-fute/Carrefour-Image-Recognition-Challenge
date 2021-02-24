import json
import numpy as np


class MetadataPreprocessing:
    "class description"
    def __init__(self, file):
        self.file = file

    def json_to_list_of_dict(self):
        """ Convert json to list of dict with keys 'barcode', 'item_desc', 'brand_name', 'nature', 'packaging',
        'arbonodes', 'structHyp', 'location'. Facing issues transforming directly the json to a dict as some fields
        have arbonodes arrays of different sizes. """
        metadata_list = []
        for line in open(self.file, 'r'):
            metadata_list.append(json.loads(line))
        self.metadata_list = metadata_list
        return self.metadata_list

    def nb_of_products(self):
        self.nb_of_products = len(self.metadata_list)
        return self.nb_of_products

    def a_random_arbonodes(self):
        """ Show what describes the key 'arbonodes'. """
        a_product_idx = np.random.randint(0, self.nb_of_products)
        example_of_arbonodes = metadata_list[a_product_idx]['arbonodes']
        print('An example of arbonodes :')
        for i in range(len(example_of_arbonodes)):
            print(example_of_arbonodes[i].items())

    def primary_link_branch(self):
        """ Focus on the branch of is_primary_link of all products. Modify the json file removing
        is_primary_link = False fields. """
        metadata_list_copy = []
        for idx, element in enumerate(self.metadata_list):
            element_arbonodes = element['arbonodes']
            element_arbonodes_length = len(element_arbonodes)
            if element_arbonodes_length > 1:
                for i in range(element_arbonodes_length):
                    element_arbonode = element_arbonodes[i]
                    condition = element_arbonode['is_primary_link'] == 'true'
                    if condition:
                        element['arbonodes'] = element_arbonode
                        metadata_list_copy.append(element)
            else:  # element_arbonode_length == 1
                metadata_list_copy.append(element)
        self.metadata_list_copy = metadata_list_copy
        return self.metadata_list_copy


# set metadata into a class
metadata = MetadataPreprocessing('metadata_images.json')

# convert json to a list of dict
metadata_list = metadata.json_to_list_of_dict()
metadata_nb_of_product = metadata.nb_of_products()

# show what describes the key 'arbonodes'
metadata.a_random_arbonodes()

# Focus on the branch of is_primary_link of all products. Modify the json file removing is_primary_link = False fields.
metadata_list_copy = metadata.primary_link_branch()

# Next things to do
# =================
# STRATEGY 1 :
#   define a level of categorization to perform a simple classification task
#   build a MetadataInspection to search which level is the more interesting
# STRATEGY 2 :
#   build a moodle which use several categorization level between 1 and 4
#   which one ? how ?
# STRATEGY 3 :
#   take profit of the desc field in the meta data to build an NLP model paired
#   with a character recognition API (a priori name of product appears in desc)
# STRATEGY 4 :
#   try to take profit data which is not in the primary link branch in a model

categorization_level = '?'
