import numpy as np
import os, glob, csv, json
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch


class MetadataPreprocessing:
    "class description"

    def __init__(self, file):
        self.file = file
        self.metadata_list = []
        self.nb_of_products = 0
        self.metadata_list_copy = []

    def json_to_list_of_dict(self):
        """ Convert json to list of dict with keys 'barcode', 'item_desc', 'brand_name', 'nature', 'packaging',
        'arbonodes', 'structHyp', 'location'. Facing issues transforming directly the json to a dict as some fields
        have arbonodes arrays of different sizes. """
        metadata_list = []
        for line in open(self.file, 'r', encoding="utf-8"):
            self.metadata_list.append(json.loads(line))
            self.nb_of_products += 1

    def a_random_arbonodes(self):
        """ Show what describes the key 'arbonodes'. """
        a_product_idx = np.random.randint(0, self.nb_of_products)
        example_of_arbonodes = self.metadata_list[a_product_idx]['arbonodes']
        print('An example of arbonodes :')
        for i in range(len(example_of_arbonodes)):
            print(example_of_arbonodes[i].items())

    def primary_link_branch(self):
        """ Focus on the branch of is_primary_link of all products. Modify the json file removing
        is_primary_link = False fields. """
        for idx, element in enumerate(self.metadata_list):
            element_arbonodes = element['arbonodes']
            element_arbonodes_length = len(element_arbonodes)
            if element_arbonodes_length > 1:
                for i in range(element_arbonodes_length):
                    element_arbonode_dict = element_arbonodes[i]
                    condition = element_arbonode_dict['is_primary_link'] == 'true'
                    if condition:
                        element['arbonodes'] = element_arbonode_dict
                        self.metadata_list_copy.append(element)
            else:  # element_arbonode_length == 1
                element['arbonodes'] = element_arbonodes[0]
                self.metadata_list_copy.append(element)

    def get_metadata_list_copy(self):
        self.json_to_list_of_dict()
        self.primary_link_branch()
        return self.metadata_list_copy


class ExploratoryDataAnalysis():
    """ Now we need to an Exploratory Data Analysis to keep only categories with enough element in it. In order to
    do that, we should know which dataset we are using. """

    def __init__(self, file_name, min_cat_size):
        self.min_cat_size = min_cat_size
        self.barcode_to_cat = {}  # dict barcode -> category id
        self.low_occurence_cat = {}  # dict barcode -> cat where only categories with less than sie_min items are listed
        self.cat_to_name = {}  # dict category id -> categorie name
        self.cat_to_quant = {}
        self.mp = MetadataPreprocessing(file_name)
        self.metadata_list_copy = self.mp.get_metadata_list_copy()
        self.init_cat_to_name()
        self.build_barcode_lastcat_dict()
        self.merge_cat_for_min_size(self.min_cat_size)
        self.normalize_cat_ids()

    def build_barcode_lastcat_dict(self):
        """ Builds a dictionary have key 'barcode' and value
         'last category' in the category tree"""
        barcode_lastcat = {}
        for elt in self.metadata_list_copy:
            elt_barcode = int(elt['barcode'])
            barcode_lastcat[elt_barcode] = int(list(elt['arbonodes'].values())[-2])
        self.barcode_to_cat = barcode_lastcat
        return None

    def init_cat_to_name(self):
        """ Build cat_to_name dictionary """
        for elt in self.metadata_list_copy:
            len_arbonodes = len(elt['arbonodes'])
            for i in range(len_arbonodes // 2):
                _cat_id = int(list(elt['arbonodes'].values())[-(i + 1) * 2])
                _cat_name = list(elt['arbonodes'].values())[-(i + 1) * 2 + 1]
                self.cat_to_name[_cat_id] = _cat_name

    def update_cat_to_quant(self):
        """ Returns the number of element per category. Has a plot_distrib
        argument to show or not the distribution over categories. """
        new_cat_to_quant = {}
        for bcode, lcat in self.barcode_to_cat.items():
            if lcat in new_cat_to_quant.keys():
                new_cat_to_quant[lcat] += 1
            else:
                new_cat_to_quant[lcat] = 1
        self.cat_to_quant = new_cat_to_quant

    def plot_distrib(self):   # for plotting the distribution over categories
        cat_labels = list(self.cat_to_quant.keys())
        cat_labels_names = [self.cat_to_name[i] for i in cat_labels]
        cat_labels_axis = np.arange(len(cat_labels))
        cat_quantity = list(self.cat_to_quant.values())
        fig, ax = plt.subplots()
        ax.bar(x=cat_labels_axis, height=cat_quantity, color='g')
        ax.set_xticks(cat_labels_axis)
        ax.set_xticklabels(cat_labels_names, rotation=25)
        ax.tick_params(axis='x', labelsize=7)
        plt.title("distribution over categories")
        ax.grid(True)
        plt.show()

    def replace_cat(self, old_last_cat_id, level_change=1):
        """ Elementary function used for merge_cat_min_size that replace all
        the element of category old_last_at_id into their own upper-level category. """
        new_barcode_last_cat_dict = {}
        for barcode, last_cat in self.barcode_to_cat.items():
            if last_cat == old_last_cat_id:  # if elt cat has to be replaced
                for el in self.metadata_list_copy:
                    if int(el['barcode']) == int(barcode):
                        elt = el
                        try:
                            new_barcode_last_cat_dict[barcode] = int(list(elt['arbonodes'].values())[-(1+level_change)*2])
                        except:
                            # it is added to the dictionary of weak categories
                            self.low_occurence_cat[int(barcode)] = last_cat
            else:  # elt can remain as it is
                new_barcode_last_cat_dict[barcode] = last_cat
        self.barcode_to_cat = new_barcode_last_cat_dict
        return self.barcode_to_cat

    def merge_cat_for_min_size(self, min_cat_size, level=1):
        """ Recursive function that merges categories as
        long as they do not gather at least min_cat_size. """
        self.update_cat_to_quant()
        change_has_been_made = False
        print("Levels merging:", level)
        for i, (cat, size) in enumerate(self.cat_to_quant.items()):
            print("\rPourcentage du level traité: %d %%" % (100 * i / len(self.cat_to_quant)), end='')
            if size < min_cat_size:  # if cat is too small
                self.replace_cat(cat, level)  # update self.barcode_lastcat_dict
                change_has_been_made = True
        print()
        if change_has_been_made:  # if change has_been_made, must check new created cats sizes : iterate recursively
            self.merge_cat_for_min_size(min_cat_size, level + 1)
        else:
            # This is the end. We add the categories with low occurrences that we could not group in the dictionary
            for key, val in self.low_occurence_cat.items():
                self.barcode_to_cat[key] = val
            print("Réduction terminée")
            print("Nombre de catégories avec une faible ocurence:", len(self.low_occurence_cat))
            return None

    def normalize_cat_ids(self):
        """
        Changes the category ids to be between 0 and cat_max. Also updates
        cat_to_name and cat_to_quant dictionaries
        """
        new_cat_to_name = {}
        new_barcode_to_cat = {}
        cat_to_new_cat = {}
        counter = 0
        for barcode, cat in self.barcode_to_cat.items():
            if not cat in cat_to_new_cat.keys():
                new_cat = counter
                counter += 1
            else:
                new_cat = cat_to_new_cat[cat]
            cat_to_new_cat[cat] = new_cat
            new_cat_to_name[new_cat] = self.cat_to_name[cat]
            new_barcode_to_cat[barcode] = new_cat

        self.barcode_to_cat = new_barcode_to_cat
        self.cat_to_name = new_cat_to_name
        self.update_cat_to_quant()
        print("We finally obtain %d different categories"
              % len(np.unique(np.array(list(self.barcode_to_cat.values())))))

    def get_barcode_to_cat(self):
        return self.barcode_to_cat

    def get_cat_to_name(self):
        return self.cat_to_name