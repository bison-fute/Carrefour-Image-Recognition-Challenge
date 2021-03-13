import numpy as np
import os, glob, csv, json
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch


class MetadataPreprocessing:
    "class description"

    def __init__(self, file):
        self.file = file

    def json_to_list_of_dict(self):
        """ Convert json to list of dict with keys 'barcode', 'item_desc', 'brand_name', 'nature', 'packaging',
        'arbonodes', 'structHyp', 'location'. Facing issues transforming directly the json to a dict as some fields
        have arbonodes arrays of different sizes. """
        metadata_list = []
        for line in open(self.file, 'r', encoding="utf-8"):
            metadata_list.append(json.loads(line))
        self.metadata_list = metadata_list
        return self.metadata_list

    def nb_of_products(self):
        self.nb_of_products = len(self.metadata_list)
        return self.nb_of_products

    def a_random_arbonodes(self, metadata_list):
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
                    element_arbonode_dict = element_arbonodes[i]
                    condition = element_arbonode_dict['is_primary_link'] == 'true'
                    if condition:
                        element['arbonodes'] = element_arbonode_dict
                        metadata_list_copy.append(element)
            else:  # element_arbonode_length == 1
                element['arbonodes'] = element_arbonodes[0]
                metadata_list_copy.append(element)
        self.metadata_list_copy = metadata_list_copy
        return self.metadata_list_copy

    def barcode_lastcat_dict(self):
        barcode_lastcat = {}
        exception_counter = 0
        for elt in self.metadata_list_copy:
            elt_barcode = int(elt['barcode'])
            barcode_lastcat[elt_barcode] = int(list(elt['arbonodes'].values())[-2])


        self.barcode_lastcat_dict = barcode_lastcat
        return self.barcode_lastcat_dict


class ExploratoryDataAnalysis():
    """ Now we need to an Exploratory Data Analysis to keep only categories with enough element in it. In order to
    do that, we should know which dataset we are using. """

    def __init__(self, dataset_paths, metadata_list_copy, min_cat_size):
        self.dataset_paths = dataset_paths  # part5/28641_Tout_pour_des_fêtes_réussies
        self.metadata_list_copy = metadata_list_copy
        self.barcode_lastcat_dict = {}
        self.min_cat_size = min_cat_size

        # dict barcode -> cat; où ne sont répertoriées que les catégories avec moins de sie_min items
        self.low_occurence_cat = {}

    def get_jpeg_barcode(self, path):
        """ Extract the barcodes from images filename. Some of them are incorrect. """
        return int(path.split('\\')[-1].split('_')[1])

    def paths_to_barcodes_as_a_set(self):
        """ Gives out a set object containing the valid barcodes. """
        barcodes = set()
        for path in self.dataset_paths:
            barcode = self.get_jpeg_barcode(path)
            barcodes.add(barcode)
        return barcodes

    def build_barcode_lastcat_dict(self):
        """ Builds a dictionary have key 'barcode' and value
         'last category' in the category tree"""
        barcode_lastcat = {}
        for elt in self.metadata_list_copy:
            elt_barcode = int(elt['barcode'])
            barcode_lastcat[elt_barcode] = int(list(elt['arbonodes'].values())[-2])

        self.barcode_lastcat_dict = barcode_lastcat
        print("ma len 1 :", len(self.barcode_lastcat_dict))
        t = np.unique(np.array(list(self.barcode_lastcat_dict.keys())))
        print("nb d'el uniques:", len(t))
        return None

    def cat_id_to_cat_name(self, cat_id):
        """ Gets category name from its id by building, if not already
         existing a dictionary between id and name for categories"""
        try:
            cat_name = self.cat_id_name_dict[cat_id]
            return cat_name
        except:
            cat_id_name_dict = {}
            for elt in self.metadata_list_copy:
                len_arbonodes = len(elt['arbonodes'])
                for i in range(len_arbonodes // 2):
                    _cat_id = int(list(elt['arbonodes'].values())[-(i + 1) * 2])
                    _cat_name = list(elt['arbonodes'].values())[-(i + 1) * 2 + 1]
                    cat_id_name_dict[_cat_id] = _cat_name
            self.cat_id_name_dict = cat_id_name_dict
            cat_name = self.cat_id_name_dict[cat_id]
            return cat_name

    def nb_of_element_per_category(self, plot_distrib=False):
        """ Returns the number of element per category. Has a plot_distrib
        argument to show or not the distribution over categories. """
        cat_quant = {}
        for bcode, lcat in self.barcode_lastcat_dict.items():
            if lcat in cat_quant.keys():
                cat_quant[lcat] += 1
            else:
                cat_quant[lcat] = 1
        self.cat_quant_dict = cat_quant
        if plot_distrib:  # for plotting the distribution over categories
            cat_labels = list(self.cat_quant_dict.keys())
            cat_labels_names = [self.cat_id_to_cat_name(i) for i in cat_labels]
            cat_labels_axis = np.arange(len(cat_labels))
            cat_quantity = list(self.cat_quant_dict.values())
            print('cat id', cat_labels)
            print('cat name', cat_labels_names)
            print('quantity', cat_quantity)
            fig, ax = plt.subplots()
            ax.bar(x=cat_labels_axis, height=cat_quantity, color='g')
            ax.set_xticks(cat_labels_axis)
            ax.set_xticklabels(cat_labels_names, rotation=25)
            ax.tick_params(axis='x', labelsize=7)
            plt.title("distribution over categories")
            ax.grid(True)
            plt.show()
        return self.cat_quant_dict

    def replace_cat(self, old_last_cat_id, level_change=1):
        """ Elementary function used for merge_cat_min_size that replace all
        the element of category old_last_at_id into their own upper-level category. """
        new_barcode_last_cat_dict = {}
        for barcode, last_cat in self.barcode_lastcat_dict.items():
            if last_cat == old_last_cat_id:  # if elt cat has to be replaced
                for el in self.metadata_list_copy:
                    if int(el['barcode']) == int(barcode):
                        elt = el
                        try:
                            new_barcode_last_cat_dict[barcode] = int(list(elt['arbonodes'].values())[-(1+level_change)*2])
                        except:
                            # print(self.cat_id_to_cat_name(last_cat), "has no upward category")

                            # on l'ajoute au dictionnaire des faibles catégories
                            self.low_occurence_cat[int(barcode)] = last_cat
            else:  # elt can remain as it is
                new_barcode_last_cat_dict[barcode] = last_cat
        self.barcode_lastcat_dict = new_barcode_last_cat_dict
        return self.barcode_lastcat_dict

    def merge_cat_for_min_size(self, min_cat_size, level=1):
        """ Recursive function that merges categories as long as they do not gather at least min_cat_size """
        # if level > 3:  # stop condition for recursive function
        #     return None
        self.nb_of_element_per_category()
        change_has_been_made = False
        print("Levels merging:", level)
        for i, (cat, size) in enumerate(self.cat_quant_dict.items()):
            print("\rPourcentage du level traitée: %d %%" % (100*i/len(self.cat_quant_dict)), end='')
            if size < min_cat_size:  # if cat is too small
                self.replace_cat(cat, level)  # update self.barcode_lastcat_dict
                change_has_been_made = True
        print()
        if change_has_been_made:  # if change has_been_made, must check new created cats sizes : iterate recursively
            self.merge_cat_for_min_size(min_cat_size, level + 1)
        else:
            # C'est la fin
            # On ajoute ls catégories à faible occurences qu'on n'a pas pu regrouper au dictionnaire
            for key, val in self.low_occurence_cat.items():
                self.barcode_lastcat_dict[key] = val
            print("Réduction terminée")
            print("Nombre de catégories avec une faible ocurence:", len(self.low_occurence_cat))
            return None

    def get_barcode_to_cat(self):
        self.build_barcode_lastcat_dict()
        self.merge_cat_for_min_size(self.min_cat_size)
        return self.barcode_lastcat_dict
