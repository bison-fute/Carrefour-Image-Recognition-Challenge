import numpy as np
import os, glob, csv, json
import matplotlib.pyplot as plt


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

    def barcode_lastcat_dict(self):
        barcode_lastcat = {}
        for elt in self.metadata_list_copy:
            elt_barcode = int(elt['barcode'])
            # len_arbonodes = len(elt['arbonodes'])
            # print('get level number', len_arbonodes // 2)

            exception_counter = 0
            print_warning = False
            try:
                barcode_lastcat[elt_barcode] = int(list(elt['arbonodes'].values())[-2])
            except:
                if not print_warning:
                    print_warning == True
                exception_counter += 1
                pass
        if print_warning:
            print("number of exceptions encountered :", exception_counter)

        self.barcode_lastcat_dict = barcode_lastcat
        return self.barcode_lastcat_dict


# set metadata into a class
metadata = MetadataPreprocessing('metadata_images.json')

# convert json to a list of dict
metadata_list = metadata.json_to_list_of_dict()
metadata_nb_of_product = metadata.nb_of_products()

# show what describes the key 'arbonodes'
metadata.a_random_arbonodes()

# focus on the branch of is_primary_link of all products. Modify the json file removing is_primary_link = False fields.
metadata_list_copy = metadata.primary_link_branch()

# create an initial {barcode, last category} dictionary
metadata_barcode_lastcat_dict = metadata.barcode_lastcat_dict()


class ExploratoryDataAnalysis():
    """ Now we need to an Exploratory Data Analysis to keep only categories with enough element in it. In order to
    do that, we should know which dataset we are using. """

    def __init__(self, dataset_paths, metadata_list_copy, barcode_lastcat_dict):
        self.dataset_paths = dataset_paths  # part5/28641_Tout_pour_des_fêtes_réussies
        self.metadata_list_copy = metadata_list_copy
        self.barcode_lastcat_dict = barcode_lastcat_dict

    def get_jpeg_barcode(self, path):
        """ Extract the barcodes from images filename. Somme of them are incorrect. """
        try:
            return int(path[-30:-17])
        except:
            pass

    def paths_to_barcodes_as_a_set(self):
        """ Gives out a set object containing the valid barcodes. """
        barcodes = set()
        for path in self.dataset_paths:
            barcode = self.get_jpeg_barcode(path)
            if len(str(barcode)) == 13:
                barcodes.add(barcode)
        return barcodes

    def filter_metata_list_copy(self):
        """ filter the metadata list keeping only entities with a
        correct barcode, checking made using the built set of barcodes. """
        dataset_barcodes = self.paths_to_barcodes_as_a_set()
        filtered_metadata_list_copy = {}
        for metadata_barcode, metadata_lastcat in self.barcode_lastcat_dict.items():
            if metadata_barcode in dataset_barcodes:
                filtered_metadata_list_copy[metadata_barcode] = metadata_lastcat
        self.filtered_metadata_list_copy = filtered_metadata_list_copy
        return self.filtered_metadata_list_copy

    def build_barcode_lastcat_dict(self):
        """ Builds a dictionary have key 'barcode' and value
         'last category' in the category tree"""
        barcode_lastcat = {}
        for elt in self.metadata_list_copy:
            elt_barcode = int(elt['barcode'])
            exception_counter = 0
            print_warning = False
            try:
                barcode_lastcat[elt_barcode] = int(list(elt['arbonodes'].values())[-2])
            except:
                if not print_warning:
                    print_warning == True
                exception_counter += 1
                pass
        if print_warning:
            print("number of exceptions encountered :", exception_counter)

        self.barcode_lastcat_dict = barcode_lastcat
        return self.barcode_lastcat_dict

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
                # print('get level number', len_arbonodes // 2)
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
                    if el['barcode'] == barcode:
                        elt = el
                        if level_change == 1:
                            new_barcode_last_cat_dict[barcode] = int(list(elt['arbonodes'].values())[-4])
                            break
                        elif level_change == 2:
                            try:
                                new_barcode_last_cat_dict[barcode] = int(list(elt['arbonodes'].values())[-6])
                                break
                            except:
                                print(self.cat_id_to_cat_name(last_cat), "has no upward category")
                        elif level_change == 3:
                            try:
                                new_barcode_last_cat_dict[barcode] = int(list(elt['arbonodes'].values())[-8])
                                break
                            except:
                                print(self.cat_id_to_cat_name(last_cat), "has no upward category")
                        else:
                            break
            else:  # elt can remain as it is
                new_barcode_last_cat_dict[barcode] = last_cat
        self.barcode_lastcat_dict = new_barcode_last_cat_dict
        return self.barcode_lastcat_dict

    def merge_cat_for_min_size(self, min_cat_size, level=1):
        """ Recursive function that merges categories as long as they do not gather at least min_cat_size """
        if level > 3:  # stop condition for recursive function
            return None
        change_has_been_made = False
        for cat, size in self.cat_quant_dict.items():
            if size < min_cat_size:  # if cat is too small
                self.replace_cat(cat, level)  # update self.barcode_lastcat_dict
                change_has_been_made = True
            else:
                continue
        if change_has_been_made:  # if change has_been_made, must check new created cats sizes : iterate recursively
            self.merge_cat_for_min_size(min_cat_size, level + 1)

    @staticmethod
    def write_dict_into_csv(expected_filename, dictionary):
        """ Turns a dictionary into a two column (key, value) csv file. """
        with open(expected_filename, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in dictionary.items():
                writer.writerow([key, value])


# add path to used data and list all path in jpeg_filepaths array
data_path = "part5"
jpeg_filepaths = glob.glob(os.path.join(data_path, "**/*.jpeg"), recursive=True)  # list all jpeg files in data

# set up an eda object
eda = ExploratoryDataAnalysis(jpeg_filepaths, metadata_list_copy, metadata_barcode_lastcat_dict)

# list barcodes present in the dataset into a set
barcodes = eda.paths_to_barcodes_as_a_set()

filtered_metadata_list_copy = eda.filter_metata_list_copy()

# create a {barcode, last category} dictionary restricted to elements present in the dataset
dataset_barcode_lastcat_dict = eda.build_barcode_lastcat_dict()

# create a dict of number of element per category
nb_of_el_per_cat = eda.nb_of_element_per_category()

# merge categories to upper levels so that it has at least 50 products
eda.merge_cat_for_min_size(50)

# get final number of elements per category
nb_of_el_per_cat_filtered = eda.nb_of_element_per_category(plot_distrib=True)

# create final {barcode, category} dictionary used for the supervised learning tasK
final_barcode_lastcat_dict = eda.barcode_lastcat_dict

# write out the dict into a two columns csv used for the supervised learning task
eda.write_dict_into_csv("barcode-vs-category.csv", final_barcode_lastcat_dict)

# Next things to do (OLD)
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
