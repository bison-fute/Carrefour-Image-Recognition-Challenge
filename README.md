# Carrefour-Image-Recognition-Challenge

## Next things to do :

**Save the output of the pretrained architecture** once *in order to save time during the learning process*.

Add a function to **get learning curves** (train and validation loss) *to be able to discuss about the learning process afterwards*.

**Train several architectures** : https://pytorch.org/vision/stable/models.html (VGG16, ResNet50, InceptionV3, DenseNet161).

Try **different *head* part architectures** : add convolution layers ? Several fully connected ?

Try **several set of parameters for data preprocessing** of categories : should categories be merge up to 50, 100, 500, 1000 elements ? How should we determine this ?


## Content : 

File metadata-processing.py : 
* a class MetadataPreprocessing : set metadata into a class, convert json to a list of dict, show what describes the key 'arbonodes', focus on the branch of is_primary_link of all products. Modify the json file removing is_primary_link = False fields.

File sujet.pdf : les slides données au début du challenge.

File sujet_precision_conf.pdf : les notes de la conférence, deuxième prise de vue.


**About evalutation of the project...**

A *written report*: send a small report describing the problem, the approach taken (data pre-processing, algorithms used), and the results obtained. Deadline: Friday, March 26. 

A *short oral presentation* of the work (15 minutes) with Data Scientists of Carrefour. Scheduled: Thursday, April 1st, 5:00 pm to 6:30 pm approximately. 
