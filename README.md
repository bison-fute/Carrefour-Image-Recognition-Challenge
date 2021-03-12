# Carrefour-Image-Recognition-Challenge

## Content : 

File metadata-processing.py : 
* a class MetadataPreprocessing : set metadata into a class, convert json to a list of dict, show what describes the key 'arbonodes', focus on the branch of is_primary_link of all products. Modify the json file removing is_primary_link = False fields.

File sujet.pdf : les slides données au début du challenge.

File sujet_precision_conf.pdf : les notes de la conférence, deuxième prise de vue.

## Next things to do :

STRATEGY 1 : define a level of categorization to perform a simple classification task, build a MetadataInspection to search which level is the more interesting 

STRATEGY 2 : build a moodle which use several categorization level between 1 and 4, which one ? how ? 

STRATEGY 3 : take profit of the desc field in the meta data to build an NLP model paired with a character recognition API (a priori name of product appears in desc)

STRATEGY 4 : try to take profit data which is not in the primary link branch in a model 

## Evalutation of the project

 Will be done in two ways: 
- a written report: each group must send a small report describing the machine learning problem, the approach taken (data pre-processing, algorithms used, etc...), and the results obtained. The deadline for submitting this report is Friday, March 26. 
- By a short oral presentation of the work done (15 minutes) for each group in front of Data Scientists of the Carrefour DataLab, which we have planned to schedule on Thursday, April 1st from 5:00 pm to 6:30 pm approximately. 
