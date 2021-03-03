#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:23:36 2021

@author: brennanpj
"""

# 2.28.21
# KIR binding prediction algorithm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#amino acids and numbers
codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        

def num_encode(peptide_list):
    ''' input a single column array of peptides in single letter format to 
    generate a list of numerized peptides
    
    returns:  single column array of number-encoded peptides '''
    
    #initialize peptide sequence
    num_list = []
    for pep in peptide_list:
        
        #initialize amino acid sequence per peptide
        aa_seq = []
        for aa in pep:
            
            #change each peptide letter to a number from codes list
            aa_seq.append(int(codes.index(aa)))
            
        #add the numerized peptide to the big list
        num_list.append(aa_seq)
        
    return np.array(num_list)


# import TRAINING csv file (must be same directory as this .py file)
# first column should be ninemers in single letter amino acid code
# second column should be normalized binding values
training_data = pd.read_csv("p2LibData.csv", header = None)



'''set data and target'''
num_data = num_encode(training_data[0])

raw_target = np.array(training_data[1])

'''
Categorize peptides into set benchmarks for:
    non-binder (0), 
    mid-binders (1), and 
    high binders (2)
    
    Can change binning values if needed:
 '''


target = []


#binning values:
mid = 0.2
high = 0.3

for value in raw_target:
    if value < mid:
        target.append(0)
    elif value < high:
        target.append(1)
    else:
        target.append(2)


'''splits data into training set and testing set for model input'''

X_train, X_test, y_train, y_test = train_test_split(
        num_data, target, test_size=0.2, random_state=509)




'''set up Random Forest Classifier model'''

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)

print('Training Set Score:  ', "{:.1f} %".format(score*100))



'''input peptides you want to predict as a list of strings'''

predict_file = pd.read_csv("PredictPeptides.csv", header = None)

pred_pep = ['AAASKGAWV','AAASKGMWV','AAASKGRRV']

pred_pep = predict_file[0]


pred_scores = rf.predict_proba(num_encode(pred_pep))
print(pred_scores)





'''
finds the index of highest value of score set for each category 
    for a prediction peptide, then links that index to 
    the prediction peptides and the category
'''

for pep_index, score_set in enumerate(pred_scores):
    
    #finds highest index of score set, saves as score_index
    (score, score_index) = max((i,score_index) for score_index,i 
                                in enumerate(score_set))
  
    #the score_index (0, 1, 2)  will retrieve values from this list
    binders = ['Non-Binder', 'Medium Binder', 'High Binder']
    
    print(pred_pep[pep_index], binders[score_index])


    

