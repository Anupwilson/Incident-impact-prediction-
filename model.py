# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 12:28:20 2021

@author: Anup w
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('incident.csv')


X = dataset.loc[:,['open_by','loc','categoryy','caller_id','subbcategory','symptom','createdby','assigned_to','resolved_by','opened_at_day']]

y = dataset.loc[:, ['i_impact']]
def convert_to_word(int):
    int_dict = {1:'High', 2:'Medium', 3:'Low'}
    return int_dict[int]
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2403,397,165,215,81,89,76,43,111,45]]))