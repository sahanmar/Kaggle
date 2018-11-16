# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import data_cleaner as dc
import numpy as np



if __name__=='__main__':

    #Load data to pandas Dataframe
    train_csv = pd.read_csv(r'C:\Users\msahan\Documents\Python Scripts\Kaggle\Titanic\all\train.csv')
    test_csv = pd.read_csv(r'C:\Users\msahan\Documents\Python Scripts\Kaggle\Titanic\all\test.csv')
    
    #Get training labels 
    train_labels = train_csv.get_values()[:,1]
    
    #Drop unwanted columns
    train_csv = train_csv.drop(['Survived','PassengerId'], axis=1)
    test_csv = test_csv.drop('PassengerId', axis=1)
    
    #Union data
    union_data = pd.concat([train_csv,test_csv])
    
    
    
    data = dc.clean_data(union_data)
    
    
    
    