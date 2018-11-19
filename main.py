# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import data_cleaner as dc
import models



if __name__=='__main__':

    #Load data to pandas Dataframe
    train_csv = pd.read_csv(r'.\all\train.csv')
    test_csv = pd.read_csv(r'.\all\test.csv')
    
    #Get training labels 
    train_labels = train_csv.get_values()[:,1]
    
    #Drop unwanted columns
    train_csv = train_csv.drop(['Survived', 'PassengerId', 'Ticket', 'Cabin' ], axis=1)
    test_csv = test_csv.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
    
    
    
    #Union data
    union_data = pd.concat([train_csv,test_csv])
    
    
    #Clean dataset
    data = dc.clean_data(union_data)
    
    #Transform to numpy format
    data_numpy = data.get_values()
    
    #Split back to train and test 
    train_data = data_numpy[:len(train_csv),:]
    test_data = data_numpy[len(train_csv):,:]
    
    #K-fold cross validation  
    results = models.k_fold_cross_valid(train_data, train_labels, 10)
    
    #Create submission documents
    models.submission_doc(train_data, train_labels, test_data)
    
    
    
    