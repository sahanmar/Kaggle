#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:18:58 2018

@author: mark
"""

import pandas as pd 
import Keras_NN as knn
import numpy as np
import umap
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score




if __name__=='__main__':
    
    
    '''
    *********************************
    Importing data and filtering them
    *********************************
    '''


    train_csv = pd.read_csv('./all/train.csv')
    
    train_csv = train_csv.drop('Ticket',1)
    #train_csv = train_csv.drop('Cabin',1) 
    train_csv = train_csv.drop('PassengerId',1)  
    train_csv = train_csv.fillna(0)
    train_csv['Cabin'] = train_csv.Cabin.astype(str).apply(lambda x: np.where(x.isdigit(),x,'1'))
    train_csv = train_csv.replace('male',0)
    train_csv = train_csv.replace('female',1)
    train_csv = train_csv.replace('C',1)
    train_csv = train_csv.replace('S',2)
    train_csv = train_csv.replace('Q',3)  
    train_csv = train_csv.drop('Embarked',1)
    train_data = train_csv.get_values()
    
    test_csv = pd.read_csv('./all/test.csv')
    
    test_csv = test_csv.drop('Ticket',1)
    #test_csv = test_csv.drop('Cabin',1)  
    test_csv = test_csv.drop('PassengerId',1)  
    test_csv = test_csv.fillna(0)
    test_csv['Cabin'] = test_csv.Cabin.astype(str).apply(lambda x: np.where(x.isdigit(),x,'1'))
    test_csv = test_csv.replace('male',0)
    test_csv = test_csv.replace('female',1)
    test_csv = test_csv.replace('C',1)
    test_csv = test_csv.replace('S',2)
    test_csv = test_csv.replace('Q',3)
    test_csv = test_csv.drop('Embarked',1)
    test_data = test_csv.get_values()
    
    
    '''
    *********************************
    Creating dictionary from tittle
    of people
    *********************************
    '''
    
    
    titles = {' Don':0, ' Col':1 , ' Dona':2 , ' Dr': 3, ' Master':4, ' Miss':5,
              ' Mr':6, ' Mrs':7, ' Ms':8, ' Rev':9, ' Mme':10, ' Major':11 , 
              ' Sir':12,' Lady':13, ' Mlle':14, ' Capt':15, ' the Countess':16,
              ' Jonkheer':17}
    
    
    '''
    *********************************
    Creating empty array for extended
    to connect titles and previous 
    data
    *********************************
    '''
    
    
    test_data_extended = np.zeros((len(test_data),24))
    train_data_extended = np.zeros((len(train_data),24))
    
    
    train_labels = train_data[:,0]
    train_data = train_data[:,1:]
    
    for counter in range(len(test_data)):
        test_data_extended[counter,6+titles[test_data[counter,1].split(',')[1].split('.')[0]]] = 1
        
    for counter in range(len(train_data)):
        train_data_extended[counter,6+titles[train_data[counter,1].split(',')[1].split('.')[0]]] = 1  
    
    test_data = np.delete(test_data, 1, 1)
    train_data = np.delete(train_data, 1, 1)
        
    train_data_extended[:,:7] = train_data
    test_data_extended[:,:7] = test_data
     
    
    '''
    *********************************
    Converting data in the float 
    format for further usage in the
    random forest algorithm and 
    defining K-fold cross-validation
    object
    *********************************
    '''
    
    
    train_data_extended = train_data_extended.astype('float')
    test_data_extended = test_data_extended.astype('float')
    train_labels = train_labels.astype('float') 
    
    
    kf = KFold(n_splits=6)
    kf.get_n_splits(train_data)
    
    train_data = train_data_extended[:,:12]
    train_data = train_data.astype('float')
    train_labels = train_labels.astype('float')
    
    test_data = test_data.astype('float')
    
        
    scores_array = []
     
    
    '''
    *********************************
    K-fold cross-validation with 
    random forest algorithm
    *********************************
    '''
    
    
    for train_index, test_index in kf.split(train_data):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]  
        #predicted, scores = knn.feed_forward_NN(X_train,y_train, X_test, y_test)
        #scores_array.append(scores)
        
        clf = RandomForestClassifier(n_estimators=400, max_depth=11,random_state=512)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores_array.append(accuracy_score(y_test, y_pred))
    print(np.mean(scores_array))    
     
    
    '''
    *********************************
    To submit task, unkomment this 
    lines and it will create csv 
    with the output data with respect
    to random forest algorithm
    *********************************
    '''
    
    
    '''
    clf = RandomForestClassifier(n_estimators=300, max_depth=11,random_state=0)
    clf.fit(train_data_extended, train_labels)
    
    predictions = clf.predict(test_data_extended).astype(int)
    
    test_csv = pd.read_csv('./all/test.csv')
    submission = pd.DataFrame({'PassengerId':test_csv['PassengerId'],'Survived':predictions})   
    
    
    filename = 'Titanic_Predictions.csv'

    submission.to_csv(filename,index=False)
    '''
