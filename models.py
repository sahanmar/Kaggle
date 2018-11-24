# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:16:23 2018

@author: msahan
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree, model_selection


from xgboost import XGBClassifier



def decision_tree(train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray):
    
    test_labels = np.zeros((len(test_data),))
    
    #Loop through all documents 
    for counter_test in range(len(test_data)):
        
        #initial pred (everybody died)
        pred = 0
        
        #If Male
        if test_data[counter_test,2] == 1:
            #If Boy
            if test_data[counter_test,1] == 'Master':
                
                ### If all family females died then then boy died ###
                
                #create counter that counts amount of family members
                family_counter = 0
                #create counter that counts amount of survived family members
                family_surv_counter = 0
                #Loop through training dataset 
                for counter_train in range(len(train_data)):
                    #If family member is female or boy
                    if train_data[counter_train, 0] == test_data[counter_test, 0] and train_data[counter_train, 1] in ['Master','Mrs','Miss']:
                        family_counter += 1
                        #If family member survived
                        if train_labels[counter_train] == 0:
                            family_surv_counter += 1
                #if family memebers equals to family memebers who not survived 
                #then prediction is zero
                if family_surv_counter == family_counter and family_surv_counter != 0:
                    pred = 0
                else:  
                    pred = 1
            #If male then prediction is zero    
            else:
                pred = 0
        else:
            #If female
            #create counter that counts amount of family members
            family_counter = 0
            #create counter that counts amount of survived family members
            family_surv_counter = 0
            #Loop through training dataset 
            for counter_train in range(len(train_data)):
                    #If family member is female or boy
                    if train_data[counter_train, 0] == test_data[counter_test, 0] and train_data[counter_train, 1] in ['Master','Mrs','Miss']:
                        family_counter += 1
                        #If family member survived
                        if train_labels[counter_train] == 0:
                            family_surv_counter += 1
            #if family memebers equals to family memebers who not survived 
            #then prediction is zero                
            if family_surv_counter == family_counter and family_surv_counter != 0:
                pred = 0
            else:  
                pred = 1
                
        test_labels[counter_test] = pred
        
    return test_labels
        

def k_fold_cross_valid(data: np.ndarray, labels: np.ndarray, split: int):
    
    kf = KFold(n_splits=split)
    kf.get_n_splits(data)
        
    #data = data.astype('float')
    #labels = labels.astype('float')
    
    scores_array = []
    
    '''
    
    param_grid = {'criterion': ['gini', 'entropy'], 
                  'max_depth': [2,4,6,8,10,None],   
                  #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
                  #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
                  #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all
                  'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
                  }
    
    
    tune_model = model_selection.GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = split)
    tune_model.fit(data, labels)
        
    print('AFTER DT Parameters: ', tune_model.best_params_)
    #print(tune_model.cv_results_['mean_train_score'])
    print("AFTER DT Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
    #print(tune_model.cv_results_['mean_test_score'])
    print("AFTER DT Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
    print("AFTER DT Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
    print('-'*10)
    
    '''
    for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]  
            #predicted, scores = knn.feed_forward_NN(X_train,y_train, X_test, y_test)
            #scores_array.append(scores)
            
            #clf = RandomForestClassifier(criterion = 'entropy', 
            #                     max_depth = 6,
            #                     n_estimators = 100,
            #                     oob_score = True,
            #                     random_state = 0)

            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #clf = AdaBoostClassifier()
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #clf = ExtraTreesClassifier()
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #clf = GradientBoostingClassifier()
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #clf = VotingClassifier()
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #clf = GaussianNB()
            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            #model = XGBClassifier()
            #model.fit(X_train, y_train)
            #y_pred = model.predict(X_test)
            #predictions = [round(value) for value in y_pred]
            #scores_array.append(accuracy_score(y_test, predictions))
            
            #clf = svm.SVC(decision_function_shape='ovo')
            #clf.fit(X_train, y_train) 
            #y_pred = clf.predict(X_test)
            #scores_array.append(accuracy_score(y_test, y_pred))
            
            y_pred = decision_tree(X_train, y_train, X_test)
            scores_array.append(1-np.abs(y_test-y_pred).sum()/len(y_pred))
            
   
    print(np.mean(scores_array))  
    return scores_array
    
     
    
def submission_doc(train_data: np.ndarray, labels: np.ndarray, test_data: np.ndarray):  
    '''
    train_data = train_data.astype('float')
    test_data = test_data.astype('float')
    labels = labels.astype('float')
    
    #Optimal parameters from the function above 
    clf = RandomForestClassifier(criterion = 'entropy', 
                                 max_depth = 6,
                                 n_estimators = 100,
                                 oob_score = True,
                                 random_state = 0)
    
    clf.fit(train_data, labels)
    
    predictions = clf.predict(test_data).astype(int)
    '''
    
    predictions = decision_tree(train_data, labels, test_data).astype(int)
      
    test_csv = pd.read_csv('./all/test.csv')
    submission = pd.DataFrame({'PassengerId':test_csv['PassengerId'],'Survived':predictions})   
    
    filename = 'Titanic_Predictions.csv'

    submission.to_csv(filename,index=False)
    