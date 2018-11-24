# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:28:29 2018

@author: msahan
"""
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def clean_data(data: pd.DataFrame):
    
    #Get titles and combine them into 5 groups (Mr, Mrs, Miss, Master, Other)
    data = get_titles(data)
    #Fill NA age data with median values with respect to each title group
    data = fill_na_age(data)
    #Fill NA fare with median values to each title group
    data = fill_na_fare(data)
    #Fill NA embarked data
    data = fill_na_embarked(data)
    #Create new columns with familty size and "is alone" 
    data = family_size_col(data)
    
    #New column with fare quantile cut to bins
    data['FareBin'] = pd.qcut(data['Fare'], 4)
    
    #New column with age cut to bins
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
    
    
    #Create dummies array
    dummies_arr = ['Name', 'Sex', 'Embarked', 'AgeBin', 'FareBin']
    
    #Create dummy columns
    for col in dummies_arr:
        data = create_dummies(data,col)
        data = data.drop(col, axis=1)
    
    #Columns for normalization    
    norm_col = ['Fare', 'Age']
    
    #Loop for normalization
    for col in norm_col:
        data = normalize_col(data, col)
        
            
    return data


def fill_na_age(data: pd.DataFrame): 
    
    medians = data.groupby(['Name'])['Age'].median()
    data = data.set_index(['Name'])
    data['Age'] = data['Age'].fillna(medians)
    data = data.reset_index()
    
    return data


def get_titles(data: pd.DataFrame):
    
    data['Name'] = data['Name'].str.split('.').str[0].str.split(',').str[1].str.split().str[0]
    data['Name'] = data['Name'].replace(['Capt', 'Don', 'Rev', 'Sir', 'Mr' ],'Mr')
    data['Name'] = data['Name'].replace(['Mlle', 'Ms', 'Miss'],'Miss')
    data['Name'] = data['Name'].replace(['Mme','Rothes', 'the', 'Lady', 'Dona'],'Mrs')
    data['Name'] = data['Name'].replace(['Dr', 'Jonkheer', 'Col', 'Major'],'Other')
    
    return data


def fill_na_fare(data: pd.DataFrame):
   
    median = data['Fare'].median()
    data['Fare'] = data['Fare'].fillna(median)
    
    return data


def fill_na_embarked(data: pd.DataFrame):
    
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    
    return data


def create_dummies(data: pd.DataFrame, column_name: str):

    #dummies = pd.get_dummies(data[column_name], prefix=column_name)
    #data = pd.concat([data, dummies], axis=1)
    label = preprocessing.LabelEncoder()
    data[column_name+'_Code'] = label.fit_transform(data[column_name])

    return data


def normalize_col(data: pd.DataFrame, col: str):
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(np.asarray(data[col]).reshape(-1, 1))
    data[col] = pd.DataFrame(x_scaled)
    
    return data

    
def family_size_col(data: pd.DataFrame):
    
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1    
    
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0
    
    return data


def decision_tree_data(data: pd.DataFrame):
    
    drop_columns = ['Pclass', 'Age', 'SibSp', 'Parch','Fare', 'Embarked']
    
    data['Tittle'] = data['Name'].str.split('.').str[0].str.split(',').str[1].str.split().str[0]
    data['Tittle'] = data['Tittle'].replace(['Capt', 'Don', 'Rev', 'Sir', 'Mr' ],'Mr')
    data['Tittle'] = data['Tittle'].replace(['Mlle', 'Ms', 'Miss'],'Miss')
    data['Tittle'] = data['Tittle'].replace(['Mme','Rothes', 'the', 'Lady', 'Dona'],'Mrs')
    data['Tittle'] = data['Tittle'].replace(['Dr', 'Jonkheer', 'Col', 'Major'],'Other')
    
    data['Name'] = data['Name'].str.split().str[0].str.split(',').str[0]
    
    dummies_arr = ['Sex']
    
    #Create dummy columns
    for col in dummies_arr:
        data = create_dummies(data,col)
        data = data.drop(col, axis=1)
    
    for col in drop_columns:
        data = data.drop(col, axis=1)
    
    return data

    
