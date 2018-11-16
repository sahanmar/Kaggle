# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:28:29 2018

@author: msahan
"""
import pandas as pd

def clean_data(data: pd.DataFrame):
    
    data = get_titles(data)
    #data = fill_na_age(data)
    
    
    
    
    
    return data


def fill_na_age(data: pd.DataFrame):
            
    data['Age'] = data['Age'].fillna(0)
    
    
    
    
    return data


def get_titles(data: pd.DataFrame):
    
    data['Name'] = data['Name'].str.split('.').str[0].str.split(',').str[1].str.split().str[0]
    data['Name'] = data['Name'].replace(['Capt', 'Don', 'Rev', 'Sir', 'Mr' ],'Mr')
    data['Name'] = data['Name'].replace(['Mlle', 'Ms', 'Miss'],'Miss')
    data['Name'] = data['Name'].replace(['Mme','Rothes', 'the', 'Lady', 'Dona'],'Mrs')
    data['Name'] = data['Name'].replace(['Dr', 'Jonkheer', 'Col', 'Major'],'Other')
    
    
    return data