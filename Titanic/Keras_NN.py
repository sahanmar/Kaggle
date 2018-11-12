#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 21:10:57 2018

@author: mark
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np


def feed_forward_NN(train_data,train_features, test_data, test_features):
    predicted_values = np.zeros((len(test_data), 1))
    
    model = Sequential()
    
    model.add(Dense(100, input_dim=len(train_data[0])))#,kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Activation('sigmoid'))
    Dropout(0.1)
    
    model.add(Dense(100))
    model.add(Activation('sigmoid'))
    Dropout(0.1)
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy', metrics=['acc'])
    model.fit(x=train_data,y=train_features, batch_size=None, epochs=2000, verbose=1)
    
    predicted_values = model.predict(test_data)
    
    scores = model.evaluate(test_data, test_features)
    
    return predicted_values, scores[1]
    
    
    
    

