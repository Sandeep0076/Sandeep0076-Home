# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:40:02 2020

@author: Sandeep
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('covid19.csv')
dataset = dataset.loc[dataset['countriesAndTerritories'] == 'Germany']
X = dataset.iloc[:, [0,4]].values
y = dataset.iloc[:,5].values

plt.scatter(X,y,color ='red')