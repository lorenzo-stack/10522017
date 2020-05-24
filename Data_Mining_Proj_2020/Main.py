#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020

@author: lorenzo
"""

#import pandas as pd

import pandas as pd

#import the train data and convert into a dataframe

df = pd.read_csv('/Users/lorenzo/Desktop/Data_Mining_Proj_2020/train.csv', sep = ',')

count_row = df.shape[0]

print("Number of rows of the dataset " + str(count_row))

# <---- Preproc ------>

#Step 1) Count missing values for each attribute

print(df.isnull().sum())
    
#step 2) How to treat missing values?
   
