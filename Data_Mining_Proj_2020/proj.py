#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020

@author: lorenzo
"""

#import pandas as pd

import pandas as pd

#import the train data and convert into a dataframe

df = pd.read_csv('/Users/lorenzo/data_Mining_Proj_2020/Data_Mining_Proj_2020/train.csv', sep = ',')

count_row = df.shape[0]

print("Number of rows of the dataset " + str(count_row))

print("\n")

# <---- Preproc ------>

#Step 1) Count missing values for each attribute, and a missing rate

print("Counting missing value for each attribute")

print("\n")

print(df.isnull().sum())

rate_of_mval = (43/5719)*100

print("\n")

print("Percentage of missing value = " + str(rate_of_mval) + "%")

print("\n")
    
#step 2) How to treat missing values?

# Compute the statistics for each attribute that has missing values, the choice an appropriate strategy.

#Filling the empty values for attribute "sales at w-1". 

print("\n")

print("Statistics for attribute week-1: ")

print("\n")

print(df['sales w-1'].describe().transpose())

#Week 0 does not have a -1 week, given the high std I decided to cutoff the row

df.drop(df.index[:1], inplace=True)

print(df.head())  #TODO: should work, check it again

#N.B. in the csv file data are indexed starting from 2, but they actually start from 0


for i in range (1,5719):
    
    if pd.isna(df.loc[i , "sales w-1"]) :
        
       df.loc[i , "sales w-1"] =  df.loc[i-1 , "target"]
        
        
        
        


#print(df['volume_on_promo w-1'].describe().transpose())
#print(df['POS_exposed w-1'].describe().transpose())

   
