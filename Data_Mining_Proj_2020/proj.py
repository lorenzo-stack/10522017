#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020

@author: lorenzo
"""

#import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt


#import the train data and convert into a dataframe

#N.B. in the csv file data are indexed starting from 2, but they actually start from 0


df = pd.read_csv('/Users/lorenzo/data_Mining_Proj_2020/Data_Mining_Proj_2020/train.csv', sep = ',') 

count_row = df.shape[0]             #Number of rows of the dataset 

# <---- Preproc ------>

#Step 1) Count missing values for each attribute, and a missing rate

df.isnull().sum()    #Counting missing value for each attribute

rate_of_mval = (43/5719)*100

    
#step 2) How to treat missing values?

# Compute the statistics for each attribute that has missing values, the choice an appropriate strategy.


df.drop(df.index[:1], inplace=True)    #Week 0 does not have a -1 week, given the high std I decided to cutoff the row


promo_stats  = df['volume_on_promo w-1'].describe().transpose()

POS_exposed_stats = df['POS_exposed w-1'].describe().transpose()



for i in range (1,5719):
    
    if pd.isna(df.loc[i , "sales w-1"]) :
        
       df.loc[i , "sales w-1"] =  df.loc[i-1 , "target"]
    
    if pd.isna(df.loc[i , "volume_on_promo w-1"]):  #For the other two attribute, I replaced missing values with the mean value for that attribute
        
        df.loc[i , "volume_on_promo w-1"] =  promo_stats["mean"]
        
    if pd.isna(df.loc[i , "POS_exposed w-1"]):
        
        df.loc[i, "POS_exposed w-1"] = POS_exposed_stats["mean"]
        


for sku in df.sku.unique():
    
    newdf = df[(df.sku == sku)]
    
    newdf['week'] = newdf.index
    
    newdf.plot(x = 'week' , y='target', kind = 'line')
    
    #xcoords = [185, 237]

    #for xc in xcoords:

    #    plt.axvline(x=xc, c = "r")
    
    #if newdf.scope == 1:
    
    print("Series of the --> " + str(sku) + " sku")
    

    plt.show()





   
