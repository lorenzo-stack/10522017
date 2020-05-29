#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020

@author: lorenzo
"""

#import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy





#import the train data and convert into a dataframe

#N.B. in the csv file data are indexed starting from 2, but they actually start from 0


df = pd.read_csv('/Users/lorenzo/data_Mining_Proj_2020/Data_Mining_Proj_2020/train.csv', sep = ',') 

corrmat = df.corr()
plt.figure(figsize=(15,15))

# Setting the default font size when plotting in notebooks (used to clear previous settings)
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

# Plot the clustermap
sns.clustermap(corrmat, annot=True,vmax=0.9,fmt=".2f")

count_row = df.shape[0]             #Number of rows of the dataset 

# <---- Preproc ------>

#Step 1) Count missing values for each attribute, and compute a missing rate

print("New print " + str(df.isnull().sum() ))   #Counting missing value for each attribute

rate_of_mval = (43/5719)*100     #Very low

    
#Step 2) How to treat missing values?

# Compute the statistics for each attribute that has missing values,choice an appropriate strategy.


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
        


newdf = df[(df.scope== 1)]    #Only the sku for which the final evaluation will be done 

unique_sku = newdf.sku.unique()

period_in_weeks = list(range(0, 133)) # every sku has tha same number of weekly data

xcoords = [52, 104] # one year = 52 weeks


for sku in unique_sku:
    
    print("Series of the product with sku =  " + str(sku))
    
    temp = newdf[(df.sku == sku)]
    
    temp['week'] = period_in_weeks 
    
    temp.plot(x = 'week' , y='target', kind = 'line')
    
    
    for xc in xcoords:

        plt.axvline(x=xc, c = "r")
    
    plt.show()

# <---- Model ------>  why we didn't choose a clustering approach! argument

#compute pearson index for the twelve time series
    
corr_index_list = []
w, h = 12, 12;
corr_index_matrix  = [[0 for x in range(w)] for y in range(h)] 


for sku_1 in unique_sku:
    
    f_series = newdf[(newdf.sku == sku_1)]
    
    for sku_2 in unique_sku:
        
        s_series = newdf[(newdf.sku==sku_2)]
        
        corr_index_list.append(numpy.corrcoef(f_series.target,s_series.target)[0, 1])
        
count = 1
        
for x in range(11):
    
    for y in range(11):
        
        corr_index_matrix[x][y] = corr_index_list[count]
        
        count = count + 1

print(len(corr_index_matrix))
