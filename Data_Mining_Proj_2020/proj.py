#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020

@author: lorenzo
"""

#import pandas as pd

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.utils import check_arrays

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





#import the train data and convert into a dataframe

#N.B. in the csv file data are indexed starting from 2, but they actually start from 0


df = pd.read_csv('/Users/lorenzo/data_Mining_Proj_2020/Data_Mining_Proj_2020/train.csv', sep = ',') 

corrmat = df.corr()

#plt.figure(figsize=(15,15))

# Setting the default font size when plotting in notebooks (used to clear previous settings)

#sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

# Plot the clustermap

#sns.clustermap(corrmat, annot=True,vmax=0.9,fmt=".2f")

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
    
    temp.target = (temp.target-temp.target.min())/(temp.target.max()-temp.target.min())
    
    temp.plot(x = 'week' , y='target', kind = 'line')
    
    for xc in xcoords:

        plt.axvline(x=xc, c = "r")
    
    plt.show()

# <---- Model ------>  why we didn't choose a clustering approach! argument

#compute pearson index for the twelve time series
    
#I created a list where I appended all the correlation coefficient
    
corr_index_list = []

corr_index_matrix  = [[0 for x in range(11)] for y in range(11)] 


for sku_1 in unique_sku:
    
    f_series = newdf[(newdf.sku == sku_1)]  #Time series of the first sku
    
    for sku_2 in unique_sku:
        
        s_series = newdf[(newdf.sku==sku_2)]  #Time series of the second sku
        
        corr_index_list.append(np.corrcoef(f_series.target,s_series.target)[0, 1])
        
#Fill a Matrix with all the coefficient in the previous list
        
count = 0
        
for x in range(11):
    
    for y in range(11):
        
        corr_index_matrix[x][y] = corr_index_list[count]
        
        count = count + 1

#print(corr_index_matrix)


#Model

del newdf['pack']
del newdf['brand']
del newdf['Unnamed: 0']


#random_state=random_seed ---> ?

# List of values to try for max_depth:
max_depth_range = list(range(1, 9))

X = newdf

y = pd.DataFrame(newdf.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# List to store the average RMSE for each value of max_depth:

accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeRegressor(max_depth = depth, 
                             random_state = 0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)

#Regression Tree

dtr_model = DecisionTreeRegressor(max_depth = 5)

dtr_model.fit(X_train, y_train)

y_preds = dtr_model.predict(X_test)

# Computing the Root Mean Square Error

rmse_tree_test = (np.sqrt(mean_squared_error(y_test, dtr_model.predict(X_test))))

# Computing R2 on the train set

r2_score_tree_test = r2_score(y_test, dtr_model.predict(X_test))

print("R2_score (test): %.3f"%r2_score_tree_test)

print("RMSE: %.3f"%rmse_tree_test)

#Random Forest

regressor_rf = RandomForestRegressor(n_estimators = 500)

regressor_rf.fit(X_train,y_train)

# Computing R2 on the train set
r2_score_rf_test = r2_score(y_test, regressor_rf.predict(X_test))

# Computing the Root Mean Square Error
rmse_rf_test = (np.sqrt(mean_squared_error(y_test, regressor_rf.predict(X_test))))

### STEP 2. Print the results

print("R2_score (test): %.3f"%r2_score_rf_test)
print("RMSE: %.3f"%rmse_rf_test)

print ("M-A-P-E") 
print(mean_absolute_percentage_error(y_test, regressor_rf.predict(X_test)))