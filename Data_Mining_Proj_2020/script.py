#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:30:27 2020
Variable description

SKU = Unique identifier for the products (int)
Pack= Type of pack in which the product is sold (str)
Size (GM) = Product weight float Product brand (str)
Brand = Product brand (str)
Price = Planned price of sale for the product in week w (float)
POS_exposed w-1 = Number of stores in which the product was put on evidence at w-1 (int)
Volume_on_promo w-1  = % Volume of product put on promo at w-1 (float)
Sales w-1 = Sales of product at w-1 (lagged target) (int)
Scope = Boolean that indicates SKUs in scope (target) (bool)
Target = Sales of product in w int

int
@author: lorenzo
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true[:, 0] - y_pred) / y_true[:, 0]))


df = pd.read_csv('train.csv', sep = ',') 


corrmat = df.corr()

plt.figure(figsize=(15,15))

# Plot the clustermap

sns.clustermap(corrmat, annot=True,vmax=0.9,fmt=".2f")

count_row = df.shape[0]             #Number of rows of the dataset 

#Step 1) Count missing values for each attribute, and compute a missing rate

mval = df.isnull().sum() #Counting missing value for each attribute

rate_of_mval = (mval/5719)*100

    
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

# <---- Model ------>  
    
#I created a list where I appended all the correlation coefficient
    
corr_index_list = []


for sku_1 in unique_sku:
    
    f_series = newdf[(newdf.sku == sku_1)]  #Time series of the first sku
    
    for sku_2 in unique_sku:
        
        s_series = newdf[(newdf.sku==sku_2)]  #Time series of the second sku
        
        corr_index_list.append(np.corrcoef(f_series.target,s_series.target)[0, 1])
        



#Model

del newdf['pack']
del newdf['brand']
del newdf['Unnamed: 0']

# List of values to try for max_depth:
max_depth_range = list(range(1, 13))

X = pd.DataFrame(newdf)
y = pd.DataFrame(newdf.target)
del X['target']

random_seed=2398745

# 10-fold crossvalidation
tenfold_xval = KFold(10, shuffle=True, random_state=random_seed)


X_train, X_test, y_train, y_test = train_test_split(X, y)

# List to store the average RMSE for each value of max_depth:

accuracy = []


for depth in max_depth_range:
    
    clf = DecisionTreeRegressor(criterion='mae', splitter='random',  max_depth = depth)
    
    cv_tree = cross_val_score(clf, X_train, y_train, cv=tenfold_xval, scoring='r2')
    
    s_i = np.mean(cv_tree)    
    
    accuracy.append(s_i)

acct = np.array(accuracy)

idxt = np.where(acct == max(acct))

nselt = max_depth_range[int(idxt[0])]

clf = DecisionTreeRegressor(criterion='mae', splitter='random',  max_depth = int(nselt))

clf.fit(X_train,y_train.values.ravel())

print ("M-A-P-E") 
print(mean_absolute_percentage_error(y_test, clf.predict(X_test)))

#Random Forest

nes = np.array([1, 3, 4, 7, 9, 12, 21, 29, 39, 49, 59,  70, 150, 225, 300])
s0f = 0
accuracyf = []
for n in nes:

    regressor_rf = RandomForestRegressor(n_estimators = n)
    
    cv_rf3 = cross_val_score(regressor_rf, X_train, y_train.values.ravel(), cv=tenfold_xval, scoring='r2')

    s_if = np.mean(cv_rf3)
       
    accuracyf.append(s_if)
    
    
acc = np.array(accuracyf)

idx = np.where(acc == max(acc))

nsel = nes[idx]

regressor_rf = RandomForestRegressor(n_estimators = int(nsel))

regressor_rf.fit(X_train,y_train.values.ravel())



# Computing R2 on the test set
r2_score_rf_test = r2_score(y_test, regressor_rf.predict(X_test))

#Scatter plot between observed & predicted values

pt = regressor_rf.predict(X_test)

mx = max(max(np.array(y_test)),max( pt) )
plt.scatter(y_test, regressor_rf.predict(X_test))
plt.xlim(0, mx)
plt.ylim(0, mx)

### STEP 2. Print the results, creation of the submission file

print("R2_score (test): %.3f"%r2_score_rf_test)

print ("M-A-P-E") 
print(mean_absolute_percentage_error(y_test, regressor_rf.predict(X_test)))


df_test = pd.read_csv('x_test.csv', sep = ',') 


del df_test['pack']
del df_test['brand']

step = df_test['Unnamed: 0']

del df_test['Unnamed: 0']



X_test_r = df_test

df_test["prediction"] = regressor_rf.predict(X_test_r)

del df_test['size (GM)']
del df_test['price']
del df_test['volume_on_promo w-1']
del df_test['sales w-1']
del df_test['scope']
del df_test["POS_exposed w-1"]

df_test['Unnamed: 0'] = step

df_test.set_index(['Unnamed: 0'])

df_test.to_csv("Prediction.csv", index = False, columns = ["Unnamed: 0","sku","prediction"])

for sku in unique_sku:
    
    print("Series of the product with sku =  " + str(sku))
    
    temp = df_test[(df_test.sku == sku)]
    
    temp.prediction = (temp.prediction-temp.prediction.min())/(temp.prediction.max()-temp.prediction.min())
    
    temp['week'] = np.arange(0, temp.shape[0])
    temp.plot(x = 'week' , y='prediction', kind = 'line')
    
    for xc in xcoords:

        plt.axvline(x=xc, c = "r")
    
    plt.show()
    
