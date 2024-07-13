#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:52:34 2020

@author: yuchenchen
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

from xgboost import XGBRegressor
original_data = pd.read_csv( 'train.csv') #讀取訓練資料
test_data = pd.read_csv( 'test.csv')  #讀取測試資料

original_data_y = original_data.SalePrice #獲取y
print (original_data_y)
original_data = original_data.drop(['SalePrice'], axis=1) #刪除y

X_train = original_data
X_test = test_data

choose_column = [col for col in X_train.columns if (X_train[col].nunique() < 10 and X_train[col].dtype == "object") or X_train[col].dtype in ['int64','float64']]
X_train = X_train[choose_column]
X_test = X_test[choose_column]

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join = 'left', axis=1)

#my_imputer = Imputer()
train_X = imp.fit_transform(X_train)
test_X = imp.transform(X_test)

my_model = XGBRegressor(learning_rate=0.01, 
                        n_estimators=3460,
                        max_depth=3, 
                        min_child_weight=0,
                        gamma=0, 
                        subsample=0.7,
                        colsample_bytree=0.7,
                        objective='reg:squarederror', 
                        nthread=-1,
                        scale_pos_weight=1, 
                        seed=27,
                        reg_alpha=0.00006)

my_model.fit(train_X, original_data_y, verbose=False)
pre_test_y = my_model.predict(test_X)
print (pre_test_y)
print (my_model)
my_submission = pd.DataFrame({'Id':X_test.Id, 'SalePrice':pre_test_y})
my_submission.to_csv('submission14.csv', index=False) #0.14250