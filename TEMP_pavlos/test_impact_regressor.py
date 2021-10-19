#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import pandas as pd
# import json
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# FILENAME = 'Impact_&_features.csv'
FILENAME = 'Impact_&_features_with_legitimate_hijacker.csv'

df = pd.read_csv(FILENAME, sep=",", index_col=0)
y = df['impact']
extra_columns = [c for c in df.columns if c.startswith('Continent') or c.startswith('number')] 
X = df.drop(['impact'],axis=1)
# X = df.iloc[:,list(range(1,51)) + list(range(799,830))]
X1 = df.iloc[:,1:51]
X2 = df.iloc[:,801:]
X3 = df.iloc[:,51:801]
X4 = pd.concat([X1,X2], axis=1)
X5 = pd.concat([X1,X3], axis=1)

X=X4

# print(X4)

scaler = MinMaxScaler() 
# scaler = StandardScaler()
X = scaler.fit_transform(X)


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)

# pca = PCA(n_components=200)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)


nn_w = 64
nn_d = 2
layers = tuple([nn_w]*nn_d)
model = MLPRegressor(hidden_layer_sizes=layers, learning_rate_init=0.001, solver='adam', max_iter=200, alpha=0.01)


model.fit(x_train,y_train)

y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)


print("RMSE (train) %.3f" % np.sqrt(mean_squared_error(y_train, y_train_predict)))
print("RMSE (test)  %.3f" % np.sqrt(mean_squared_error(y_test, y_test_predict)))