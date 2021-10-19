import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

FILENAME = 'Impact_&_features_with_legitimate_hijacker.csv'

df = pd.read_csv(FILENAME, sep=",", index_col=0)
y = df['impact']
X = df.drop(['impact'],axis=1)

scaler = MinMaxScaler() 
X = scaler.fit_transform(X)


x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)


nn_w = 64
nn_d = 2
layers = tuple([nn_w]*nn_d)
model = MLPRegressor(hidden_layer_sizes=layers, learning_rate_init=0.001, solver='adam', max_iter=200, alpha=0.01)
model.fit(x_train,y_train)

y_train_predict = model.predict(x_train)
y_test_predict = model.predict(x_test)


print("RMSE (train) %.3f" % np.sqrt(mean_squared_error(y_train, y_train_predict)))
print("RMSE (test)  %.3f" % np.sqrt(mean_squared_error(y_test, y_test_predict)))