import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import model_selection
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler

# MSE
from sklearn.metrics import mean_squared_error
# MAE
from sklearn.metrics import mean_absolute_error
# R2
from sklearn.metrics import r2_score

data = pd.read_csv("../../Datasets/improvements20210601.txt", sep=" ")

# checking data before we start preprocessing
print(data.shape)
print(data.dtypes)
print("------------------------------")

# Give name to columns
data.columns = ['location', 'IPV4-6', 'AS_number', 'improvement_score']

# keep only GLOBAL and IPV-4 examples
new_data = data.loc[(data["location"] == "GLOBAL") & (data["IPV4-6"] == 4)]

new_data = new_data[new_data[['AS_number']].apply(lambda x: x[0].isdigit(), axis=1)]
print(new_data.shape)

# Checking the missing values
print("-- Check for missing values --")
print(new_data.isnull().sum())
print("------------------------------")

# new_data.location.value_counts(normalize=True)
# new_data.location.value_counts(normalize=True).plot.barh()
# plt.show()

print("Analyze improvement-score")
print(new_data.improvement_score.describe())
print("------------------------------")

print("Analyze AS_number")
print(new_data.AS_number.describe())
print("------------------------------")

# needed to convert to a string first, then an float.
new_data['AS_number'] = new_data['AS_number'].astype(str).astype(float)
print(new_data.dtypes)

#plot the scatter plot of AS_number and improvement_score variable in data
# plt.scatter(data.AS_number, data.improvement_score)
# plt.show()


#Correlation Matrix
new_data[['AS_number', 'improvement_score']].corr()
sns.heatmap(new_data[['AS_number', 'improvement_score']].corr(), annot=True, cmap='Blues')
plt.show()

print("-------- Shapiro Test --------")
df = new_data[['AS_number', 'improvement_score']]
stat, p = stats.shapiro(df)
print('Statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
print("------------------------------")

#Correlation Tests
print("---- Pearson’s Correlation ----")
data1 = new_data['AS_number']
data2 = new_data['improvement_score']
statistics, p1 = stats.pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (statistics, p1))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
print("------------------------------")

#Spearman’s Rank Correlation
print("--- Spearman’s Correlation ---")
stat2, p2 = stats.spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat2, p2))
if p2 > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
print("------------------------------")


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
#
# # Linear Regression
# linearRegressionModel = LinearRegression()
# linearRegressionModel.fit(x_train, y_train)
# y_predicted = linearRegressionModel.predict(x_test)
# print("-------------- Linear Regression: ---------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("-------------------------------------------------")
# print()
#
# # Lasso Regression
# lassoRegressionModel = Lasso(alpha=1)
# lassoRegressionModel.fit(x_train, y_train)
# y_predicted = lassoRegressionModel.predict(x_test)
# print("-------------- Lasso Regression: ---------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("------------------------------------------------")
# print()
#
# # Ridge Regression
# ridgeRegressionModel = Ridge()
# ridgeRegressionModel.fit(x_train, y_train)
# y_predicted = ridgeRegressionModel.predict(x_test)
# print("-------------- Ridge Regression: --------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("------------------------------------------------")
# print()
#
# # Support Vector Regression
# svRegressionModel = SVR(kernel="poly", max_iter=30000)
# svRegressionModel.fit(x_train, y_train)
# y_predicted = svRegressionModel.predict(x_test)
# print("----------- Support Vector Regression: ------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("---------------------------------------------------")
# print()
#
# # k-NN Regression
# kNNRegressionModel = KNeighborsRegressor()
# kNNRegressionModel.fit(x_train, y_train)
# y_predicted = kNNRegressionModel.predict(x_test)
# print("--------- k-Nearest Neighbors Regression: ---------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("---------------------------------------------------")
# print()
#
# # Decision Tree Regression
# treeRegressionModel = DecisionTreeRegressor(random_state=0)
# treeRegressionModel.fit(x_train, y_train)
# y_predicted = treeRegressionModel.predict(x_test)
# print("------------ Decision Tree Regression: ------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("---------------------------------------------------")
# print()
#
# # Stacking Ensemble Machine Learning
# level0 = list()
# # define the base models
# level0.append(('Ridge', Ridge()))
# level0.append(('Lasso', Lasso(alpha=1)))
# level0.append(('SVR', SVR(kernel="poly", max_iter=30000)))
# level0.append(('KNN', KNeighborsRegressor()))
# level0.append(('DTR', DecisionTreeRegressor(random_state=0)))
# # define meta learner model
# level1 = LinearRegression()
# # define the stacking ensemble
# model = StackingRegressor(estimators=level0, final_estimator=level1)
# model.fit(x_train, y_train)
# y_predicted = model.predict(x_test)
#
# print("------------ Stacking Ensemble: ------------")
# print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
# print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
# print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
# print("R2 score: %2f" % r2_score(y_test, y_predicted))
# print("---------------------------------------------------")
# print()
