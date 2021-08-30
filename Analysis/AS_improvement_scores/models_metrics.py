from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# MSE
from sklearn.metrics import mean_squared_error
# MAE
from sklearn.metrics import mean_absolute_error
# R2
from sklearn.metrics import r2_score


def get_metrics(y_test, y_predicted):

    print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
    print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
    print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
    print("R2 score: %2f" % r2_score(y_test, y_predicted))
    print("--------------------------------------------")
    print()

def get_feature_importance(X, linearRegressionModel):

    # Calculate the importance of each feature
    features = X.columns
    importances = linearRegressionModel.coef_
    indices = np.argsort(importances)[::-1]
    top_k = 12
    new_indices = indices[:top_k]
    plt.title(str(linearRegressionModel))
    plt.bar(range(len(new_indices)), importances[new_indices], color='b', align='center')
    plt.xticks(range(len(new_indices)), [features[i] for i in new_indices], rotation=45, ha='right')
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def show_importance(X, model):

    features = X.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_k = 12
    new_indices = indices[:top_k]
    plt.title(str(model))
    plt.bar(range(len(new_indices)), importances[new_indices], color='b', align='center')
    plt.xticks(range(len(new_indices)), [features[i] for i in new_indices], rotation=45, ha='right')
    plt.ylabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def get_scatter_plot(model, y_test, y_predicted):

    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_predicted, c='crimson')
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(y_predicted), max(y_test))
    p2 = min(min(y_predicted), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title(str(model))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def train_models(X, x_train, x_test, y_train, y_test):

    # Linear Regression
    linearRegressionModel = LinearRegression()
    linearRegressionModel.fit(x_train, y_train)
    y_predicted = linearRegressionModel.predict(x_test)
    print("-------------- Linear Regression: ---------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(linearRegressionModel, y_test, y_predicted)
    get_feature_importance(X, linearRegressionModel)

    # Lasso Regression
    lassoRegressionModel = Lasso(alpha=1)
    lassoRegressionModel.fit(x_train, y_train)
    y_predicted = lassoRegressionModel.predict(x_test)
    print("-------------- Lasso Regression: ---------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(lassoRegressionModel, y_test, y_predicted)

    # Ridge Regression
    ridgeRegressionModel = Ridge()
    ridgeRegressionModel.fit(x_train, y_train)
    y_predicted = ridgeRegressionModel.predict(x_test)
    print("-------------- Ridge Regression: --------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(ridgeRegressionModel, y_test, y_predicted)

    # Support Vector Regression
    svRegressionModel = SVR(kernel="poly", max_iter=30000)
    svRegressionModel.fit(x_train, y_train)
    y_predicted = svRegressionModel.predict(x_test)
    print("----------- Support Vector Regression: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(svRegressionModel, y_test, y_predicted)

    # k-NN Regression
    kNNRegressionModel = KNeighborsRegressor()
    kNNRegressionModel.fit(x_train, y_train)
    y_predicted = kNNRegressionModel.predict(x_test)
    print("--------- k-Nearest Neighbors Regression: ---------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(kNNRegressionModel, y_test, y_predicted)

    # Decision Tree Regression
    treeRegressionModel = DecisionTreeRegressor(random_state=0)
    treeRegressionModel.fit(x_train, y_train)
    y_predicted = treeRegressionModel.predict(x_test)
    print("------------ Decision Tree Regression: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(treeRegressionModel, y_test, y_predicted)
    show_importance(X, treeRegressionModel)

    # Random Forest Regression
    randomForestModel = RandomForestRegressor(random_state=0)
    randomForestModel.fit(x_train, y_train)
    y_predicted = randomForestModel.predict(x_test)
    print("------------ Random Forest Regression: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(randomForestModel, y_test, y_predicted)
    show_importance(X, randomForestModel)

    # XGBoost Regression
    xgbreg = XGBRegressor()
    xgbreg.fit(x_train, y_train)
    y_predicted = xgbreg.predict(x_test)
    print("------------XGBoost Regression: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(xgbreg, y_test, y_predicted)
    show_importance(X, xgbreg)

    # Stacking Ensemble Machine Learning
    level0 = list()
    # define the base models
    level0.append(('Ridge', Ridge()))
    level0.append(('Lasso', Lasso(alpha=1)))
    level0.append(('SVR', SVR(kernel="poly", max_iter=30000)))
    level0.append(('KNN', KNeighborsRegressor()))
    level0.append(('DTR', DecisionTreeRegressor(random_state=0)))
    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    print("------------ Stacking Ensemble: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(model, y_test, y_predicted)

    # DummyClassifier
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_train, y_train)
    y_predicted = dummy_clf.predict(x_test)
    y_predicted_train = dummy_clf.predict(x_train)
    print("------------ Dummy Classifier: ------------")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(dummy_clf, y_test, y_predicted)