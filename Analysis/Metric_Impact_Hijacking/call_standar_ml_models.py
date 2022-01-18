from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
import numpy as np


def cross_val_with_Random_Forest(final_df, X, y):

        scaler = MinMaxScaler()
        final_df = scaler.fit_transform(final_df)
        cv = model_selection.LeaveOneOut()

        pca = PCA(n_components=150)
        pca_x = pca.fit_transform(X)

        randomForestModel = RandomForestRegressor(random_state=1, n_estimators=100)
        scores = model_selection.cross_val_score(randomForestModel, pca_x, y, scoring='neg_mean_squared_error',
                                 cv=cv, n_jobs=-1)
        print(np.sqrt(np.mean(np.absolute(scores))))


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


def ml_models(x_train, x_test, y_train, y_test):

        # Linear Regressor
        linearRegressionModel = LinearRegression()
        linearRegressionModel.fit(x_train, y_train)
        y_predicted_train = linearRegressionModel.predict(x_train)
        y_predicted = linearRegressionModel.predict(x_test)
        print("-------------- Linear Regressor: ---------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(linearRegressionModel, y_test, y_predicted)

        # Support Vector Regressor
        svRegressionModel = SVR(kernel="poly", max_iter=30000)
        svRegressionModel.fit(x_train, y_train)
        y_predicted_train = svRegressionModel.predict(x_train)
        y_predicted = svRegressionModel.predict(x_test)
        print("----------- Support Vector Regressor: ------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(svRegressionModel, y_test, y_predicted)

        # Decision Tree Regressor
        treeRegressionModel = DecisionTreeRegressor(random_state=0)
        treeRegressionModel.fit(x_train, y_train)
        y_predicted_train = treeRegressionModel.predict(x_train)
        y_predicted = treeRegressionModel.predict(x_test)
        print("------------ Decision Tree Regressor: ------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(treeRegressionModel, y_test, y_predicted)

        # XGBoost Regressor
        xgbreg = XGBRegressor(eta=0.2, max_depth=8, subsample=0.45, gamma=0.5)
        eval_set = [(x_test, y_test)]
        xgbreg.fit(x_train, y_train, early_stopping_rounds=42, eval_metric="rmse", eval_set=eval_set, verbose=True)
        y_predicted_train = xgbreg.predict(x_train)
        y_predicted = xgbreg.predict(x_test)
        print("------------XGBoost Regressor: ------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(xgbreg, y_test, y_predicted)

        # Multi-layer Perceptron Regressor
        mlpreg = MLPRegressor(hidden_layer_sizes=(200, 200), learning_rate_init=0.005, alpha=0.5, max_iter=400)
        mlpreg.fit(x_train, y_train)
        y_predicted_train = mlpreg.predict(x_train)
        y_predicted = mlpreg.predict(x_test)
        print("------------Multi-layer Perceptron Regressor: ------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(mlpreg, y_test, y_predicted)

        # Random Forest Regressor
        randomForestModel = RandomForestRegressor(random_state=1, n_estimators=150, max_depth=7)
        randomForestModel.fit(x_train, y_train)
        y_predicted_train = randomForestModel.predict(x_train)
        y_predicted = randomForestModel.predict(x_test)
        print("------------ Random Forest Regressor: ------------")
        print("RMSE: in training set %2f" % np.sqrt(mean_squared_error(y_train, y_predicted_train)))
        print("RMSE: in testing set %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
        # get_scatter_plot(randomForestModel, y_test, y_predicted)
