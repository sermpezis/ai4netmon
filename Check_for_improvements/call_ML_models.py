from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np


def get_y_without_log(y_test, y_predicted):
    y_pred_without_log = []
    for i in y_predicted:
        x = np.exp(i)
        y_pred_without_log.append(x)
    y_predicted_new = np.array(y_pred_without_log)

    y_test_without_log = []
    for i in y_test:
        x = np.exp(i)
        y_test_without_log.append(x)
    y_test_new = np.array(y_test_without_log)

    return y_test_new, y_predicted_new


def get_metrics(y_test, y_predicted):
    print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
    print("Mean Absolute Error: %2f" % mean_absolute_error(abs(y_test), abs(y_predicted)))
    print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
    print("R2 score: %2f" % r2_score(y_test, y_predicted))
    print("--------------------------")
    print()


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


def call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca):
    svRegressionModel = SVR(kernel="poly", max_iter=30000)
    svRegressionModel.fit(x_train, y_train)
    y_predicted = svRegressionModel.predict(x_test)
    print("Support Vector Regression: ")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(svRegressionModel, y_test, y_predicted)

    randomForestModel = RandomForestRegressor(random_state=0)
    randomForestModel.fit(x_train_pca, y_train_pca)
    y_predicted = randomForestModel.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("Random Forest Regression: ")
    get_metrics(y_test_without_log, y_predicted_without_log)
    get_scatter_plot(randomForestModel, y_test_without_log, y_predicted_without_log)

    reg = GradientBoostingRegressor()
    reg.fit(x_train_pca, y_train_pca)
    y_predicted = reg.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("GradientBoosting Regression: ")
    get_metrics(y_test_without_log, y_predicted_without_log)
    get_scatter_plot(reg, y_test_without_log, y_predicted_without_log)

    dummy_reg = DummyRegressor(strategy="median")
    dummy_reg.fit(x_train_pca, y_train_pca)
    y_predicted = dummy_reg.predict(x_test_pca)
    print("Dummy Regression: ")
    get_metrics(y_test_pca, y_predicted)

    mlp_reg = MLPRegressor(hidden_layer_sizes=2048, activation="relu", random_state=1, max_iter=512)
    mlp_reg.fit(x_train, y_train)
    y_predicted = mlp_reg.predict(x_test)
    print("MLP Regression: ")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(mlp_reg, y_test, y_predicted)
