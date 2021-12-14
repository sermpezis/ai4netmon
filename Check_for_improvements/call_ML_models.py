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

list1 = []
list2 = []
list3 = []


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


def cal_metrics(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2 = r2_score(y_test, y_predicted)

    return mse, mae, rmse, r2


def get_metrics(y_test, y_predicted):
    print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
    print("Mean Absolute Error: %2f" % mean_absolute_error(y_test, y_predicted))
    print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
    print("R2 score: %2f" % r2_score(y_test, y_predicted))
    print("--------------------------")
    print()


def get_scatter_plot(model, y_test, y_predicted):
    """
    :param model: The machine learning algorithm
    :param y_test: The true/real y label
    :param y_predicted: The y label that our model predict
    :return: Plots showing the performance of our Machine Learning model
    """
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


def clear_lists():
    list1.clear()
    list2.clear()
    list3.clear()


def call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca):
    """
    :param x_train: The features that are given as input to our model for training.
    :param x_test: The features that will be given to our machine learning in order to predict their labels.
    :param y_train: The given features label.
    :param y_test: The y label that our model predicts.
    :param x_train_pca: The features that are given as input to our model for training. The features have been
    transformed using PCA.
    :param x_test_pca: The features that will be given to our machine learning in order to predict their labels.
    The features have been transformed using PCA.
    :param y_train_pca: The given features label.The features have been transformed using PCA.
    :param y_test_pca: The y label that our model predicts. The features have been transformed using PCA.
    """
    svRegressionModel = SVR(kernel="rbf")
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
    dummy_reg.fit(x_train, y_train)
    y_predicted = dummy_reg.predict(x_test)
    print("Dummy Regression: ")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(dummy_reg, y_test, y_predicted)

    mlp_reg = MLPRegressor(hidden_layer_sizes=2048, activation="relu", solver='sgd', random_state=1, max_iter=750)
    mlp_reg.fit(x_train_pca, y_train_pca)
    y_predicted = mlp_reg.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("MLP Regression: ")
    get_metrics(y_test_without_log, y_predicted_without_log)
    get_scatter_plot(mlp_reg, y_test_without_log, y_predicted_without_log)

    mlp_reg = MLPRegressor(hidden_layer_sizes=2048, activation="relu", solver='sgd', random_state=1, max_iter=512)
    mlp_reg.fit(x_train, y_train)
    y_predicted = mlp_reg.predict(x_test)
    print("MLP Regression: ")
    get_metrics(y_test, y_predicted)
    get_scatter_plot(mlp_reg, y_test, y_predicted)


def call_methods_without_log(x_train, x_test, y_train, y_test):
    """
    :param x_train: The features that are given as input to our model for training.
    :param x_test: The features that will be given to our machine learning in order to predict their labels.
    :param y_train: The given features label.
    :param y_test: The y label that our model predicts.
    """
    svRegressionModel = SVR(kernel="poly")
    svRegressionModel.fit(x_train, y_train)
    y_predicted = svRegressionModel.predict(x_test)
    print("Support Vector Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test, y_predicted)
    list1.append([mse, mae, rmse, r2])

    dummy_reg = DummyRegressor(strategy="median")
    dummy_reg.fit(x_train, y_train)
    y_predicted = dummy_reg.predict(x_test)
    print("Dummy Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test, y_predicted)
    list2.append([mse, mae, rmse, r2])

    mlp_reg = MLPRegressor(hidden_layer_sizes=2048, activation="relu", solver='sgd', random_state=1, max_iter=512)
    mlp_reg.fit(x_train, y_train)
    y_predicted = mlp_reg.predict(x_test)
    print("MLP Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test, y_predicted)
    list3.append([mse, mae, rmse, r2])


def get_lists_containing_metrics():
    return [list1, list2, list3]


def call_methods_with_log(x_train_pca, x_test_pca, y_train_pca, y_test_pca):
    """
    :param x_train_pca: The features that are given as input to our model for training. The features have been
    transformed using PCA.
    :param x_test_pca: The features that will be given to our machine learning in order to predict their labels.
    The features have been transformed using PCA.
    :param y_train_pca: The given features label.The features have been transformed using PCA.
    :param y_test_pca: The y label that our model predicts. The features have been transformed using PCA.
    """
    randomForestModel = RandomForestRegressor(random_state=0)
    randomForestModel.fit(x_train_pca, y_train_pca)
    y_predicted = randomForestModel.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("Random Forest Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test_without_log, y_predicted_without_log)
    list1.append([mse, mae, rmse, r2])

    reg = GradientBoostingRegressor()
    reg.fit(x_train_pca, y_train_pca)
    y_predicted = reg.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("GradientBoosting Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test_without_log, y_predicted_without_log)
    list2.append([mse, mae, rmse, r2])

    mlp_reg = MLPRegressor(hidden_layer_sizes=2048, activation="relu", solver='sgd', random_state=1, max_iter=750)
    mlp_reg.fit(x_train_pca, y_train_pca)
    y_predicted = mlp_reg.predict(x_test_pca)
    y_test_without_log, y_predicted_without_log = get_y_without_log(y_test_pca, y_predicted)
    print("MLP Regression: ")
    mse, mae, rmse, r2 = cal_metrics(y_test_without_log, y_predicted_without_log)
    list3.append([mse, mae, rmse, r2])
