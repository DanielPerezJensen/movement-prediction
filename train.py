import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense
from keras import metrics
from keras import backend as K
import time


def train_test_split(X, y, test_size=0.25, random_seed=11):
    """
    Splits X, y into X_train, X_test, y_train and y_test
    """
    assert len(X) == len(y)

    np.random.seed(random_seed)

    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    y = y[shuffler]

    train_n = int((1 - test_size) * len(X))
    X_train, X_test = X[:train_n], X[train_n:]
    y_train, y_test = y[:train_n], y[train_n:]

    return X_train, X_test, y_train, y_test


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def create_model(*layers):
    """Creates a model of unspecified amount of layers given *layers"""
    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.add(Dense(2))
    return model


def create_nn_models():
    """
    Returns list of models we want to fit and evaluate
    """
    models = []
    model_names = []

    models.append(create_model(LSTM(256, input_shape=(3, 2)),
                               Dense(256, activation='tanh'),
                               Dense(128, activation='tanh')))
    model_names.append("reference")

    # models.append(create_model(LSTM(256, input_shape=(3, 2)),
    #                            Dense(256, activation='linear'),
    #                            Dense(128, activation='tanh')))
    # model_names.append("t1")

    return models, model_names


def main():

    # Create matrix to store loss in
    performance_matrix = np.zeros(shape=(4, 5))

    # Gather list of models
    models, model_names = create_nn_models()

    # Train and evaluate all neural networks
    for i, (m, mn) in enumerate(zip(models, model_names)):

        for n in [1, 2, 3, 4, 5]:

            # gather and preprocess data
            path_df = pd.read_csv(f"data/preprocessed_data-n={n}.csv",
                                  sep="\t", index_col=0)

            X = path_df[["x0", "y0", "x1", "y1", "x2", "y2"]].to_numpy()
            y = path_df[["x3", "y3"]].to_numpy()

            X = X.reshape(len(X), 3, 2)

            X_train, X_test, y_train, y_test = train_test_split(X, y)

            # Create neural model, train and save
            m.compile(loss=root_mean_squared_error, optimizer="rmsprop")
            history = m.fit(X_train, y_train, batch_size=32,
                            epochs=200, verbose=2,
                            validation_data=(X_test, y_test))

            m.save(f"models/{mn}-n={n}")

            # Evaluate performance of each model and store in array
            val_loss = history.history['val_loss'][-1]
            performance_matrix[i][n - 1] = val_loss

            with open(f"histories/{mn}-n={n}", "wb") as f:
                pickle.dump(history.history, f)

    # Train and evaluate all linear regression model
    for n in [1, 2, 3, 4, 5]:
        path_df = pd.read_csv(f"data/preprocessed_data-n={n}.csv",
                              sep="\t", index_col=0)

        X = path_df[["x0", "y0", "x1", "y1", "x2", "y2"]].to_numpy()
        y = path_df[["x3", "y3"]].to_numpy()

        # gather and preprocess data
        path_df = pd.read_csv(f"data/preprocessed_data-n={n}.csv",
                              sep="\t", index_col=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        linear_model = LinearRegression().fit(X_train, y_train)

        polynomial_features = PolynomialFeatures(degree=2)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.fit_transform(X_test)
        polyn_model = LinearRegression().fit(X_train_poly, y_train)

        y_lin_pred = linear_model.predict(X_test)
        y_poly_pred = polyn_model.predict(X_test_poly)

        rmse_lin = mean_squared_error(y_test, y_lin_pred, squared=False)
        rmse_poly = mean_squared_error(y_test, y_poly_pred, squared=False)

        # Store rmse of linear models
        performance_matrix[2][n - 1] = rmse_lin
        performance_matrix[3][n - 1] = rmse_poly

    print(performance_matrix)


if __name__ == "__main__":
    main()
