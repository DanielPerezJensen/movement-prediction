import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense


def train_test_split(X, y, test_size=0.25, random_seed=11):
    """
    Splits X, y into X_train, X_test, y_train and y_test
    """
    assert len(X) == len(y)

    # np.random.seed(random_seed)

    # shuffler = np.random.permutation(len(X))
    # X = X[shuffler]
    # y = y[shuffler]

    train_n = int((1 - test_size) * len(X))
    X_train, X_test = X[:train_n], X[train_n:]
    y_train, y_test = y[:train_n], y[train_n:]

    return X_train, X_test, y_train, y_test


def manhattan_distance(p1, p2):
    distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    return distance


def accuracy(lr, X_test, y_test):
    """
    Returns the accuracy of the linear regression model when rounding the
    predictions to better suit the coordinates our data comes from, evaluates
    the actual accuracy and the average distance the prediction was from the
    actual outcome.
    """
    predictions = lr.predict(X_test)
    predictions = np.array([[int(i) for i in n] for n in predictions])

    total_distance = 0
    correct = 0
    n = len(X_test)

    for pred, golden in zip(predictions, y_test):
        total_distance += manhattan_distance(pred, golden)

        if np.all(pred == golden):
            correct += 1

    return correct / n, total_distance / n


path_df = pd.read_csv("data/preprocessed_data.csv", sep="\t", index_col=0)

X = path_df[["x0", "y0", "x1", "y1", "x2", "y2"]].to_numpy()
y = path_df[["x5", "y5"]].to_numpy()

X = X.reshape(len(X), 3, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = Sequential()

model.add(LSTM(256, input_shape=(3, 2)))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(2))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, epochs=2000, verbose=2)
