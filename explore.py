import matplotlib.pyplot as plt
import pickle
import os


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def plot_history(history, n, title):
    """
    Plots all information about a model given its history
    """
    plt.figure()
    plt.plot(history['loss'], label="training")
    plt.plot(history['val_loss'], label="validation")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(f"{title} loss on training and validation data (n={n})")
    plt.ylim(0, 2)
    plt.legend()
    plt.savefig(f"figures/{title}-loss")


def main():
    for filename in os.listdir("histories"):
        with open(f"histories/{filename}", "rb") as history_file:
            history = pickle.load(history_file)

        n = filename[-1]
        plot_history(history, n, filename)
        print(round(history['val_loss'][-1], 3))


if __name__ == "__main__":
    main()
