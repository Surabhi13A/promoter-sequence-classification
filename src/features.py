import numpy as np


def prepare_classical_features(df):
    X = df["one_hot_sequence"]
    y = df["class"]

    X = X.apply(lambda seq: seq.flatten())
    X = np.stack(X)

    return X, y
