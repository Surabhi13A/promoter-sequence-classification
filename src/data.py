import pandas as pd
import numpy as np
import pathlib

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "raw" / "promoters.data"

MAPPING = {
    "a": [1, 0, 0, 0],
    "c": [0, 1, 0, 0],
    "g": [0, 0, 1, 0],
    "t": [0, 0, 0, 1],
}


def load_data():
    """
    Load the promoter dataset from the UCI Machine Learning Repository.

    Returns:
        pd.DataFrame: A DataFrame containing the promoter dataset.
    """
    column_names = ["class", "identifier", "sequence"]
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)
    df["sequence"] = df["sequence"].str.strip()
    return df


def one_hot_encode(seq):
    """One-hot encode a DNA sequence.

    Args:
        seq (string): A string representing a DNA sequence.

    Returns:
        np.ndarray: A numpy array representing the one-hot encoded sequence.
    """
    base = seq.lower()
    unexpected_bases = set(base) - set(MAPPING.keys())
    if unexpected_bases:
        raise ValueError(f"Unexpected bases found: {unexpected_bases}")
    return np.array([MAPPING[base] for base in seq.lower()])


def preprocess_data(df):
    """
    Preprocess the promoter dataset by one-hot encoding the DNA sequences.

    Args:
        df (pd.DataFrame): A DataFrame containing the promoter dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed promoter dataset.
    """

    df["one_hot_sequence"] = df["sequence"].apply(one_hot_encode)
    return df
