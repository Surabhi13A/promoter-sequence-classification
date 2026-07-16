import pandas as pd
import numpy as np


def load_data():
    """
    Load the promoter dataset from the UCI Machine Learning Repository.

    Returns:
        pd.DataFrame: A DataFrame containing the promoter dataset.
    """
    column_names = ["class", "identifier", "sequence"]
    df = pd.read_csv("../data/raw/promoters.data", header=None, names=column_names)
    df["sequence"] = df["sequence"].str.strip()
    return df
