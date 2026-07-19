import pandas as pd
import numpy as np
import pytest
from src.data import load_data, preprocess_data, one_hot_encode


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "class" in df.columns
    assert "identifier" in df.columns
    assert "sequence" in df.columns


def test_one_hot_encode_known_bases():
    seq = "acgt"
    encoded_seq = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    result = one_hot_encode(seq)
    print(type(result))
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(seq), 4)
    assert np.array_equal(result, np.array(encoded_seq))


def test_one_hot_encode_unexpected_nucleotide():
    seq = "acgty"
    with pytest.raises(ValueError, match=r".*y.*"):
        one_hot_encode(seq)


def test_one_hot_encode_valid():
    seq = load_data()["sequence"].iloc[0]
    result = one_hot_encode(seq)
    assert isinstance(result, np.ndarray)
    assert result.shape == (57, 4)


def test_preprocess_data():
    df = load_data()
    preprocessed_df = preprocess_data(df)
    assert isinstance(preprocessed_df, pd.DataFrame)
    assert "one_hot_sequence" in preprocessed_df.columns
    assert isinstance(preprocessed_df["one_hot_sequence"].iloc[0], np.ndarray)
    assert "class" in preprocessed_df.columns
    assert "identifier" in preprocessed_df.columns
    assert "sequence" in preprocessed_df.columns
    assert preprocessed_df.shape[0] == df.shape[0]
