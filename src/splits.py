from sklearn.model_selection import train_test_split


def data_splitting(X, y):
    """selects features and splits the data for training and testing

    Args:
        df (pd.DataFrame): A DataFrame containing the promoter dataset.

    Returns:
        _type_: _description_
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=True, random_state=13, stratify=y
    )
    return X_train, X_test, y_train, y_test
