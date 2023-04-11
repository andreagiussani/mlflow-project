from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(filepath: str) -> Union[pd.DataFrame]:
    diabetes = pd.read_csv(filepath)

    X = diabetes.drop(['diabetes'], axis=1)
    y = diabetes.diabetes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
