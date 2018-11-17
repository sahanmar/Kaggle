import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


def cross_validate_random(estimator_name: str, random_grid: dict,
                          df: pd.DataFrame,
                          label: str, leading_col: str):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop([label, leading_col], axis=1)
    features = np.asarray(features)

    # Instantiate model
    if estimator_name == "RandomForestClassifier":
        estimator = RandomForestClassifier()
    else:
        raise Exception("No such estimator is implemented here.")

    # Random search of parameters, using 3-fold cross-validation
    random_search = RandomizedSearchCV(estimator=estimator,
                                       param_distributions=random_grid,
                                       n_iter=100, cv=3, verbose=2,
                                       random_state=123, n_jobs=-1)
    # Fit the random search model
    random_search.fit(features, labels)

    return random_search.best_params_


def cross_validate_grid(estimator_name: str, param_grid: dict,
                        df: pd.DataFrame,
                        label: str, leading_col: str):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop([label, leading_col], axis=1)
    features = np.asarray(features)

    # Instantiate model
    if estimator_name == "RandomForestClassifier":
        estimator = RandomForestClassifier()
    else:
        raise Exception("No such estimator is implemented here.")

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(features, labels)

    return grid_search.best_params_
