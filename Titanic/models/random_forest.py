import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


def cross_validate_randomized(df: pd.DataFrame, label: str):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop(label, axis=1)
    features = np.asarray(features)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # noinspection PyTypeChecker
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2,
                                   random_state=123, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(features, labels)

    return rf_random.best_params_


def cross_validate_grid(df: pd.DataFrame, label: str):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop(label, axis=1)
    features = np.asarray(features)

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [True],
        'max_depth': [i for i in range(5, 12, 1)],
        'max_features': [2, 3],
        'min_samples_leaf': [i for i in range(3, 7, 1)],
        'min_samples_split': [2, 3, 4],
        'n_estimators': [int(x) for x in np.linspace(start=475, stop=625,
                                                     num=20)]
    }

    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(features, labels)

    return grid_search.best_params_


def train_model(df: pd.DataFrame, label: str, params: dict, test=False):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop(label, axis=1)
    features = np.asarray(features)

    # Split data into train and test set if required
    if test:
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, test_size=0.2, random_state=123)
    else:
        train_features = features
        train_labels = labels
        test_features = []
        test_labels = []

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(**params)

    # Train the model
    print("Training the random forest model...")
    rf.fit(train_features, train_labels)

    if test:
        print("Calculating model accuracy...")
        predictions = rf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print('Accuracy:', accuracy)
        return accuracy
    else:
        return rf


def predict_and_format(model: RandomForestClassifier,
                       test_set: pd.DataFrame,
                       id_colname: str, predicted_colname: str):
    print("Formatting the output...")
    id_col = test_set[id_colname]
    test_set = test_set.drop([id_colname], axis=1)
    predictions = model.predict(np.asarray(test_set))
    output = {id_colname: id_col,
              predicted_colname: predictions}
    output = pd.DataFrame(output)
    return output
