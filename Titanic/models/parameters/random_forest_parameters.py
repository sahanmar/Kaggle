import numpy as np

"""
    Parameter grids for cross-validation.
"""

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
# Random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Parameter grid based on the results of random search
param_grid = {
    'bootstrap': [False],
    'max_depth': [8, 10, 12],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [5, 10],
    'n_estimators': [int(x) for x in np.linspace(start=30, stop=90,
                                                 num=10)]
}


"""
    Accuracies were calculated for the test set which
    was generated as 20% of the train set.
"""

# Accuracy: 0.8603351955307262
params_age_factorized_title = {
    'bootstrap': True,
    'max_depth': 12,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 83
}

# Accuracy: 0.8491620111731844
params_age_normalized_title = {
    'bootstrap': False,
    'max_depth': 10,
    'max_features': 'auto',
    'min_samples_leaf': 2,
    'min_samples_split': 10,
    'n_estimators': 120
}

# Accuracy: 0.8659217877094972 (~ 77% on Kaggle)
params_age_normalized = {
    'bootstrap': False,
    'max_depth': 10,
    'max_features': 'sqrt',
    'min_samples_leaf': 4,
    'min_samples_split': 5,
    'n_estimators': 120
}

# Accuracy: 0.8547486033519553
params_age_factorized = {
    'bootstrap': True,
    'max_depth': 50,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 670
}
