import pandas as pd
import numpy as np
from sklearn import preprocessing


# Data loading
def load_data(variables: list, process=False):
    # Load csv-files
    print("Loading the data...")
    train_set = pd.read_csv(".\\data\\train.csv", encoding="UTF-8")
    test_set = pd.read_csv(".\\data\\test.csv", encoding="UTF-8")

    # Data processing
    if process:
        print("Processing the data...")
        train_set, test_set = process_data(train_set, test_set, variables)

    # Filter columns
    col_filtered_train = [col for col_out in variables
                          for col in train_set if col.startswith(col_out)]
    col_filtered_test = [col for col in col_filtered_train
                         if col in list(test_set.columns.values)]

    return train_set[col_filtered_train], test_set[col_filtered_test]


# Data processing
def process_data(train_set: pd.DataFrame, test_set: pd.DataFrame,
                 variables: list):

    # Ignore the leading columns
    variables = [var for var in variables
                 if var not in ['PassengerId', 'Survived']]

    # Variables to fill and normalize
    var_normalize = []

    # Variables with special pre-processing
    if 'Age' in variables:
        var = 'Age'
        factorize = False
        if factorize:
            # Make age a factor variable
            cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
            label_names = ['Missing', 'Infant', 'Child', 'Teenager',
                           'Young Adult', 'Adult', 'Senior']
            train_set = cut_age(train_set, cut_points, label_names)
            test_set = cut_age(test_set, cut_points, label_names)
            variables.append('AgeCategories')
        else:
            var_normalize.append(var)
        variables.remove(var)

    if 'Fare' in variables:
        var = 'Fare'
        var_normalize.append(var)
        variables.remove(var)

    if 'Title' in variables:
        var = 'Title'
        # Extract passenger titles
        train_set[var] = split_titles(train_set, 'Name')
        test_set[var] = split_titles(test_set, 'Name')

    # Dummy creation
    for var in variables:
        # Create dummies for var
        train_set = create_dummies(train_set, var)
        test_set = create_dummies(test_set, var)

        # Drop the source column
        train_set = train_set.drop(labels=[var], axis=1)
        test_set = test_set.drop(labels=[var], axis=1)

    # Fill missing columns caused by dummy creation
    train_set, test_set = fill_missing_cols(train_set, test_set)

    # Filling and normalizing required columns
    train_set = fill_and_normalize(train_set, var_normalize)
    test_set = fill_and_normalize(test_set, var_normalize)

    return train_set, test_set


# Process titles
def split_titles(df: pd.DataFrame, colname: str= 'Name'):

    name_split = df[colname].str.split(', ', n=1, expand=True)
    name_split = name_split[1].str.split('.', n=1, expand=True)

    return name_split[0]


# Age processing
def cut_age(df: pd.DataFrame,
            cut_points, label_names):

    df['Age'] = df['Age'].fillna(-0.5)
    df['AgeCategories'] = pd.cut(df["Age"], cut_points, labels=label_names)

    return df


# Creation of dummies
def create_dummies(df: pd.DataFrame, column_name: str):

    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)

    return df


# Fill in missing columns in train and test data with zeros
def fill_missing_cols(train_set: pd.DataFrame, test_set: pd.DataFrame):

    train_colnames = set(train_set.columns.values)
    test_colnames = set(test_set.columns.values)
    test_colnames.add('Survived')
    test_missing = train_colnames.difference(test_colnames)
    test_missing = pd.DataFrame({title: [0 for _ in
                                         range(test_set.shape[0])]
                                 for title in test_missing})
    test_set = pd.concat([test_set, test_missing], axis=1, join="outer")
    train_missing = test_colnames.difference(train_colnames)
    train_missing = pd.DataFrame({title: [0 for _ in
                                          range(train_set.shape[0])]
                                  for title in train_missing})
    train_set = pd.concat([train_set, train_missing], axis=1, join="outer")

    return train_set, test_set


# Fill NA in the column with mean and normalize
def fill_and_normalize(df: pd.DataFrame, cols: list):

    for col in cols:
        min_max_scaler = preprocessing.MinMaxScaler()
        df[col] = min_max_scaler.fit_transform(
            np.asarray(df[col]).reshape(-1, 1)
        )
        df[col] = df[col].fillna(df[col].mean())

    return df
