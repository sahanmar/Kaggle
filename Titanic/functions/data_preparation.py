import pandas as pd


# Data loading
def load_data(process=False):
    # Load csv-files
    print("Loading the data...")
    train = pd.read_csv(".\\all\\train.csv", encoding="UTF-8")
    test = pd.read_csv(".\\all\\test.csv", encoding="UTF-8")
    if process:
        print("Processing the data...")
        train, test = process_data(train, test)
        # Drop unnecessary columns
        train = train.drop(labels=['Age', 'AgeCategories', 'Cabin',
                                   'Embarked', 'Fare', 'Name', 'Parch',
                                   'Sex', 'SibSp', 'Pclass', 'Ticket'],
                           axis=1)
        test = test.drop(labels=['Age', 'AgeCategories', 'Cabin',
                                 'Embarked', 'Fare', 'Name', 'Parch',
                                 'Sex', 'SibSp', 'Pclass', 'Ticket'],
                         axis=1)
    return train, test


# Data processing
def process_data(train_set: pd.DataFrame, test_set: pd.DataFrame):
    # Make age a factor variable
    # cut_points = [-1, 0, 18, 100]
    # label_names = ["Missing", "Child", "Adult"]
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult',
                   'Adult', 'Senior']
    train_set = process_age(train_set, cut_points, label_names)
    test_set = process_age(test_set, cut_points, label_names)

    # Create dummies for passenger class
    train_set = create_dummies(train_set, "Pclass")
    test_set = create_dummies(test_set, "Pclass")

    # Create dummies for gender
    train_set = create_dummies(train_set, "Sex")
    test_set = create_dummies(test_set, "Sex")

    # Create dummies for age categories
    train_set = create_dummies(train_set, "AgeCategories")
    test_set = create_dummies(test_set, "AgeCategories")
    return train_set, test_set


# Age processing
def process_age(df: pd.DataFrame,
                cut_points, label_names):

    df["Age"] = df["Age"].fillna(-0.5)
    df["AgeCategories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    return df


# Creation of dummies
def create_dummies(df: pd.DataFrame, column_name: str):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df
