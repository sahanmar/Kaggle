import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def train_model(estimator_name: str, params: dict,
                df: pd.DataFrame, label: str, leading_col: str,
                test=False):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop([label, leading_col], axis=1)
    features = np.asarray(features)

    # Split data into train and test set if required
    if test:
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, test_size=.2, random_state=123)
    else:
        train_features = features
        train_labels = labels
        test_features = []
        test_labels = []

    # Instantiate model
    if estimator_name == "RandomForestClassifier":
        estimator = RandomForestClassifier(**params)
    else:
        raise Exception("No such estimator is implemented here.")

    # Train the model
    print("Training the model...")
    estimator.fit(train_features, train_labels)

    if test:
        print("Calculating model accuracy...")
        predictions = estimator.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        print('Accuracy:', accuracy)
        return accuracy
    else:
        return estimator


def predict_and_format(estimator, test_set: pd.DataFrame,
                       leading_col: str, predicted_colname: str):
    print("Formatting the output...")
    id_col = test_set[leading_col]
    test_set = test_set.drop([leading_col], axis=1)
    predictions = estimator.predict(np.asarray(test_set))
    output = {leading_col: id_col,
              predicted_colname: predictions}
    output = pd.DataFrame(output)
    return output
