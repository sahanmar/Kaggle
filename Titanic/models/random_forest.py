import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(df: pd.DataFrame, label: str, test=False):

    # Separate data into labels and features
    labels = np.asarray(df[label])
    features = df.drop(label, axis=1)
    features = np.asarray(features)

    # Split data into train and test set if required
    if test:
        train_features, test_features, train_labels, test_labels = \
            train_test_split(features, labels, test_size=0.25, random_state=123)
    else:
        train_features = features
        train_labels = labels
        test_features = []
        test_labels = []

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=10000, random_state=123,
                                warm_start=True, max_depth=25)

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
