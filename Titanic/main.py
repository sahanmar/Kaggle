from Titanic.functions.data_preparation import *
from Titanic.models.random_forest import *


# Load the data (both train and test sets)
train, test = load_data(process=True)

# Cross-validation
# best_params = cross_validate_randomized(train, label="Survived")
# best_params = cross_validate_grid(train, label="Survived")

params = {
    'bootstrap': True,
    'max_depth': 5,
    'max_features': 3,
    'min_samples_leaf': 3,
    'min_samples_split': 2,
    'n_estimators': 475
}

# Train the model
model = train_model(train.drop(["PassengerId"], axis=1),
                    label="Survived", params=params, test=False)

# Make predictions
submission_df = predict_and_format(model=model, test_set=test,
                                   id_colname="PassengerId",
                                   predicted_colname="Survived")

# Save the submission file
submission_df.to_csv('.\\data\\titanic_predicted.csv', index=False)
