from Titanic.functions.data_preparation import *
from Titanic.models.random_forest import *


# Load the data (both train and test sets)
train, test = load_data(process=True)

# Train the model
model = train_model(train.drop(["PassengerId"], axis=1),
                    label="Survived", test=False)

# Make predictions
submission_df = predict_and_format(model=model, test_set=test,
                                   id_colname="PassengerId",
                                   predicted_colname="Survived")

# Save the submission file
submission_df.to_csv('.\\all\\titanic_predicted.csv', index=False)
