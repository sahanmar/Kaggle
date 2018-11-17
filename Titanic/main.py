from Titanic.functions.data_preparation import *
from Titanic.models.sklearn_train import *
from Titanic.models.parameters.random_forest_parameters import *
from Titanic.models.sklearn_cv import *
from Titanic.models.parameters.random_forest_parameters import \
    random_grid, param_grid


# Load the data (both train and test sets)
var_list = ['PassengerId', 'Survived',
            'Age', 'Pclass', 'Sex', 'Fare', 'Embarked']
train, test = load_data(variables=var_list, process=True)

# Cross-validation hyperparameter search
cv = False
if cv:
    params = cross_validate_random(df=train, label="Survived",
                                   leading_col="PassengerId",
                                   estimator_name="RandomForestClassifier",
                                   random_grid=random_grid)
    # params = cross_validate_grid(df=train, label="Survived",
    #                              leading_col="PassengerId",
    #                              estimator_name="RandomForestClassifier",
    #                              param_grid=param_grid)

# Train the model
model = train_model(estimator_name="RandomForestClassifier",
                    params=params_age_normalized,
                    df=train, label="Survived", leading_col="PassengerId",
                    test=True)

# Make predictions and save the submission file
# submission_df = predict_and_format(estimator=model, test_set=test,
#                                    leading_col="PassengerId",
#                                    predicted_colname="Survived")
# submission_df.to_csv('.\\data\\titanic_predicted.csv', index=False)
