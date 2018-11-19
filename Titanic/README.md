## Data cleaning

| Task        | Status           | Path  |
| ------------- |:-------------:| -----:|
| Get 'Title' from the names and group some values together | **DONE** | *dev-sahanmar -> clean_data.py* |
| Get titles from the names and create dummies for all of them | **DONE** | *dev-salisare -> functions/data_preparation.py* |
| Calculate median values with respect to each group of titles and fill in these values instead if NAN | **-** | *-* |
| Create dummies for 'Sex' | **DONE** | *dev-salisare -> functions/data_preparation.py* |
| Create dummies for 'Embarked' | **DONE** | *dev-salisare -> functions/data_preparation.py* |
| Fill in median value of 'Fare' on the NAN places | **-** | *-* |
| Normalize 'Fare' and fill in NAN with average value | **DONE** | *dev-salisare -> functions/data_preparation.py* |
| Normalize 'Age' and fill in NAN with average value | **DONE** | *dev-salisare -> functions/data_preparation.py* |
| Factorize 'Age' and create dummies for the result | **DONE** | *dev-salisare -> functions/data_preparation.py* |

## Modelling

| Model Name | Variables | Accuracy on the Local Test Set | Accuracy on Kaggle |
| ------------- |:-------------:| -----:| -----:|
| Random Forest | (normalized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', 'Embarked' dummies) | **~86.6%** | **~77.5%** |
| Random Forest | (factorized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', 'Embarked' dummies, non-grouped 'Title') | **~86%** | **~76.1%** |
| Random Forest | (factorized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', non-grouped 'Title') | **-** | **~75%** |
