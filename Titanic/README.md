## Data cleaning

| Task        | Status           | Path  | Date |
| ------------- |:-------------:| -----:| ------:|
| Get 'Title' from the names and group some values together | **DONE** | *dev-sahanmar -> clean_data.py* | 16.11.18 |
| Get titles from the names and create dummies for all of them | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Calculate median values with respect to each group of titles and fill in these values instead if NAN | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |
| Create dummies for 'Sex' | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Create dummies for 'Embarked' | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Fill in median value of 'Fare' on the NAN places | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |
| Normalize 'Fare' and fill in NAN with average value | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Normalize 'Age' and fill in NAN with average value | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Factorize 'Age' and create dummies for the result | **DONE** | *dev-salisare -> functions/data_preparation.py* | 17.11.18 |
| Create new column with family size | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |
| Create new column if is alone or not | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |
| Create new column with fare quantile cut to bins and converted to factorized | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |
| Create new column with age cut to bins and converted to factorized | **DONE** | *dev-sahanmar -> clean_data.py* | 19.11.18 |


## Modelling

| Author | Model Name | Variables | Accuracy on the Local Test Set | Accuracy on Kaggle | Date |
 -----:| ------------- |:-------------:| -----:| -----:| -----:|
salisare | Random Forest | (normalized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', 'Embarked' dummies) | **~86.6%** | **~77.5%** | 16.11.18
salisare | Random Forest | (factorized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', 'Embarked' dummies, non-grouped 'Title') | **~86%** | **~76.1%** | 18.11.18
salisare| Random Forest | (factorized 'Age', 'Pclass' dummies, 'Sex' dummies, normalized 'Fare', non-grouped 'Title') | **-** | **~75%** |
sahanmar| Random Forest | (normalized 'Age', 'Pclass', 'Sex' factorized, normalized 'Fare', grouped factorized 'Title', ' SibSp', 'Parch', 'FamilySize', 'IsAlone', factorized 'NameCode', factorized 'Embarked', group factorized 'AgeBin', group factorized 'FareBin' ) | **~83.5%** | **~78.9%** | 
