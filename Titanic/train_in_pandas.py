import pandas as pd

df = pd.read_csv("data/train.csv", sep = ",", header = 0, skip_blank_lines = True)
titanic_dict = df.to_dict('list')
print(list(df))
#titanic_dict = df.T.to_dict('list') # T means transporting
print(titanic_dict)
import train_new
###string and int???!!!
