import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

train = pd.read_csv("data/train.csv", sep = ",", header = 0, skip_blank_lines = True)
name1 = train['Name'].str.split('.',expand=True)
name2 = name1[0].str.split(',',expand=True)
train['Title'] = name2[1]
train = train.drop('Cabin', axis=1)
train = train.drop('Embarked', axis=1)
train = train.drop('Fare', axis=1)
train = train.drop('Ticket', axis=1)
train = train.drop('Parch', axis=1)
train = train.drop('SibSp', axis=1)
train = train[['PassengerId', 'Survived', 'Pclass', 'Title', 'Name', 'Sex', 'Age']]
female_analyse = train.loc[train['Sex'] == 'female'][['Age','Survived']].dropna().to_dict('list')
male_analyse = train.loc[train['Sex'] == 'male'][['Age','Survived']].dropna().to_dict('list')


female_dict = {}
male_dict = {}
for i in range(0,9):
    female_dict[10*(i+1)] = 0
    male_dict[10*(i+1)] = 0 
    for j in range(0,len(female_analyse['Age'])):
        if female_analyse['Age'][j] >= 10*i and female_analyse['Age'][j] < 10*(i+1) and female_analyse['Survived'][j] == 1:
            female_dict[10*(i+1)] += 1
    female_dict[10*(i+1)] = female_dict[10*(i+1)]*100/len(female_analyse['Age'])
    for k in range(0,len(male_analyse['Age'])):
        if male_analyse['Age'][k] >= 10*i and male_analyse['Age'][k] < 10*(i+1) and male_analyse['Survived'][k] == 1:
            male_dict[10*(i+1)] += 1
    male_dict[10*(i+1)] = male_dict[10*(i+1)]*100/len(male_analyse['Age'])

data1 = sorted(female_dict.items())
x1, y1 = zip(*data1)
plt.bar(x1, y1)
plt.xlabel('Age')
plt.ylabel('Percentage survived female') 
plt.show()
data2 = sorted(male_dict.items())
x2, y2 = zip(*data2)
plt.bar(x2, y2)
plt.xlabel('Age')
plt.ylabel('Percentage survived male') 
plt.show()


##How get rid of reading csv each time i need smth.



