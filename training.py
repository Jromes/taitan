# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:53:26 2020

@author: GO
"""


import os
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

WKPATH = "C:/Users/GO/Desktop/1/GIT/taitan"

os.chdir(WKPATH)

train_data_name = "train.csv"
train_data_path = "/".join([WKPATH,train_data_name])

test_data_name = "test.csv"
test_data_path = "/".join([WKPATH,test_data_name])

gs_data_name = "gender_submission.csv"
gs_data_path = "/".join([WKPATH,gs_data_name])

train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')

# 1.Question or problem definition.
#泰坦尼克号生存
# survival:结果是否生存 0 = No, 1 = Yes
# pclass:票等级 代表身份地位 1 = 1st, 2 = 2nd, 3 = 3rd
# sex:性别 男女
# Age:年龄 
# sibsp:兄弟姐妹或者配偶在船上的数量
# parch:船上父母子女数量
# ticket:票号
# fare:票价
# cabin:船舱号
# embarked:上机地点Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton




# 2.Acquire training and testing data.

#训练数据
train_df = pd.read_csv('train.csv')

#测试数据
test_df = pd.read_csv('test.csv')

# 3.Wrangle, prepare, cleanse the data.
# train_df_desc = train_df.describe()
# train_df_head = train_df.head()

# train_df.info()
# print('_'*40)
# test_df.info()

# train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# 4.Analyze, identify patterns, and explore the data.
# 5.Model, predict and solve the problem.
# 6.Visualize, report, and present the problem solving steps and final solution.
# 7.Supply or submit the results.








