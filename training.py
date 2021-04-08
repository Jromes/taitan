# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:53:26 2020

@author: GO
"""

# KNN or k-Nearest Neighbors
# Support Vector Machines
# Naive Bayes classifier
# Decision Tree
# Random Forrest
# Perceptron
# Artificial neural network
# RVM or Relevance Vector Machine



# 6.Visualize, report, and present the problem solving steps and final solution.
# 7.Supply or submit the results.


import os
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import pickle
import gc
from collections import defaultdict
from pandas_profiling import ProfileReport as ppf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost
#读取数据
WKPATH = "C:/Users/GO/Desktop/1/GIT/taitan"
os.chdir(WKPATH)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df,test_df]

#分析数据,列名

# train_df.info()

# =============================================================================
#泰坦尼克号生存
# survival:结果是否生存 0 = No, 1 = Yes
##存活率0.38
# pclass:票等级 代表身份地位 1 = 1st, 2 = 2nd, 3 = 3rd
##大部分为3级,
# sex:性别 男女
# Age:年龄 
# sibsp:兄弟姐妹或者配偶在船上的数量
# parch:船上父母子女数量
# ticket:票号
# fare:票价
# cabin:船舱号
# embarked:上机地点Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# =============================================================================


#profile简单分析图
# profile = ppf(train_df,title="Pandas Profiling Report")
# profile.to_file("your_report.html")


# train_df.Pclass.value_counts()
#train_df.groupby('Pclass').size()
# train_df.groupby('Pclass')["Survived"].agg('sum')/train_df.groupby('Pclass')

#根据目标值区分存活率
def com_prop(data,group_str):
    x = pd.DataFrame(columns=['prop','sum'])
    d_sum = data.sum()
    for a,b in data.groupby([group_str]):
        x.loc[a] = [b['Survived'].sum()/b['Survived'].count(),b['Survived'].sum()]

    return x

# =============================================================================
# com_prop(train_df,'Pclass')
# train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',ascending=False)
# #1生存率0.6,3为0.24
# com_prop(train_df,'Sex')
# #女0.74
# com_prop(train_df,'SibSp')
# #正态分布,1为0.5,58为0
# com_prop(train_df,'Parch')
# #同上
# com_prop(train_df,'Embarked')
# #C0.5,不明显
# =============================================================================

# =============================================================================
# train_df['Fare'].value_counts()
# fig,axes = plt.subplots(2,1,figsize=(10,8))
# train_df['Fare'].value_counts().plot(kind='bar')
# 
# =============================================================================
# =============================================================================
# pd.qcut(ttt_df['Age'], 3)
# bins = [0, 10, 15, 20, 25,30,40,50,100]
# cats = pd.cut(ttt_df['Age'], bins)
# cats.value_counts()
# =============================================================================

# =============================================================================
# g= sns.FacetGrid(train_df,col='Survived')
# g.map(plt.hist,'Age',bins=20)
# 
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();
# 
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# 
# 
# grid = sns.FacetGrid(train_df,col='Embarked',row='Survived',size =2.2,aspect=1.6)
# grid.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)
# grid.add_legend()
# =============================================================================


# print('before',train_df.shape,test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'],axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train_df['Title'],train_df['Sex'])    

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
pd.crosstab(train_df['Title'],train_df['Sex'])    
com_prop(train_df,'Title')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df=test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] =dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))


for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i)&(dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j]=int(age_guess/0.5+0.5)*0.5
            
    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1),'Age']=guess_ages[i,j]
        
    dataset['Age'] = dataset['Age'].astype(int)

train_df["AgeBand"] = pd.cut(train_df["Age"],5)

# com_prop(train_df,'AgeBand')

for dataset in combine:
    dataset.loc[dataset['Age']<=16,'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']=4

# com_prop(train_df,'Age')

train_df = train_df.drop(['AgeBand'], axis =1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1
    
# com_prop(train_df,'IsAlone')

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# com_prop(train_df,'Embarked')

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]





















