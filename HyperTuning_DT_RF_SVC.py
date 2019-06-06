#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:24:21 2019

@author: abhineet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

data = pd.read_csv('sonar.all-data', delimiter=',')
#Data Pre-processing
df=data.replace(to_replace =("R","M"), value =(1,0)) 
df = (df - df.min())/(df.max()-df.min())

#print(data.shape)
#Seperating independent and dependent variables
X = df.iloc[:,: -1]
y = df.iloc[:, 60:].values.ravel()

rf = RandomForestClassifier(n_estimators= 100, max_features = 42, max_depth = 5, random_state=42)
rf.fit(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2)
'''
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(accuracy_score(y_test, y_pred))
'''

#***********FUNCTIONS***********


#Hyperparameter tuning for Random Forest

def hyper_tuning_rf(train_X, train_y, test_X, test_y):
    
    #hyperparameters
    #number of trees
    for i in range(50,100):
        rf = RandomForestClassifier(criterion = 'gini', n_estimators = i)
        rf.fit(train_X, train_y)
        train_pred = rf.predict(train_X)
        test_pred = rf.predict(test_X)
        #Plotting AUC Curves to find the best n_estimators value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("AUC")
    plt.show()
    
    #maximum features
    for i in range(1,30):
        rf = RandomForestClassifier(criterion = 'gini', max_features = i)
        rf.fit(train_X, train_y)
        train_pred = rf.predict(train_X)
        test_pred = rf.predict(test_X)
        #Plotting AUC Curves to find the best max_features value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for max_features")
    plt.xlabel("max_features")
    plt.ylabel("AUC")
    plt.show()
    
    #returning accuracy score
    rf = RandomForestClassifier(criterion = 'gini', n_estimators = 100, max_features = 19)
    rf.fit(train_X, train_y)
    
    return accuracy_score(test_y, rf.predict(test_X))

rfacc = hyper_tuning_rf(X_train, y_train, X_test, y_test)
print("Accuracy after tuning for Random Forest is:",rfacc)

#Hyperparameter tuning for Decision Trees

def hyper_tuning_dt(train_X, train_y, test_X, test_y):
    
    #hyperparameters
    #maximum depth
    for i in range(1,50):
        dt = DecisionTreeClassifier(criterion = 'gini', max_depth = i)
        dt.fit(train_X, train_y)
        train_pred = dt.predict(train_X)
        test_pred = dt.predict(test_X)
        #Plotting AUC Curves to find the best max_depth value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for max_depth")
    plt.xlabel("max_depth")
    plt.ylabel("AUC")
    plt.show()
    
    #minimum number of samples in a leaf
    for i in range(1,50):
        dt = DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = i)
        dt.fit(train_X, train_y)
        train_pred = dt.predict(train_X)
        test_pred = dt.predict(test_X)
        #Plotting AUC Curves to find the best min_samples_leaf value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for min_samples_leaf")
    plt.xlabel("min_samples_leaf")
    plt.ylabel("AUC")
    plt.show()
    
    #returning accuracy score
    dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 4, min_samples_leaf = 13)
    dt.fit(train_X, train_y)
    
    return accuracy_score(test_y, dt.predict(test_X))

dtacc = hyper_tuning_dt(X_train, y_train, X_test, y_test)
print("Accuracy after tuning for Decision Trees is:",dtacc)

#Hyperparameter tuning for Support Vector Machines

def hyper_tuning_svm(train_X, train_y, test_X, test_y):
    
    #hyperparameters
    #gamma
    for i in range(-3,3):
        svmc = svm.SVC(kernel='rbf', gamma = 10**i)
        svmc.fit(train_X, train_y)
        train_pred = svmc.predict(train_X)
        test_pred = svmc.predict(test_X)
        #Plotting AUC Curves to find the best gamma value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for gamma")
    plt.xlabel("gamma")
    plt.ylabel("AUC")
    plt.show()
    
    #Iterating over C
    for i in range(1,10):
        svmc = svm.SVC(kernel='rbf',C=10*i,gamma='auto')
        svmc.fit(train_X,train_y)
        train_pred = svmc.predict(train_X)
        test_pred = svmc.predict(test_X)
        #Plotting AUC Curves to find the best C value
        fpr, tpr, th = roc_curve(train_y, train_pred)
        auc_train = auc(fpr, tpr)
        fpr, tpr, th = roc_curve(test_y, test_pred)
        auc_test = auc(fpr, tpr)
        plt.scatter(i,auc_train, color='blue',label='Training data')
        plt.scatter(i,auc_test, color='green',label='Testing data')
    plt.title("AUC for C")
    plt.xlabel("C")
    plt.ylabel("AUC")
    plt.show()
    
    #returning accuracy score
    svmc = svm.SVC(kernel='rbf',C=3*10,gamma=10**0)
    svmc.fit(train_X, train_y)
    
    return accuracy_score(test_y, svmc.predict(test_X))
 
svmacc = hyper_tuning_svm(X_train, y_train, X_test, y_test)
print("Accuracy after tuning for Support Vector Machines is:",svmacc)

#Final Accuracies
print("Accuracy after tuning for Random Forest is:",rfacc, "\n")
print("Accuracy after tuning for Decision Trees is:",dtacc, "\n")
print("Accuracy after tuning for Support Vector Machines is:",svmacc)