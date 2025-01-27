# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:47:27 2025

@author: Zeynep
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score

dataset=pd.read_csv("breast_cancer.csv")

X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_predict=classifier.predict(X_test)
y_predict_prob=classifier.predict_proba(X_test)

cm = confusion_matrix(y_test, y_predict)
AccuracyScore=accuracy_score(y_test, y_predict)

# k-Fold Cross Validation Yöntemiyle Modelin Performansının Ölçülmesi
from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator=classifier, X=X, y=y, cv=10)
AccuriesMean=accuricies.mean()*100
StandartDeviation=accuricies.std()*100