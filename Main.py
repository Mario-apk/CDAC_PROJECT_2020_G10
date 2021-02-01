import pandas as pd
import sys
from Logistic_model import logistic_regression_model
from SVM_model import svm_models
from Boosting_Model import GradientBoosting_Classifier,AdaBoosting_Classifier,XG_Boosting_Classifier
from DecisionTree_Classifier_model import DecisionTree_model
from Randomforest_model import Randomforest_Classifier_model
from Preprocessing_data import data_preprocessing,balance_data

emp=pd.read_csv("EmployeeData.csv")

emp=data_preprocessing(emp)

print("Employee Data after Preprocessing \n",emp.head())
print("************************************************************************")
turnover_rate =( emp.turnover.value_counts() / 14999 )*100
print("Employee Turnover rate is \n",turnover_rate)
print("We can conclude from Turnover rate our data is imbalanced so we have to make it balanced")
print("************************************************************************")

X = emp.drop('turnover', axis=1)

y=emp['turnover']

X_train,y_train,xtest,ytest = balance_data(X,y)

choice = 0

while choice!=8:
    
    print("**********************************************************************")
    print("Choose the model")
    print("   1.DecisionTree")
    print("   2.Logistic Regression")
    print("   3.SVM")
    print("   4.RandomForest")
    print("   5.GradientBoostingModel")
    print("   6.AdaBoostingModel")
    print("   7.XGBoostingModel")
    print("   8. Exit")
    print("**********************************************************************")
    
    choice=int(input("enter choice "))
    
    if choice == 1:
        DecisionTree_model(X_train,y_train,xtest,ytest)
              
    elif choice == 2: 
        logistic_regression_model(X_train,y_train,xtest,ytest)
        
    elif choice == 3:
        svm_models(X_train,y_train,xtest,ytest)
             
    elif choice == 4:
        Randomforest_Classifier_model(X_train,y_train,xtest,ytest)
               
    elif choice == 5:
        GradientBoosting_Classifier(X_train,y_train,xtest,ytest)
               
    elif choice == 6:
        AdaBoosting_Classifier(X_train,y_train,xtest,ytest)
            
    elif choice == 7:
        XG_Boosting_Classifier(X_train,y_train,xtest,ytest)
             
    elif choice == 8:
        sys.exit(0)
    






