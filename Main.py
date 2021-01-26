#Main.py

import sys
import pandas as pd
from Datapreprocessing import prerocess_data,encode_cat_data
from Correlation import corr_matrix_data
from Accuracyscore import accuracy_function
from Databalance import balance_data
from ModelBuilding import model_test

df  = pd.read_csv(input("Please Enter file name with .csv format"+"\n"))
print(df.head())
print(df.shape) 
print(
	df = df.rename(columns={
                        	'sales' : 'department'
                       		}))
print(df.columns)
print(df.info())
print(df.isnull().sum().sum())
print(df.dtypes)

df_num=df.select_dtypes(exclude='object')
df_cat=df.select_dtypes(include='object')

print("Numerial Columns :", df_num.columns + "\n")
print("Categorical Columns :", df_cat.columns+"\n")

print(encode_cat_data(df_cat).head(),"\n")


X_train,Y_train,X_test,Y_test = balance_data(X,y)
print ("choose option for model test : 1. logistic regression\n 2. Support vector machine\n 3. K Neighbours Classifier\n 4. Random Forest Classifier\n 5. Decision Tree\n 6. Gradient Boosting Classifier\n 7. XGBoost Classifier\n 8. exit")

flag=True
while(flag): 
    ch = int(input())
    if (ch < 7):
        model_test(X_train, Y_train,X_test,Y_test,choice = ch)
    else :
        flag= False
