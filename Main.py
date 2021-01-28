import sys
import pandas as pd
from DataCleaning import clean_data
from DataNormalization import normalize_num_data,encode_cat_data
from Datafill import fill_numeric_data
from Databalance import balance_data
from ModelBuilding import model_test

emp  = pd.read_csv(input("Please Enter file name with .csv format"+"\n"))


#print(emp.head())
#print(emp.shape) 
emp = emp.rename(columns={'sales' : 'department'})
print(emp.columns)
print(emp.info())
print(emp.isnull().sum())
print(emp.dtypes)

if(int(input("Enter 1 to start cleaning process"))==1):
    clean_data(emp)
    print("cleaing processe finished ,Data is cleaned now")
else:
    sys.exit("pls choose from given option")

emp_num=emp.select_dtypes(exclude='object')
emp_cat=emp.select_dtypes(include='object')

print("Numerial Columns :", emp_num.columns + "\n")
print("Categorical Columns :", emp_cat.columns+"\n")

print("Data Filliing process starting",fill_numeric_data(emp_num),"Data Filled"+"\n")

print("Normalization process Starts")
print(normalize_num_data(emp_num).head(),"\n")

print(encode_cat_data(emp_cat).head(),"\n")
print("Normalization process ends")

X=pd.concat([encode_cat_data(emp_cat),normalize_num_data(emp_num)],axis=1)
y=emp['turnover']




X_train,y_train,xtest,ytest = balance_data(X,y)
print ("choose option for model test : 1. logistic regression\n 2. Support vector machine\n 3. K Neighbours Classifier\n 4. Random Forest Classifier\n 5. Decision Tree\n 6. Gradient Boosting Classifier\n 7. exit")

flag=True
while(flag): 
    ch = int(input())
    if (ch < 6):
        model_test(X_train, y_train,xtest,ytest,choice = ch)
    else :
        flag= False
