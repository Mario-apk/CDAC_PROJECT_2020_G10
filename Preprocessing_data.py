
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_preprocessing(emp):
    
     print("The shape of our data is ",emp.shape)  
     print("**********************************************************************")

#Checking Null Values we can see there is no null values it means our data is cleaned
     print("Checking for null values")
     print(emp.isnull().any())
     print("We can see there is no null values so we can say our data is a cleaned data ")
     print("***************************************************************************")
                
#Renaming Column Names

     emp = emp.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

     emp = emp[['turnover','satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany','workAccident','promotion','department','salary'
         ]]
     
     
     salary = {'low':0, 'medium':1,'high':2}
     emp['salary'] = emp['salary'].map(lambda x : salary[x])
     
  
#One Hot Encoding 
    
     emp = pd.get_dummies(emp,\
        columns=['department'])
         
#Normalization     
     X = emp.drop('turnover', axis=1)
     minmaxscaler = MinMaxScaler()
     X=minmaxscaler.fit_transform(X)  
     return emp 
    
    
#DataBalancing
    
def balance_data(X,y):
  
    xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,stratify=y)
    sm = SMOTE(random_state=12, sampling_strategy = 1)
    X_train, y_train = sm.fit_sample(xtrain, ytrain)
    print("After Balancing Data by Upsampling minority Class\n",y_train.value_counts())  
    return X_train,y_train,xtest,ytest