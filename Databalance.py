"""# **Data Balance using smote** """

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

def balance_data(X,y):
    y=y.astype('int64')
    xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,stratify=y)
    smotenc = SMOTENC([0,1,2,3,4,5])
    X_train,y_train = smotenc.fit_resample(xtrain,ytrain)
    #print(y_train.value_counts())
    
    return X_train,y_train,xtest,ytest

X_train,y_train,xtest,ytest = balance_data(X,y)

new_series = pd.Series(y_train)
new_series.value_counts()
