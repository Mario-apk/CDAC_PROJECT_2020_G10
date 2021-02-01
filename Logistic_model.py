from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,plot_roc_curve
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix 
import matplotlib.pyplot as plt


def logistic_regression_model(X_train, y_train,X_test,y_test):
    logis = LogisticRegression()
    
    logis.fit(X_train, y_train)
    
    y_pred=logis.predict(X_test)
    
    print("**********************************************************************")
    
    print("This is  a Logistic Regression Model")
     
    print("Accuracy of Logistic Regression = %2.2f" % accuracy_score(y_test, y_pred))
    
    print ("Logistic AUC = %2.2f" % roc_auc_score(y_test,y_pred))
    
    print("Classification_report of Logistic Regression is as below \n",classification_report(y_test,y_pred))
    
    print("Confusion_matrix of Logistic Regression is as below \n" ,confusion_matrix(y_test,y_pred))
      
    plot_roc_curve(logis , X_test , y_test)
    
    plt.show()
    
    print("**********************************************************************")
   
   
