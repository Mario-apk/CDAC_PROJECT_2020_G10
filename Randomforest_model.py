from sklearn.metrics import roc_auc_score,plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix 

def Randomforest_Classifier_model(X_train, y_train,X_test,y_test):
    
        print("**********************************************************************")
    
        print("This is RandomForestClassifier")
        
        rf_model = RandomForestClassifier()
        
        rf_model.fit(X_train, y_train)
        
        y_pred=rf_model.predict(X_test)
          
        print("Accuracy of RandomForest is = %2.2f" % accuracy_score(y_test, y_pred))
               
        print ("Random Forest AUC = %2.2f" % roc_auc_score(y_test,y_pred))
        
        print("Classification report of RandomForest is as below \n",classification_report(y_test,y_pred))
        
        print("Confusion matrix of RandomForest is as below \n",confusion_matrix(y_test,y_pred))
        
        plot_roc_curve(rf_model , X_test , y_test)
    
        plt.show()
        
        print("**********************************************************************")
   
   
        
  

