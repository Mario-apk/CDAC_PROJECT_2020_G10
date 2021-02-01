from sklearn.metrics import roc_auc_score,plot_roc_curve
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix 
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def GradientBoosting_Classifier(X_train, y_train,X_test,y_test):
    
        print("**********************************************************************")
      
        print("This is GradientBoosting Classifier")

        GBoost=GradientBoostingClassifier(n_estimators=100)
        
        GBoost.fit(X_train, y_train)
        
        y_pred=GBoost.predict(X_test)
        
        print("Accuracy of GradientBoosting = %2.2f " % accuracy_score(y_test, y_pred))
              
        print ("GradientBoosting AUC = %2.2f" % roc_auc_score(y_test,y_pred))
        
        print("Classification_report is as below \n",classification_report(y_test,y_pred))
        
        print("Confusion_matrix is as below \n",confusion_matrix(y_test,y_pred))
        
        plot_roc_curve(GBoost , X_test , y_test)
    
        plt.show()
        
        print("**********************************************************************")
        

def AdaBoosting_Classifier(X_train, y_train,X_test,y_test):
      
        print("**********************************************************************")
        
        print ("This is AdaBoost Model")
  
        ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
      
        ada.fit(X_train,y_train)
      
        pred=ada.predict(X_test)
      
        print("Accuracy of  AdaBoosting = %2.2f" % accuracy_score(y_test, pred))
                  
        print ("AdaBoost AUC = %2.2f" % roc_auc_score(y_test,pred))
          
        print("Classification Report of  AdaBoosting is as below \n" ,classification_report(y_test, pred))
            
        print("Confusion_matrix of  AdaBoosting is as below \n",confusion_matrix(y_test,pred))
            
            
        plot_roc_curve(ada , X_test , y_test)
        
        plt.show()
        
        print("**********************************************************************")
        
def XG_Boosting_Classifier(X_train, y_train,X_test,y_test):
    
        print("**********************************************************************")
    
        print ("This is XGBoost Model")
  
        xg = XGBClassifier(n_estimators=400,random_state=2000)
      
        xg.fit(X_train,y_train)
      
        y_predi=xg.predict(X_test)
      
        print("Accuracy of  XGBoosting = %2.2f" % accuracy_score(y_test, y_predi))
                  
        print ("XGBoost AUC = %2.2f" % roc_auc_score(y_test,y_predi))
          
        print("Classification Report of  XGBoosting is as below \n" ,classification_report(y_test, y_predi))
            
        print("Confusion_matrix of  XGBoosting is as below \n",confusion_matrix(y_test,y_predi))
            
        plot_roc_curve(xg , X_test , y_test)
        
        plt.show()
        
        print("**********************************************************************")
        
    