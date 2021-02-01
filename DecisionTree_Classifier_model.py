from sklearn.metrics import roc_auc_score,plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix 
import matplotlib.pyplot as plt

def DecisionTree_model(X_train, y_train,X_test,y_test):

        decision_tree = DecisionTreeClassifier()
        
        decision_tree.fit(X_train, y_train)
        
        y_pred=decision_tree.predict(X_test)
        
        print("**********************************************************************")
        
        print("This is a DecisionTreeClassifier")
        
        print("Accuracy of Decision Tree = %2.2f" % accuracy_score(y_test, y_pred))
            
        print ("Decision Tree AUC = %2.2f" % roc_auc_score(y_test,y_pred))
        
        print("Classification report of Decision Tree is as below \n",classification_report(y_test,y_pred))
        
        print("Confusion matrix of Decision Tree is as below \n",confusion_matrix(y_test,y_pred))
        
        
        plot_roc_curve(decision_tree, X_test , y_test)
    
        plt.show()
         
        print("**********************************************************************")


