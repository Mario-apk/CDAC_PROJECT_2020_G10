from sklearn.metrics import roc_auc_score,plot_roc_curve
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix 
import matplotlib.pyplot as plt

def svm_models(X_train, y_train,X_test,y_test):
    
    print("**********************************************************************")
    
    print("This is SVM KERNEL")
      
    svc_model= SVC(kernel="poly",degree=5)
    
    svc_model.fit(X_train,y_train)
    
    pred = svc_model.predict(X_test)
    
    print("Accuracy of SVM = %2.2f "% accuracy_score(y_test,pred))
     
    print ("SVM  AUC = %2.2f " % roc_auc_score(y_test,pred))
    
    print("Classification report of SVM is as below \n",classification_report(y_test,pred))
     
    print("Confusion_matrix of SVM is as below \n",confusion_matrix(y_test,pred))
     
    plot_roc_curve(svc_model, X_test , y_test)
    
    plt.show()
     
    print("**********************************************************************")

