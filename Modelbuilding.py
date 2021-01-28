"""# **Model Building**"""

#Modelbuilding.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix , plot_roc_curve
import matplotlib.pyplot as plt


def model_test(X_train, y_train,xtest,ytest,choice = 1):

    if choice == 1:
        
        from sklearn.linear_model import LogisticRegression     
        #from sklearn.model_selection import KFold,cross_val_score
        log=LogisticRegression()
        log.fit(X_train, y_train)
        ypred=log.predict(xtest)
        #CLASSIFICATION REPORT
        print(classification_report(ytest,ypred))
        
        precision,recall,fscore,support=score(ytest,ypred)   # ,average='macro'
        print('Precision : {}'.format(precision))
        print('Recall    : {}'.format(recall))
        print('F-score   : {}'.format(fscore))
        print('Support   : {}'.format(support))
        #ROC CURVE
        plot_roc_curve(log , xtest , ytest)
        plt.show()

        acc_log = round(log.score(xtest , ytest) * 100, 2)
	acc_log
        #CONFUSION MATRIX
        emp_confusion = confusion_matrix(ytest, ypred)
        emp_confusion
        
        plt.rcParams['figure.figsize'] = (10, 6) 
        cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
        sns.heatmap(emp_confusion,cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,fmt='d')
        f1 = f1_score(ytest,ypred,labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        print(f1)        
        
    elif choice == 2:
    
        from sklearn.svm import SVC, LinearSVC
        from sklearn.model_selection import KFold,cross_val_score
        
	svc = SVC()
	svc.fit(X_train, y_train)
	ypred=svc.predict(xtest)

	#CLASSIFICATION REPORT
        print(classification_report(ytest,ypred))
        #ROC CURVE
        plot_roc_curve(svc , xtest , ytest)
        plt.show()


	acc_svc = round(svc.score(xtest , ytest) * 100, 2)
	acc_svc

        
      elif choice == 3:
        from sklearn.neighbors import KNeighborsClassifier
        KNN = KNeighborsClassifier(n_neighbors = 3)
        KNN.fit(X_train, y_train)
        ypred=KNN.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(KNN , xtest , ytest)
        plt.show()

      elif choice == 4:
    
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        ypred=rf.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(rf , xtest , ytest)
        plt.show()

	acc_rf = round(rf.score(xtest , ytest) * 100, 2)
	acc_rf


      elif choice == 5:
        from sklearn.tree import DecisionTreeClassifier
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, y_train)
        ypred=decision_tree.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(decision_tree , xtest , ytest)
        plt.show()

        acc_decision_tree = round(decision_tree.score(xtest, ytest) * 100, 2)
        acc_decision_tree

      elif choice == 6:
        from sklearn.ensemble import GradientBoostingClassifier
        GBoost=GradientBoostingClassifier(n_estimators=100)
        GBoost.fit(X_train, y_train)
        ypred=GBoost.predict(xtest)
        print(classification_report(ytest,ypred))
        plot_roc_curve(GBoost , xtest , ytest)
        plt.show()

        acc_GBoost = round(GBoost.score(xtest, ytest) * 100, 2)
        acc_GBoost
