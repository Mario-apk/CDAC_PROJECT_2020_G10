#Modelbuilding.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score , classification_report , confusion_matrix , plot_roc_curve
import matplotlib.pyplot as plt


def model_test(X_train, Y_train,X_test,Y_test,choice = 1):

    if choice == 1:
        from sklearn.linear_model import LogisticRegression
        #from sklearn.model_selection import KFold,cross_val_score
        log=LogisticRegression(penalty='l2', C=1)
        log.fit(X_train, Y_train)
        ypred=log.predict(X_test)

	"""model = LogisticRegression(penalty='l2', C=1)
	model.fit(X_train, Y_train)"""
	print ("Logistic accuracy is %2.2f" % accuracy_function(Y_test, model.predict	(X_test)))


        #CLASSIFICATION REPORT
        print(classification_report(Y_test,ypred))
        #ROC CURVE
        plot_roc_curve(log , X_test , Y_test)
        plt.show()
        
        #BIAS AND VARIANCE
        #kf = KFold(shuffle=True , n_splits=5 , random_state=7)
        #score = cross_val_score(log , X , y , cv=kf , scoring='roc_auc')
        #bias1 = np.mean(1-score)
        #variance1 = np.std(score , ddof=1)
        #print(bias1 , variance1)
        
    elif choice == 2:
  
	from sklearn.svm import SVC, LinearSVC
        from sklearn.model_selection import KFold,cross_val_score
        
	svc = SVC()
	svc.fit(X_train, Y_train)
	ypred=svc.predict(X_test)

	#CLASSIFICATION REPORT
        print(classification_report(Y_test,ypred))
        #ROC CURVE
        plot_roc_curve(SVC , X_test , Y_test)
        plt.show()


	acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
	acc_svc

        
    elif choice == 3:
        from sklearn.neighbors import KNeighborsClassifier
        KNN = KNeighborsClassifier(n_neighbors = 3)
        KNN.fit(X_train, Y_train)
        ypred=KNN.predict(X_test)
        print(classification_report(Y_test,ypred))
        plot_roc_curve(KNN , X_test , Y_test)
        plt.show()

    elif choice == 4:
    
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X_train, Y_train)
        ypred=rf.predict(X_test)
        print(classification_report(Y_test,ypred))
        plot_roc_curve(rf , X_test , Y_test)
        plt.show()

        print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(Y_test, rf.predict(X_test))))



    elif choice == 5:
        from sklearn.tree import DecisionTreeClassifier
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)
        ypred=decision_tree.predict(X_test)
        print(classification_report(Y_test,ypred))
        plot_roc_curve(decision_tree , X_test , Y_test)
        plt.show()

        acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
        acc_decision_tree

    elif choice == 6:
        from sklearn.ensemble import GradientBoostingClassifier
        GBoost=GradientBoostingClassifier(n_estimators=100)
        GBoost.fit(X_train, Y_train)
        ypred=GBoost.predict(X_test)
        print(classification_report(Y_test,ypred))
        plot_roc_curve(GBoost , X_test , Y_test)
        plt.show()

    elif choice ==7:
        from xgboost import XGBClassifier
        XGB = XGBClassifier()
        XGB
        XGB.fit(X_test, Y_test,eval_metric='auc')
        XGB
        ypred = XGB.predict(X_test)
        print(classification_report(Y_test,ypred))
        plot_roc_curve(XGB , X_test , Y_test)
        plt.show()
