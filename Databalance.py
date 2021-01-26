#Databalance.py 

def balance_data(X,y):
	nHead = int(len(df_obj)*0.85)
	nTail = int(len(df_obj)*0.15)
	X_train = df_obj.drop("left", axis=1).head(nHead)
	X_test  = df_obj.drop("left", axis=1).tail(nTail)
	Y_train = df_obj["left"].head(nHead)
	Y_test = df_obj["left"].tail(nTail)
	X_train.shape, X_test.shape
