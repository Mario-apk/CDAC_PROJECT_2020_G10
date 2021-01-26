#Datapreprocessing.py

from sklearn import preprocessing
def prerocess_data(df):
	"""to process the data pass the dataframe itself"""
	#imputation
	df.isna().sum()

	#Handling outliers
	data = ['satisfaction_level', 'last_evaluation','average_montly_hours']
	figure = df.boxplot(column= data)
	#NO outlier situation

	"""**Analyze by binary relationship table**

	From above we see "left" with "satisfaction_level", "salary" and "work_accident" has 	strong negative correlation, "left" and "time_spend_company" have strong positive 		correlation. We can further discover the precise relationship below.
	"""
	df[['left', 'satisfaction_level']].groupby(['left'], as_index=False).mean			().sort_values(by='satisfaction_level', ascending=False)

	
def encode_cat_data(df_cat):
	#Onehot encoding (nominal)
	df_obj = pd.get_dummies(df,columns=['department', 'salary'])
	return df_obj
