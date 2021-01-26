#Correlation.py

def corr_matrix_data:
	
	"""Initioal dataframe as input"""
	#Correlation Matrix
	corr = df.corr()
	corr = (corr)
	sns.heatmap(corr, 
            	xticklabels=corr.columns.values,
            	yticklabels=corr.columns.values)

	corr	

	"""Encoded dataframe as input"""
	#Correlation Matrix
	corr = df_obj.corr()
	corr = (corr)
	sns.heatmap(corr, 
            	xticklabels=corr.columns.values,
            	yticklabels=corr.columns.values)

	corr
