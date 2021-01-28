"""# **Normalization**"""

from sklearn.preprocessing import MinMaxScaler
def normalize_num_data(emp_num):
  emp_num=emp_num.drop('turnover',axis=1)
  minmaxscaler = MinMaxScaler()
  emp_num_pt=minmaxscaler.fit_transform(emp_num)
  emp_num_pt=pd.DataFrame(emp_num_pt)
  emp_num_pt.columns=emp_num.columns
  return emp_num_pt

#Onehot encoding (nominal)
def encode_cat_data(emp_cat):
  emp_cat_dum = pd.get_dummies(emp_cat,columns=list(emp_cat.columns),drop_first=True)
  return emp_cat_dum
