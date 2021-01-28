"""# **Imputation**"""

from sklearn.impute import KNNImputer
def fill_numeric_data(emp,neighbors = 2):
    """ provide dataframe and neighbors , by default it is 2 """
    imputer = KNNImputer(n_neighbors=neighbors, weights="uniform")
    cols = emp.columns
    filled_array = imputer.fit_transform(emp)
    emp_filled = pd.DataFrame(filled_array, columns = cols)
    return emp_filled
