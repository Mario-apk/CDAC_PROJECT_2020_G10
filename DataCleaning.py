"""# **Data cleaning**"""

def clean_data(emp):
    """to clean the data pass the dataframe itself """
    print("Columns with all null values are")
    print(emp.columns[emp.isnull().all()])
    
    # Drop columns which have all NaN values
    c=emp.columns[emp.isnull().all()]
    emp.drop(c, inplace=True, axis=1)
    
    # Drop Columns which have more than 90% NAs
    emp.dropna(axis=1, thresh=int(0.1 * emp.shape[0]),inplace=True)
    
    # Find rows with missing values greater than 50%
    print(emp.isnull().sum(axis=1))
    # Drop rows with missing values greater than 50%
    emp = emp[emp.isnull().sum(axis=1) <=(emp.shape[1] * 0.5) ]
    
    return emp

clean_data(emp)
