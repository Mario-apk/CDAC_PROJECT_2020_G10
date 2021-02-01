"""
#In a sense, it’s the employees who make the company. It’s the employees who do the work. It’s the employees who shape the company’s culture. Long-term success, a healthy work environment, and high employee retention are all signs of a successful company. But when a company experiences a high rate of employee turnover, then something is going wrong. This can lead the company to huge monetary losses by these innovative and valuable employees.

#Companies that maintain a healthy organization and culture are always a good sign of future prosperity. Recognizing and understanding what factors that were associated with employee turnover will allow companies and individuals to limit this from happening and may even increase employee productivity and growth. These predictive insights give managers the opportunity to take corrective steps to build and preserve their successful business.

#"You don't build a business. You build people, and people build the business." - Zig Ziglar

# Objective

Employee Turnover Prediction means to predict whether an employee is going to leave the organization in the coming period.

# Obtaining the Data
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

emp=pd.read_csv("EmployeeData.csv")

"""# Scrubbing the Data

Typically, cleaning the data requires a lot of work and can be a very tedious procedure. This dataset is super clean and contains no missing values. But still, I will have to examine the dataset to make sure that everything else is readable and that the observation values match the feature names appropriately.
"""

emp.isnull().any()

emp.head()

emp = emp.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

emp = emp[['turnover','satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany','workAccident','promotion','department','salary'
         ]]

"""# Exploring the Data

# Statistical Overview
"""

emp.shape

emp.dtypes

# Looks like about 76% of employees stayed and 24% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = emp.turnover.value_counts() / 14999
turnover_rate

salary = {'low':0, 'medium':1,'high':2}
emp['salary'] = emp['salary'].map(lambda x : salary[x])

emp_obj = pd.get_dummies(emp,\
        columns=['department'])
emp_obj

# Display the statistical overview of the employees
emp_obj.describe()

# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = emp_obj.groupby('turnover')
turnover_Summary.mean()

"""# Correlation Matrix & Heatmap

Moderate Positively Correlated Features:

projectCount vs evaluation: 0.349333
projectCount vs averageMonthlyHours: 0.417211
averageMonthlyHours vs evaluation: 0.339742
Moderate Negatively Correlated Feature:

satisfaction vs turnover: -0.388375

Summary:

From the heatmap, there is a positive(+) correlation between projectCount, averageMonthlyHours, and evaluation. Which could mean that the employees who spent more hours and did more projects were evaluated highly.

For the negative(-) relationships, turnover and satisfaction are highly correlated. I'm assuming that people tend to leave a company more when they are less satisfied.
"""

corr = emp_obj.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
corr

"""# Distribution Plots (Satisfaction - Evaluation - AverageMonthlyHours)

Summary: Let's examine the distribution on some of the employee's features. Here's what I found:

Satisfaction - There is a huge spike for employees with low satisfaction and high satisfaction.
Evaluation - There is a bimodal distrubtion of employees for low evaluations (less than 0.6) and high evaluations (more than 0.8)

AverageMonthlyHours - There is another bimodal distribution of employees with lower and higher average monthly hours (less than 150 hours & more than 250 hours)

The evaluation and average monthly hour graphs both share a similar distribution.
Employees with lower average monthly hours were evaluated less and vice versa.

If you look back at the correlation matrix, the high correlation between evaluation and averageMonthlyHours does support this finding.
"""

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(emp.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Evaluation
sns.distplot(emp.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(emp.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')

"""# Salary V.S. Turnover

Summary: This is not unusual. Here's what I found:

Majority of employees who left either had low or medium salary.

Barely any employees left with high salary

Employees with low to average salaries tend to leave the company.
"""

f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="salary", hue='turnover', data=emp).set_title('Employee Salary Turnover Distribution');

"""# Department V.S. Turnover

Summary: Let's see more information about the departments. Here's what I found:

The sales, technical, and support department were the top 3 departments to have employee turnover

The management department had the smallest amount of turnover
"""

color_types = sns.color_palette("Set2", 10)
sns.countplot(x='department', data=emp, palette=color_types).set_title('Employee Department Distribution');
 
# Rotate x-labels
plt.xticks(rotation=-90)

f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="department", hue='turnover', data=emp).set_title('Employee Department Turnover Distribution');

"""# Turnover V.S. ProjectCount

Summary: This graph is quite interesting as well. Here's what I found:

More than half of the employees with 2,6, and 7 projects left the company

Majority of the employees who did not leave the company had 3,4, and 5 projects

All of the employees with 7 projects left the company

There is an increase in employee turnover rate as project count increases
"""

ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=emp, estimator=lambda x: len(x) / len(emp) * 100)
ax.set(ylabel="Percent")

"""# Turnover V.S. Evaluation

Summary:

There is a biomodal distribution for those that had a turnover.

Employees with low performance tend to leave the company more

Employees with high performance tend to leave the company more

The sweet spot for employees that stayed is within 0.6-0.8 evaluation
"""

# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')
plt.title('Employee Evaluation Distribution - Turnover V.S. No Turnover')

"""# Turnover V.S. AverageMonthlyHours

Another bi-modal distribution for employees that turnovered

Employees who had less hours of work (~150hours or less) left the company more

Employees who had too many hours of work (~250 or more) left the company

Employees who left generally were underworked or overworked.
"""

fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')
plt.title('Employee AverageMonthly Hours Distribution - Turnover V.S. No Turnover')

"""# Turnover V.S. Satisfaction

Summary:

There is a tri-modal distribution for employees that turnovered

Employees who had really low satisfaction levels (0.2 or less) left the company more

Employees who had low satisfaction levels (0.3~0.5) left the company more

Employees who had really high satisfaction levels (0.7 or more) left the company more
"""

fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 0),'satisfaction'] , color='b',shade=True, label='no turnover')
ax=sns.kdeplot(emp.loc[(emp['turnover'] == 1),'satisfaction'] , color='r',shade=True, label='turnover')
plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')

"""# Turnover V.S. YearsAtCompany

Summary: Let's see if theres a point where employees start leaving the company. Here's what I found:

More than half of the employees with 4 and 5 years left the company

Employees with 5 years should highly be looked into
"""

ax = sns.barplot(x="yearsAtCompany", y="yearsAtCompany", hue="turnover", data=emp, estimator=lambda x: len(x) / len(emp) * 100)
ax.set(ylabel="Percent")

