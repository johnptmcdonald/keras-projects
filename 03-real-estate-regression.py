# Regression analysis is the statistical process of studying the relationship between a set of independent variables (explanatory variables) and the dependent variable (response variable)

# regression is good for
# 1) explanatory analysis
# 2) predictive analysis

# it can be linear regression or non-linear regression

# single explanatory variable = simple regression
# many explanatory variables = multiple regression
# single dependent variable = univariate regression
# many dependent varibales = multivariate regression

# thus we can have 'multiple multivariate regression'

# it the response variable is dichotomous, we use logistic regression

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# GET THE DATA
BHNames = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
		   'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'medv']

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'

data = pd.read_csv(url, delim_whitespace=True, names=BHNames)

print(data.info())
print(data.describe().transpose())

data.boxplot(column=BHNames)
plt.show()

# do min-max (feature) scaling on the data
scaler = MinMaxScaler()
dataScaled = scaler.fit_transform(data) # returns a np array
dataScaled = pd.DataFrame(dataScaled, columns=BHNames)
print(dataScaled.describe().transpose())

corrData = dataScaled.corr(method='pearson')
# with pd.option_context('display.max_rows', None,'display.max_columns', corrData.shape[1]):
print(corrData)

plt.matshow(corrData)
plt.xticks(range(len(corrData.columns)), corrData.columns)
plt.yticks(range(len(corrData.columns)), corrData.columns)
plt.colorbar()
plt.show()
