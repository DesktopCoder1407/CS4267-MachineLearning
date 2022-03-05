# Importing libraries
import numpy as np
import datetime
import pandas as pd

startTime = datetime.datetime.now()

# Importing the dataset (X is Product_1, Product_2, and Product_3)
dataset = pd.read_csv('data/Multiple-Linear-Dataset.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression().fit(X_train, y_train)

# Output R^2 Score
print('R^2 score for the test set: ', regression.score(X_test, y_test))
print('R^2 score for the train set: ', regression.score(X_train, y_train))
print('Time to train: ', datetime.datetime.now() - startTime, '\n')

# Extra Results
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
