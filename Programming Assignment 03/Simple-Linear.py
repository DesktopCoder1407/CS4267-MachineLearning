# Importing libraries
import datetime
import matplotlib.pyplot as plt
import pandas as pd

startTime = datetime.datetime.now()

# Importing the dataset
dataset = pd.read_csv('data/Simple-Linear-Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression().fit(X_train, y_train)

# Output R^2 Score
print('R^2 score for the test set: ', regression.score(X_test, y_test))
print('R^2 score for the train set: ', regression.score(X_train, y_train))
print('Time to train: ', datetime.datetime.now() - startTime)

# Visualising the results of the regression
plt.scatter(X, y, color='red', label='Raw Data')
plt.plot(X, regression.predict(X), color='blue', label='Linear Regression Line')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
