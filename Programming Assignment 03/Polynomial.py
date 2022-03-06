# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('data/Polynomial-Dataset.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset with degree=2
poly_reg_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_reg_2.fit_transform(X)
lin_reg_2 = LinearRegression().fit(X_poly_2, y)

# Fitting Polynomial Regression to the dataset with degree=3
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg_3.fit_transform(X)
lin_reg_3 = LinearRegression().fit(X_poly_3, y)

# Fitting Polynomial Regression to the dataset with degree=4
poly_reg_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_reg_4.fit_transform(X)
lin_reg_4 = LinearRegression().fit(X_poly_4, y)

# Outputting the result requested from the assignment: Employee's salary at level 6.5
print('''Linear Regression | Employee's salary at level 6.5:\t\t\t\t\t''', lin_reg.predict([[6.5]]))
print('''Polynomial Regression of Degree 2 | Employee's salary at level 6.5:\t''',
      lin_reg_2.predict(poly_reg_2.fit_transform([[6.5]])))
print('''Polynomial Regression of Degree 3 | Employee's salary at level 6.5:\t''',
      lin_reg_3.predict(poly_reg_3.fit_transform([[6.5]])))
print('''Polynomial Regression of Degree 4 | Employee's salary at level 6.5:\t''',
      lin_reg_4.predict(poly_reg_4.fit_transform([[6.5]])))

# Visualising the Linear Regression and Polynomial Regression results
plt.scatter(X, y, color='black', label='Raw Data')
plt.plot(X, lin_reg.predict(X), color='orange', label='Linear')
plt.plot(X, lin_reg_2.predict(X_poly_2), color='blue', label='Polynomial of Degree 2')
plt.plot(X, lin_reg_3.predict(X_poly_3), color='green', label='Polynomial of Degree 3')
plt.plot(X, lin_reg_4.predict(X_poly_4), color='red', label='Polynomial of Degree 4')
plt.title('Estimation of Salary based on Employee Level')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.legend()
plt.show()
