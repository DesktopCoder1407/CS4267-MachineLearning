from preprocess import *
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

start_time = time.time()

data = generate_one_hot_data(TRIMMED_DATA_PATH)
x = data.drop(columns=['sellingprice'])
y = data['sellingprice']

model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

model.fit(x_train, y_train)

training_time = time.time() - start_time
test_accuracy = r2_score(y_test, model.predict(x_test))
train_accuracy = r2_score(y_train, model.predict(x_train))
mse_loss = mean_squared_error(y, model.predict(x))
root_mse_loss = sqrt(mse_loss)

print(f'The accuracy of the test set is {test_accuracy}.')
print(f'The accuracy of the training set is {train_accuracy}.')
print(f'The mean squared error loss is {mse_loss}.')
print(f'The root mean squared error loss is {root_mse_loss}.')
print(f'It took {training_time / 60} minutes to train the model.')
