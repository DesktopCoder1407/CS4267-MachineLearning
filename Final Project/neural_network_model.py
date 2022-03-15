from preprocess import *
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

start_time = time.time()

data = generate_one_hot_data(TRIMMED_DATA_PATH)
x = data.drop(columns=['sellingprice'])
y = data['sellingprice']

model = Sequential([
    Dense(125, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error',
              metrics=['mean_squared_error'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

model.fit(x_train, y_train, epochs=250, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
])

training_time = time.time() - start_time
test_accuracy = r2_score(y_test, model.predict(x_test))
train_accuracy = r2_score(y_train, model.predict(x_train))
mse_loss = mean_squared_error(y, model.predict(x))

print(f'The accuracy of the test set is {test_accuracy}.')
print(f'The accuracy of the training set is {train_accuracy}.')
print(f'The mean squared error loss is {mse_loss}.')
print(f'It took {training_time / 60} minutes to train the model.')


def find_optimal_model():
    # Neural Network Parameters
    hidden_1 = hp.HParam('hidden_layer_1', hp.Discrete([25, 50, 75, 100, 125, 150]))  # Size of First Hidden Layer
    hidden_2 = hp.HParam('hidden_layer_2', hp.Discrete([25, 50, 75, 100, 125]))  # Size of Second Hidden Layer
    learning_rate = hp.HParam('learning_rate', hp.Discrete([.001, .005, .01, .03, .05, .07, .09, .1]))  # Learning Rate

    def train_test_model(hparams, logdir):
        model = Sequential([
            Dense(hparams[hidden_1], activation='relu'),
            Dense(hparams[hidden_2], activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(hparams[learning_rate]), loss='mean_squared_error',
                      metrics=['mean_squared_error'])

        model.fit(x_train, y_train, epochs=250, callbacks=[
                      tf.keras.callbacks.TensorBoard(logdir),
                      hp.KerasCallback(logdir, hparams),
                      tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
                  ])

        _, mse = model.evaluate(x_test, y_test)
        r2 = r2_score(y_test, model.predict(x_test))
        return mse, r2

    def run(hparams, logdir):
        with tf.summary.create_file_writer(logdir).as_default():
            hp.hparams_config(
                hparams=[hidden_1, hidden_2, learning_rate],
                metrics=[hp.Metric('mean_squared_error', display_name='mse'),
                         hp.Metric('r2', display_name='r2')],
            )
            mse, r2 = train_test_model(hparams, logdir)
            tf.summary.scalar('mean_squared_error', mse, step=1)
            tf.summary.scalar('r2', r2, step=1)

    session_num = 0
    for h1 in hidden_1.domain.values:
        for h2 in hidden_2.domain.values:
            for lr in learning_rate.domain.values:
                hparams = {
                    hidden_1: h1,
                    hidden_2: h2,
                    learning_rate: float("%.3f" % float(lr)),
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run(hparams, 'logs/hparam_tuning/' + run_name)
                session_num += 1


# if __name__ == '__main__':
#     find_optimal_model()
