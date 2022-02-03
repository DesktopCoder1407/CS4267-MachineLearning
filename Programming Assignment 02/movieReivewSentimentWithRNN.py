import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
from keras import layers, models, losses, optimizers
from keras_preprocessing.sequence import pad_sequences

#Import data
vocab_size = 5000 #keep this many (most frequently appearing) words.
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# --ANALYZE DATA--
def analyzeData():
    #View dataset (Training set is perfectly balanced)
    print('Number of training samples:', len(y_train)) #25000
    print('Number of positive sambles:', sum(y_train)) #12500
    print('Number of test samples:', len(y_test)) #25000

    #View a training sample
    print(X_train[0])

    #View the training sample as a set of words
    word_index = imdb.get_word_index()
    index_word = {index: word for word, index in word_index.items()}
    print([index_word.get(i, '') for i in X_train[0]])

    #Analyze the length of each sample
    review_lengths = [len(x) for x in X_train]
    import matplotlib.pyplot as plt
    plt.hist(review_lengths, bins=10)
    plt.show()
analyzeData()

# --PREPROCESS DATA--
#Pad or truncate all data so it is the length of maxlen
maxlen = 200 #The length of the majority of the reviews
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print('X_train shape after padding:', X_train.shape)
print('X_test shape after padding:', X_test.shape)

# --DEVLOP A SIMPLE LSTM (Long Short-Term Memory) NETWORK--
def createSimpleLSTM():
    tf.random.set_seed(42)
    model = models.Sequential()

    #Embed our input (with a size of vocab_size) into dense vectors of size embedding_size
    embedding_size = 32
    model.add(layers.Embedding(vocab_size, embedding_size))

    #Recurrent, LSTM layer with 50 nodes
    model.add(layers.LSTM(50))

    #Output layer with sigmoid activation function (sigmoid chosen because the problem is binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))

    #Model summary. Shows all layers.
    print(model.summary())

    #Compile model with Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Train & validate the model. Batch size of batch_size, Epochs of n_epoch
    batch_size = 64
    n_epoch = 3
    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test))

    #Evaluate the classification accuracy
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print('Test accuracy:', acc)
createSimpleLSTM()

# --PERFORMANCE BOOST WITH MULTIPLE LSTM LAYERS--
def createMultipleLSTM():
    #Embed our input (with a size of vocab_size) into dense vectors of size embedding_size
    tf.random.set_seed(42)
    embedding_size = 32

    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_size)) #Same as first model
    model.add(layers.LSTM(50, return_sequences=True, dropout=0.2)) #50 nodes, return_sequences=True 
        #to feed the entire output sequence to the second LSTM, dropout of 20% to reduce overfitting.
    model.add(layers.LSTM(50, dropout=0.2)) #50 nodes, dropout of 20% to reduce overfitting.
    model.add(layers.Dense(1, activation='sigmoid')) #Same as first model

    #Model summary. Shows all layers.
    print(model.summary())

    #Compile model with Adam
    optimizer = tf.keras.optimizers.Adam(lr=0.003)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    #Train & validate the model. Batch size of batch_size, Epochs of n_epoch
    batch_size = 64
    n_epoch = 7
    model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test))

    #Evaluate the test accuracy
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print('Test accuracy with stacked LSTM:', acc)
createMultipleLSTM()