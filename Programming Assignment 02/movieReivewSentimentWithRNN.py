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

#analyzeData()
# --PREPROCESS DATA--
#Pad or truncate all data so it is the length of maxlen
maxlen = 200 #The length of the majority of the reviews
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print('X_train shape after padding:', X_train.shape)
print('X_test shape after padding:', X_test.shape)