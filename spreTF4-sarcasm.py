
# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# NLP QUESTION
#
# For this task you will build a classifier for the sarcasm dataset
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown
# It will be tested against a number of sentences that the network hasn't previously seen
# And you will be scored on whether sarcasm was correctly detected in those sentences

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 4 - sarcasm
# val_loss: 0.3650
# val_acc: 0.83
# =================================================== #
# =================================================== #


import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    # YOUR CODE HERE

    # Collect sentences and labels into the lists
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_size = 20000

    # Split the sentences
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    # Split the labels
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    import numpy as np
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    vocab_size = 10000
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"

    # Initialize the Tokenizer class
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    # Generate the word index dictionary
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Generate and pad the training sequences
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Generate and pad the testing sequences
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Convert the labels lists into numpy arrays
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    import tensorflow as tf

    # Parameters
    embedding_dim = 16
    lstm_dim = 32
    dense_dim = 24


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
        tf.keras.layers.Dense(dense_dim, activation='relu'),

        # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Print the model summary
    model.summary()

    checkpointpath='sarcasm.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpointpath, save_weights_only=True,
        save_best_only=True, monitor='val_loss', verbose=1)

    NUM_EPOCHS = 10

    # Train the model
    history = model.fit(training_padded, training_labels, epochs=NUM_EPOCHS,
                        validation_data=(testing_padded, testing_labels),
                        callbacks=[checkpoint])
    import matplotlib.pyplot as plt
    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()

    # Plot the accuracy and loss history
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    checkpointpath = 'sarcasm.ckpt'
    model.load_weights(checkpointpath)
    model.save("TF4-sarcasm.h5")

