
# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets question
#
# For this task you will train a classifier for Iris flowers using the Iris dataset
# The final layer in your neural network should look like: tf.keras.layers.Dense(3, activation=tf.nn.softmax)
# The input layer will expect data in the shape (4,)
# We've given you some starter code for preprocessing the data
# You'll need to implement the preprocess function for data.map

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 2 - iris
# val_loss: 0.14
# val_acc: 0.93
# =================================================== #
# =================================================== #


import tensorflow as tf
import tensorflow_datasets as tfds

#data = tfds.load("iris", split=tfds.Split.TRAIN.split(tfds.percent[:80]))
train_dataset = tfds.load('iris', split='train[:80%]')
valid_dataset = tfds.load('iris', split='train[80%:]')

def preprocess(data):
    # YOUR CODE HERE
    # Should return features and one-hot encoded labels
    x = data['features']
    y = data['label']
    y = tf.one_hot(y, 3)
    return x, y

def solution_model():
    train_data = train_dataset.map(preprocess).batch(10)
    valid_data = valid_dataset.map(preprocess).batch(10)
    model1 = tf.keras.models.Sequential([
        # input_shape는 X의 feature 갯수가 4개 이므로 (4, )로 지정합니다.
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        # Classification을 위한 Softmax, 클래스 갯수 = 3개
        tf.keras.layers.Dense(3, activation='softmax'),
    ])

    from tensorflow.keras.layers import Dense
    model = tf.keras.models.Sequential([
        # input_shape는 X의 feature 갯수가 4개 이므로 (4, )로 지정합니다.
        Dense(512, activation='relu', input_shape=(4,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # Classification을 위한 Softmax, 클래스 갯수 = 3개
        Dense(3, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_data, validation_data=(valid_data), epochs=20)
    

    # YOUR CODE TO TRAIN A MODEL HERE


    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-iris.h5")
