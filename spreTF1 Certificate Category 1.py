# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]


import tensorflow as tf
import numpy as np
from tensorflow import keras

def solution_model():
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)

    # (2) 모델 정의 (Sequential)

    
    # (3) 컴파일 (compile)


    # (4) 학습 (fit)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    new_y = 20.0
    prediction = model.predict([new_y])[0]
    print(prediction)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

    # model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    # model.compile(optimizer='sgd', loss='mean_squared_error')
    #
    # xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    # ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    #
    # model.fit(xs, ys, epochs=500, verbose=0)
    # print(model.predict([10.0]))

