





# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the 
# input shape should be the native size of the Fashion MNIST dataset which is 
# 28x28 monochrome. Do not resize the data. YOur input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 2 - fashion mnist
# val_loss: 0.33
# val_acc: 0.89
# =================================================== #
# =================================================== #



import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalize the pixel values
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([

        # Add convolutions and max pooling
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Add the same layers as before
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Use same settings
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    print(f'\nMODEL TRAINING:')
    model.fit(training_images, training_labels, epochs=5)

    # Evaluate on the test set
    print(f'\nMODEL EVALUATION:')
    test_loss = model.evaluate(test_images, test_labels)
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF2-fashion-mnist.h5")
