
# ==============================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
#
# COMPUTER VISION WITH CNNs
#
# Create and train a classifier to classify images between three categories
# of beans using the beans dataset.
# ==============================================================================
# ABOUT THE DATASET
#
# Beans dataset has images belonging to 3 classes as follows:
# 2 disease classes (Angular leaf spot, bean rust)
# 1 healthy class (healthy).
# The images are of different sizes and have 3 channels.
# ==============================================================================
#
# INSTRUCTIONS
#
# We have already divided the data for training and validation.
#
# Complete the code in following functions:
# 1. preprocess()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
# 1. The input shape of your model must be (300,300,3), because the testing
#    infrastructure expects inputs according to this specification. You must
#    resize all the images in the dataset to this size while pre-processing
#    the dataset.
# 2. The last layer of your model must be a Dense layer with 3 neurons
#    activated by softmax since this dataset has 3 classes.
#
# HINT: Your neural network must have a validation accuracy of approximately
# 0.75 or above on the normalized validation dataset for top marks.
#


import tensorflow as tf
import tensorflow_datasets as tfds

# Use this constant wherever necessary
IMG_SIZE = 300

# This function normalizes and resizes the images.

# COMPLETE THE CODE IN THIS FUNCTION
def preprocess(image, label):
    # RESIZE YOUR IMAGES HERE (HINT: After resizing the shape of the images
    # should be (300, 300, 3).
    # NORMALIZE YOUR IMAGES HERE (HINT: Rescale by 1/.255))
    image /= 255
    image = tf.image.resize(image, size=(300,300))
    return image, label


# This function loads the data, normalizes and resizes the images, splits it into
# train and validation sets, defines the model, compiles it and finally
# trains the model. The trained model is returned from this function.

# COMPLETE THE CODE IN THIS FUNCTION.
def solution_model():
    # Loads and splits the data into training and validation splits using tfds.
    (ds_train, ds_validation), ds_info = tfds.load(
        name='beans',
        split=['train', 'validation'],
        as_supervised=True,
        with_info=True)

    BATCH_SIZE = 32 # YOUR CODE HERE

    # Resizes and normalizes train and validation datasets using the
    # preprocess() function.
    # Also makes other calls, as evident from the code, to prepare them for
    # training.
    ds_train = ds_train.map(preprocess).cache().shuffle(
        ds_info.splits['train'].num_examples).batch(BATCH_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE)
    ds_validation = ds_validation.map(preprocess).batch(BATCH_SIZE).cache(

    ).prefetch(tf.data.experimental.AUTOTUNE)

    # Code to define the model
    model = tf.keras.models.Sequential([

        # ADD LAYERS OF THE MODEL HERE

        # If you don't adhere to the instructions in the following comments,
        # tests will fail to grade your model:
        # The input layer of your model must have an input shape of
        # (300,300,3).
        # Make sure that your last layer has 3 (number of classes) neurons
        # activated by softmax.

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])

    # Code to compile and train the model
    model.compile(
        # YOUR CODE HERE
        optimizer='rmsprop',  # ,또는  adam
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        ds_train,
        epochs=10, # orginal 15
        validation_data=ds_validation
    )
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
