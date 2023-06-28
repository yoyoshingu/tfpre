
# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model
# should be trained to predict the next 24 observations of the 7 variables.
#
# ==============================================================================

# =========== 합격 기준 가이드라인 공유 ============= #
# 2021년 7월 1일 신규 출제된 문제                     #
# 5/5가 잘 나오지 않으므로 모델 많이 만들어 둘 것     #
# =================================================== #
# 문제명: Category 5 - household electric power consumption
# val_loss: 0.053
# val_mae: 0.053 (val_loss와 동일)
# =================================================== #
# =================================================== #

# ABOUT THE DATASET
#
# Original Source:
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption
#
# The original 'Individual House Hold Electric Power Consumption Dataset'
# has Measurements of electric power consumption in one household with
# a one-minute sampling rate over a period of almost 4 years.
#
# Different electrical quantities and some sub-metering values are available.
#
# For the purpose of the examination we have provided a subset containing
# the data for the first 60 days in the dataset. We have also cleaned the
# dataset beforehand to remove missing values. The dataset is provided as a
# csv file in the project.
#
# The dataset has a total of 7 features ordered by time.
# ==============================================================================
#
# INSTRUCTIONS
#
# Complete the code in following functions:
# 1. windowed_dataset()
# 2. solution_model()
#
# The model input and output shapes must match the following
# specifications.
#
# 1. Model input_shape must be (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7),
#    since the testing infrastructure expects a window of past N_PAST = 24
#    observations of the 7 features to predict the next 24 observations of
#    the same features.
#
# 2. Model output_shape must be (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7)
#
# 3. DON'T change the values of the following constants
#    N_PAST, N_FUTURE, SHIFT in the windowed_dataset()
#    BATCH_SIZE in solution_model() (See code for additional note on
#    BATCH_SIZE).
# 4. Code for normalizing the data is provided - DON't change it.
#    Changing the normalizing code will affect your score.
#
# HINT: Your neural network must have a validation MAE of approximately 0.055 or
# less on the normalized validation dataset for top marks.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure.
#
# WARNING: If you are using the GRU layer, it is advised not to use the
# 'recurrent_dropout' argument (you can alternatively set it to 0),
# since it has not been implemented in the cuDNN kernel and may
# result in much longer training times.
import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the un windowed time series dataset to a
# windowed dataset so as to prepare it for training and validation.
# A window of features are constructed by shifting the window's starting
# position forward, one at a time (indicated by shift=1).
# For a window of 'n_past' number of observations of all time indexed variables in
# the dataset, the target for the window is the next 'n_future' number
# of observations of these variables, after the end of the window.
# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    # This line converts the dataset into a windowed dataset where a
    # window consists of both the observations to be included as features
    # and the targets.
    # Don't change the shift parameter. The test windows are
    # created with the specified shift and hence it might affect your
    # scores. Calculate the window size so that based on
    # the past 24 observations
    # (observations at time steps t=1,t=2,...t=24) of the 7 variables
    # in the dataset, you predict the next 24 observations
    # (observations at time steps t=25,t=26....t=48) of the 7 variables
    # of the dataset.
    # Hint: Each window should include both the past observations and
    # the future observations which are to be predicted. Calculate the
    # window size based on n_past and n_future.
    ds = ds.window(size=  # YOUR CODE HERE,
                   shift = shift,
                           drop_remainder = True)
    # This line converts the windowed dataset into a tensorflow dataset.
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    # Now each window in the dataset has n_past and n_future observations.
    # This line maps each window to the form (n_past observations,
    # n_future observations) in the format needed for training the model.
    # Note: You can use a lambda function to map each window in the
    # dataset to it's respective (features, targets).
    ds = ds.map(
        # YOUR CODE HERE
    )
    return ds.batch(batch_size).prefetch(1)


# This function loads the data from csv file, normalizes the data and
# splits the dataset into train and validation data. It also uses the
# 'windowed_dataset' to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network.
# This function returns the trained model.
#
# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()
    # Reads the dataset from the csv.
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)
    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = len(df.columns)
    # Normalizes the data
    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    # Splits the data into training and validation sets.
    x_train = data[:split_time]
    x_valid = data[split_time:]
    # DO NOT CHANGE 'BATCH_SIZE' IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS
    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 24  # DO NOT CHANGE THIS
    # Number of future time steps which are to be predicted.
    N_FUTURE = 24  # DO NOT CHANGE THIS
    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS
    # Code to create windowed train and validation datasets.
    # Complete the code in windowed_dataset.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    # Code to define your model.
    model = tf.keras.models.Sequential([
        # ADD YOUR LAYERS HERE.
        # If you don't follow the instructions in the following comments,
        # tests will fail to grade your code:
        # Whatever your first layer is, the input shape will be
        # (BATCH_SIZE, N_PAST = 24, N_FEATURES = 7)
        # The model must have an output shape of
        # (BATCH_SIZE, N_FUTURE = 24, N_FEATURES = 7).
        # Make sure that there are N_FEATURES = 7 neurons in the final dense
        # layer since the model predicts 7 features.
        tf.keras.layers.Dense(N_FEATURES)
    ])
    # Code to train and compile the model
    optimizer =  # YOUR CODE HERE
    model.compile(
        # YOUR CODE HERE
    )
    model.fit(
        # YOUR CODE HERE
    )
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("model.h5")
