
# 본 코드는 데이터셋을 임의로 다운로드 받기 위한 코드이므로, 시험 보실 때는 포함하지 않습니다.
get_ipython().system('wget -O Weekly_U.S.Diesel_Retail_Prices.csv https://www.dropbox.com/s/eduk281didil1km/Weekly_U.S.Diesel_Retail_Prices.csv?dl=1 ')


# In[ ]:


# ==============================================================================
# TIME SERIES QUESTION
#
# Build and train a neural network to predict the time indexed variable of
# the univariate US diesel prices (On - Highway) All types for the period of
# 1994 - 2021.
# Using a window of past 10 observations of 1 feature , train the model
# to predict the next 10 observations of that feature.
#
# ==============================================================================

# =========== 합격 기준 가이드라인 공유 ============= #
# =================================================== #
# 문제명: Category 5 - Weekly US Retail Price
# val_mae: 0.026 혹은 그 이하
# =================================================== #
# =================================================== #

import pandas as pd
import tensorflow as tf

# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)
    
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS
    
    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))
    
    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS
    N_PAST = 10  # DO NOT CHANGE THIS
    N_FUTURE = 10  # DO NOT CHANGE THIS
    SHIFT = 1  # DO NOT CHANGE THIS

    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    # Code to define your model.
    model = tf.keras.models.Sequential([
  
        
        
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

