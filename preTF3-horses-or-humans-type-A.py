
# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# This task requires you to create a classifier for horses or humans using
# the provided dataset. 
#
# Please make sure your final layer has 2 neurons, activated by softmax 
# as shown. Do not change the provided output layer, or tests may fail.
#
# IMPORTANT: Please note that the test uses images that are 300x300 with 
# 3 bytes color depth so be sure to design your input layer to accept
# these, or the tests will fail.
#

# =========== 합격 기준 가이드라인 공유 ============= #
# val_loss 기준에 맞춰 주시는 것이 훨씬 더 중요 #
# val_loss 보다 조금 높아도 상관없음. (언저리까지 OK) #
# =================================================== #
# 문제명: Category 3 - Horses Or Humans type A
# val_loss: 0.028
# val_acc: 0.98
# =================================================== #
# =================================================== #


import tensorflow_datasets as tfds
import tensorflow as tf


dataset_name = 'horses_or_humans'
dataset, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

def preprocess(data):
    # YOUR CODE HERE

    return x, y


def solution_model():
    train_dataset = dataset.map(preprocess).batch(32)

    model = Sequential([
    
    # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("TF3-horses-or-humans-type-A.h5")

