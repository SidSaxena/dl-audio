import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_dataset(num_samples, test_size):

    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([i[0] + i[1] for i in x])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test 

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_dataset(10, 0.2)
    # print(f'x_test: \n {X_test}')
    # print(f'y_test: \n {y_test}')

    # build model
    model = tf.keras.Sequential()