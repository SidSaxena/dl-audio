import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.layers.advanced_activations import ReLU


def generate_dataset(num_samples, test_size):

    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([i[0] + i[1] for i in x])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test 

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = generate_dataset(5000, 0.33)
    # print(f'x_test: \n {X_test}')
    # print(f'y_test: \n {y_test}')

    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
        
    ])
    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser, loss='MSE')

    # train
    model.fit(X_train, y_train, epochs=100)

    # evaluate
    print('Model Evaluation:')
    model.evaluate(X_test, y_test, verbose=1)

    # predict
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print('\nSome Predictions:')
    for d, p in zip(data, predictions):
        print(f'{d[0]} + {d[1]} = {p[0]}')
