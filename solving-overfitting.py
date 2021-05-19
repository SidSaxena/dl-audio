import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "data_10.json"

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y

# def predict(model, X, y):
    X = X[np.newaxis, ...]
    prediction = model.predict(X) # X -> (1, 130, 13, 1)
    predicted_index = np.argmax(prediction, axis=1) # [3]
    print("Excpected index: {}, Predicted index: {}".format(y, predicted_index))


def plot_history(history):

    fig, axes = plt.subplots(2)

    # create accuracy subplot 
    axes[0].plot(history.history['accuracy'], label='train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='test accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='best')
    axes[0].set_title('Accuracy Eval')
    
    # create error subplot 
    axes[1].plot(history.history['loss'], label='train error')
    axes[1].plot(history.history['val_loss'], label='test error')
    axes[1].set_ylabel('Error')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='best')
    axes[1].set_title('Error Eval')
    
    plt.show()

if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        
        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        
        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        
        # output layer
        keras.layers.Dense(11, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

    plot_history(history)
    
    # X = X_test[99]
    # y = y_test[99]
    # predict(model, X, y)