import json
import numpy as np

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = 'data_10.json'

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


if __name__ == '__main__':
    # create train, validation and test sets


    # build the cnn 

    # compile the network

    # train the cnn

    # evaluate the cnn on the test set

    # make a prediction on a sample