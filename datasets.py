
# Note: this file is derived from:
#     https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
#     (Apache 2.0 License, Copyright 2017, Guillaume Chevalier and Yu Zhao)
# Which in turns had some derived code from there, regarding loading the Opportunity dataset:
#     https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
# There is also some code derived from:
#     https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
#     (MIT License, Copyright 2017, Guillaume Chevalier)

import os
import pickle
import time

import numpy as np

from data.sliding_window import sliding_window


class Dataset:
    LABELS = list()

    def __init__(self, verbose=False):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.load_train_test()

        if verbose:
            print("Shapes for [self.X_train, self.Y_train, self.X_test, self.Y_test]:")
            for ds in [self.X_train, self.Y_train, self.X_test, self.Y_test]:
                print(ds.shape)


class UCIHARDataset(Dataset):
    NAME = "UCIHAR"

    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    def __init__(self, verbose=False):
        super().__init__(verbose)

    def load_train_test(self):
        # Those are separate normalised input features for the neural network
        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]
        DATA_PATH = "data/"
        DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"

        TRAIN = "train/"
        TEST = "test/"

        X_train_signals_paths = [
            DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
        X_test_signals_paths = [
            DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]
        Y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
        Y_test_path = DATASET_PATH + TEST + "y_test.txt"

        self.X_train = self.load_X(X_train_signals_paths)
        self.X_test = self.load_X(X_test_signals_paths)
        self.Y_train = self.load_Y(Y_train_path)
        self.Y_test = self.load_Y(Y_test_path)


    def load_X(self, X_signals_paths):
        """Load "X" (the neural network's training and testing inputs).

        Given attribute (train or test) of feature, read all 9 features into an
        np ndarray of shape [sample_sequence_idx, time_step, feature_num]
            argument:   X_signals_paths str attribute of feature: 'train' or 'test'
            return:     np ndarray, tensor of features
        """
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'rb')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_Y(self, y_path):
        """Load "Y" (the neural network's training and testing outputs).

        Read Y file of values to be predicted
            argument: y_path str attibute of Y: 'train' or 'test'
            return: Y ndarray / tensor of the 6 one_hot labels of each sample
        """
        file = open(y_path, 'rb')
        # Read dataset from disk, dealing with text file's syntax
        Y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return self.one_hot(Y_ - 1, n_classes=6)

    def one_hot(self, Y_, n_classes):
        """Function to encode output labels from number indexes.

        e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
            argument:   Y_, a 2D np.ndarray containing features by index
            return:     Y_, a 2D np.ndarray containing one-hot features
        """
        Y_ = Y_.reshape(len(Y_))
        return np.eye(n_classes)[np.array(Y_, dtype=np.int32)]  # Returns FLOATS


class OpportunityDataset(Dataset):
    NAME = "Opportunity"

    def __init__(self, verbose=False):
        super().__init__(verbose)

    def load_train_test(self):
        # Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
        NB_SENSOR_CHANNELS = 113
        NB_SENSOR_CHANNELS_WITH_FILTERING = 149

        # Hardcoded number of classes in the gesture recognition problem
        NUM_CLASSES = 18

        # Hardcoded length of the sliding window mechanism employed to segment the data
        SLIDING_WINDOW_LENGTH = 24

        # Length of the input sequence after convolutional operations
        FINAL_SEQUENCE_LENGTH = 8

        # Hardcoded step of the sliding window mechanism employed to segment the data
        SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
        SLIDING_WINDOW_STEP_SHORT = SLIDING_WINDOW_STEP

        # Batch Size
        BATCH_SIZE = 100

        # Number filters convolutional layers
        NUM_FILTERS = 64

        # Size filters convolutional layers
        FILTER_SIZE = 5

        # Number of unit in the long short-term recurrent layers
        NUM_UNITS_LSTM = 128

        print("Loading data...")
        X_train, Y_train, X_test, Y_test = self.load_dataset('data/oppChallenge_gestures.data')

        assert (NB_SENSOR_CHANNELS_WITH_FILTERING == X_train.shape[1] or NB_SENSOR_CHANNELS == X_train.shape[1])

        def opp_sliding_window(data_x, data_y, ws, ss):
            data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
            data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
            data_x, data_y = data_x.astype(np.float32), one_hot(data_y.reshape(len(data_y)).astype(np.uint8))
            print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, Y_test.shape))
            return data_x, data_y

        # Sensor data is segmented using a sliding window mechanism
        self.X_test, self.Y_test = opp_sliding_window(X_test, Y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP_SHORT)
        self.X_train, self.Y_train = opp_sliding_window(X_train, Y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

    def load_dataset(self, filename):

        f = file(filename, 'rb')
        data = pickle.load(f)
        f.close()

        X_train, Y_train = data[0]
        X_test, Y_test = data[1]

        print(" ..from file {}".format(filename))
        print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # The targets are casted to int8 for GPU compatibility.
        Y_train = Y_train.astype(np.uint8)
        Y_test = Y_test.astype(np.uint8)

        return X_train, Y_train, X_test, Y_test
