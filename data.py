from keras.utils import np_utils
import scipy.io
import numpy as np

np.random.seed(0)


def load_data():

    rows, cols = 28, 28
    nb_classes = 10

    DATA_DIR = 'notMNIST_small.mat'
    mat = scipy.io.loadmat(DATA_DIR)

    X = mat['images']
    Y = mat['labels']

    # Move last column to front
    X = np.rollaxis(X, 2)

    # Reshape and format input
    X = X.reshape(X.shape[0], rows, cols, 1)
    X = X.astype('float32')
    X /= 255.0

    # Hot encoding
    Y = Y.astype(int)
    Y = np_utils.to_categorical(Y, nb_classes)

    # Divide into test and train sets
    perm = np.random.permutation(X.shape[0])

    train_size = 13000

    X_train = X[perm[:train_size]]
    X_test = X[perm[train_size:]]

    Y_train = Y[perm[:train_size]]
    Y_test = Y[perm[train_size:]]

    return (X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':

    load_data()
    pass
