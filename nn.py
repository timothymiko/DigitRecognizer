import pandas as pd
import numpy as np
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet


def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('hidden3', layers.DenseLayer),
            ('output', layers.DenseLayer)],

        input_shape=(None, 1, 28, 28),
        conv1_num_filters=7,
        conv1_filter_size=(3, 3),
        conv1_nonlinearity=lasagne.nonlinearities.rectify,

        pool1_pool_size=(2, 2),

        conv2_num_filters=12,
        conv2_filter_size=(2, 2),
        conv2_nonlinearity=lasagne.nonlinearities.rectify,

        hidden3_num_units=1000,
        output_num_units=10,
        output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=0.0001,
        update_momentum=0.9,

        max_epochs=n_epochs,
        verbose=1,
    )
    return net1


if __name__ == "__main__":

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train = train_df.iloc[:, 1:].values
    target = train_df[[0]].values.ravel()

    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    target = target.astype(np.uint8)
    test = np.array(test_df).reshape((-1, 1, 28, 28)).astype(np.uint8)

    cnn = CNN(25)

    # Roughly 97% accuracy currently
    # Supposedly, 99% accuracy can be achieved when tuned correctly
    cnn.fit(train, target)

