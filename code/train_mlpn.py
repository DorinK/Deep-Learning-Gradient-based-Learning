import mlpn as mlp
import numpy as np
import random
from utils import load_train_set, get_common_features, load_validation_set

STUDENT = {'name': 'Dorin Keshales'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.

    # Get the common features of the train set
    common_features = get_common_features()

    # Building a numpy vector for the example's features
    features_vec = np.zeros(len(common_features))

    for feature in features:
        if feature in common_features:
            features_vec[common_features.index(feature, 0, len(common_features))] += 1

    return features_vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0

    for label, features in dataset:

        # YOUR CODE HERE
        x = feats_to_vec(features)  # convert features to a vector.
        y = label  # convert the label to number if needed.

        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if mlp.predict(x, params) == y:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = mlp.loss_and_gradients(x, y, params)
            cum_loss += loss

            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for element, gradient in zip(params, grads):
                element -= learning_rate * gradient

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    train, features_size, labels_size = load_train_set('bigrams')
    validation = load_validation_set('bigrams')
    learning_rate = 0.001
    dims = [700, 100, 20, 6]
    params = mlp.create_classifier(dims)
    trained_params = train_classifier(train, validation, 10, learning_rate, params)
