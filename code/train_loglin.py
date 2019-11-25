import loglinear as ll
import numpy as np
import random
from utils import load_train_set, get_common_features, load_validation_set, load_test_set, index_to_language

STUDENT = {'name': 'Dorin Keshales',
           'ID': '313298424'}


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
        if ll.predict(x, params) == y:
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
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss

            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            W, b = params
            gW, gb = grads
            W -= learning_rate * gW
            b -= learning_rate * gb

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def test_predictions(test_data, params):
    import os

    # Clearing the content of the file if it already exists; Otherwise, creating the file.
    if os.path.exists("./test.pred"):
        os.remove("./test.pred")
    f = open("./test.pred", "a+")

    # For each example we find calculate the model prediction
    for label, features in test_data:
        x = feats_to_vec(features)

        # Get the index of the max log-probability
        prediction = ll.predict(x, params)

        # Write to the file
        f.write("{0}\n".format(index_to_language(prediction)))

    # Close the file.
    f.close()


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    # For Bigrams:
    train, features_size, labels_size = load_train_set('bigrams')
    validation = load_validation_set('bigrams')
    learning_rate = 0.005
    params = ll.create_classifier(features_size, labels_size)
    trained_params = train_classifier(train, validation, 10, learning_rate, params)
    # test = load_test_set()
    # test_predictions(test, trained_params)

    # For Unigrams:
    # train, features_size, labels_size = load_train_set('unigrams')
    # validation = load_validation_set('unigrams')
    # learning_rate = 0.007
    # params = ll.create_classifier(features_size, labels_size)
    # trained_params = train_classifier(train, validation, 100, learning_rate, params)
