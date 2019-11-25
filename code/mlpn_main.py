import mlpn as mlp
from train_mlpn import train_classifier
from utils import load_train_set, load_validation_set

STUDENT = {'name': 'Dorin Keshales',
           'ID': '313298424'}

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...

    # Allowing the user to decide how many hidden layers he wants and define their sizes.
    dimensions = [700]
    d = input(
        "Enter the dimensions of the hidden layers separated by comma.\nIf you don't want hidden layers press enter.\n")
    if len(d) != 0:
        d = d.split(",")
        for dim in d:
            dimensions.append(int(dim))
    dimensions.append(6)

    train, features_size, labels_size = load_train_set('bigrams')
    validation = load_validation_set('bigrams')
    learning_rate = 0.001
    params = mlp.create_classifier(dimensions)
    trained_params = train_classifier(train, validation, 10, learning_rate, params)
