import numpy as np

STUDENT = {'name': 'Dorin Keshales',
           'ID': '313298424'}


def classifier_output(x, params):
    # YOUR CODE HERE.

    global input_lst
    input_lst = [x]

    # linear operation on the first two params
    result = np.dot(x, params[0]) + params[1]

    it = iter(params[2::1])
    rest_params = list(zip(it, it))

    for W, b in rest_params:
        # Saving copy of the hidden layer input - before the tanh
        input_lst.append(result.copy())

        # Using tanh as activation function
        result = np.tanh(result)

        # Saving copy of the hidden layer output - after the tanh
        input_lst.append(result.copy())

        # linear operation on the current two params
        result = np.dot(result, W) + b

    # normalizing before the softmax and then activating the softmax
    result -= result.max()
    probs = np.exp(result) / np.sum(np.exp(result))

    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE

    gradients = []

    # Calculating the loss
    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])

    # Turning the correct label(=y) into a vector
    label = y
    y = np.zeros(len(y_hat))
    y[label] = 1

    i = iter(params)
    params_pairs = list(zip(i, i))

    # Using the inputs to each layer which i saved earlier in the classifier_output function.
    inputs = input_lst.copy()

    error = y_hat - y

    # We are looping over the params pairs from the last ones to the first ones
    for W, b in reversed(params_pairs):

        # Popping the output of the last tanh
        input_layer = inputs.pop()
        gb = error
        gW = np.outer(input_layer, gb)
        gradients = [gW, gb] + gradients

        # If there are hidden layers
        if len(inputs) != 0:
            # Popping the input to the tanh
            input_layer = inputs.pop()

            # Calculating the error to the previous layer
            error = (np.dot(W, error)) * (1 - np.square(np.tanh(input_layer)))

    return loss, gradients


# Initialization function to the weights matrices and the bias vectors.
def my_random(size1, size2=None):
    t = 1 if size2 is None else size2
    eps = np.sqrt(6.0 / (size1 + t))
    return np.random.uniform(-eps, eps, (size1, size2)) if size2 is not None else np.random.uniform(-eps, eps, size1)


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    params = []

    for dim_in, dim_out in zip(dims[0::1], dims[1::1]):
        params.append(my_random(dim_in, dim_out))
        params.append(my_random(dim_out))

    return params
