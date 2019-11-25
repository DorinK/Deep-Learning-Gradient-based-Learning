import numpy as np

STUDENT = {'name': 'Dorin Keshales',
           'ID': '313298424'}


def classifier_output(x, params):
    # YOUR CODE HERE.

    W, b, U, b_tag = params

    # Calculation of the input to the hidden layer
    result = np.dot(x, W) + b

    # Saving copy of the hidden layer input - before the tanh
    global z1
    z1 = result.copy()

    # Using tanh as activation function
    result = np.tanh(result)

    # Saving copy of the hidden layer output - after the tanh
    global h1
    h1 = result.copy()

    # Calculating the output of the model and using SoftMax
    result = np.dot(result, U) + b_tag
    result -= result.max()
    probs = np.exp(result) / np.sum(np.exp(result))

    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE

    W, b, U, b_tag = params

    # Calculating the loss
    model_output = classifier_output(x, params)
    loss = -np.log(model_output[y])

    # derivative of the loss by b_tag
    gb_tag = model_output.copy()
    gb_tag[y] -= 1

    # derivative of loss by U
    copy_output = model_output.copy()
    copy_h1 = h1.copy()
    gU = np.outer(copy_h1, copy_output)
    gU[:, y] -= copy_h1

    # derivative of softmax by h1 which represents the vector after the tanh
    ds_dh1 = np.dot(U, model_output) - U[:, y]

    # derivative of the vector after the tanh (h1) by the vector before the tanh (z1)
    copy_z1 = z1.copy()
    dh1_dz1 = 1 - np.square(np.tanh(copy_z1))

    # derivative of the loss by b
    gb = ds_dh1 * dh1_dz1
    # derivative of the loss by W
    gW = np.outer(x, gb.copy())

    return loss, [gW, gb, gU, gb_tag]


# Initialization function to the weights matrices and the bias vectors.
def my_random(size1, size2=None):
    t = 1 if size2 is None else size2
    eps = np.sqrt(6.0 / (size1 + t))
    return np.random.uniform(-eps, eps, (size1, size2)) if size2 is not None else np.random.uniform(-eps, eps, size1)


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """

    W = my_random(in_dim, hid_dim)
    b = my_random(hid_dim)
    U = my_random(hid_dim, out_dim)
    b_tag = my_random(out_dim)

    params = [W, b, U, b_tag]
    return params
