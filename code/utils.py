from collections import Counter

STUDENT = {'name': 'Dorin Keshales',
           'ID': '313298424'}


# Reading the data drom the requested file
def read_data(fname):
    with open(fname, "r", encoding="utf-8") as file:
        data = []
        for line in file:
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


# Splitting the data into bigrams
def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


# Splitting the data into unigrams
def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]


# Replacing the labels of all examples in the data set into the respective ID of each
def language_to_index(dataset, labels):
    for index in range(len(dataset)):
        label = labels.index(dataset[index][0], 0, len(labels))
        feats = dataset[index][1]
        dataset[index] = (label, feats)
    return dataset


# Replacing the ID of the label with the respective label(=language)
def index_to_language(pred):
    language = keys[pred]
    return language


# Returns the list of common features on the test set
def get_common_features():
    return features


# Loading the test set data
def load_test_set():
    TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]
    return TEST


# Loading the validation set data according to the requested representation of the features.
def load_validation_set(representation):
    DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")] if representation == 'bigrams' else [
        (l, text_to_unigrams(t)) for l, t in read_data("dev")]
    return language_to_index(DEV, keys)


# Loading the training set data according to the requested representation of the features and pulling out the common
# features on the training set
def load_train_set(representation):
    TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("train")] if representation == 'bigrams' else [
        (l, text_to_unigrams(t)) for l, t in read_data("train")]
    num_desired_features = 700 if representation == 'bigrams' else 90

    fc = Counter()
    for l, feats in TRAIN:
        fc.update(feats)

    # 700 most common bigrams/unigrams(following representation) in the training set.
    vocab = set([x for x, c in fc.most_common(num_desired_features)])

    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}

    # feature strings to IDs
    F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}

    global keys, features
    features = list(F2I.keys())
    keys = list(L2I.keys())

    # Replacing the labels of all examples in the data set into the respective ID of each.
    new_train = language_to_index(TRAIN, keys)

    return new_train, len(features), len(keys)
