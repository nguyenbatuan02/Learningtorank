import numpy as np


def readDataset(path):

    X_train = []  # <feature-value>[136]
    Y_train = []  # <label>
    Query = []  # <qid>

    print('Reading training data from file...')
    with open(path, 'r') as file:
        for line in file:
            split = line.split(' ')
            Y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query.append(int(split[1].split(':')[1]))

        for i in range(len(X_train)):
            X_train[i] = normalize_input(X_train[i])
    print('Read %d lines from file...' % (len(X_train)))
    return X_train, Y_train, Query


def normalize_input(x):
    muy = np.mean(x)
    sigma = np.std(x)
    x = (x-muy)/sigma

    return x


def extractFeatures(split):

    features = []
    for i in range(2, 138):
        features.append(float(split[i].split(':')[1]))

    return features


def extractPairsOfRatedSites(Y_train, Query):

    pairs = []

    for i in range(0, 100):
        for j in range(i + 1, 100):
            # Only look at queries with the same id
            if Query[i] != Query[j]:
                break
            # Document pairs found with different rating
            if Query[i] == Query[j] and Y_train[i] != Y_train[j]:
                # Sort by saving the largest index in position 0
                if Y_train[i] > Y_train[j]:
                    pairs.append([i, j])
                else:
                    pairs.append([j, i])
    print('Found %d document pairs' % (len(pairs)))

    return pairs


