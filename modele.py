import numpy as np


def initialiser_parametres(nb_pixels, nb_classes):
    W = np.zeros((nb_pixels, nb_classes))
    b = np.zeros(nb_classes)
    return W, b


def calculer_scores(X, W, b):
    return X @ W + b

def softmax(scores):
    scores_stables = scores - np.max(scores, axis=1, keepdims=True)
    exponentielles = np.exp(scores_stables)

    return exponentielles / np.sum(
        exponentielles, axis=1, keepdims=True
    )

def calculer_perte(probabilites, y):
    nb_images = len(y)

    probabilites_correctes = probabilites[
        np.arange(nb_images), y
    ]

    return -np.mean(
        np.log(np.clip(probabilites_correctes, 1e-15, 1.0))
    )