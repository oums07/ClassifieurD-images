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

def calculer_gradients(X, probabilites, y):
    nb_images = X.shape[0]

    # Construire les bonnes réponses au format one-hot
    Y = np.zeros_like(probabilites)
    Y[np.arange(nb_images), y] = 1

    # Dérivée de la perte de chaque image par rapport aux scores
    ecarts = probabilites - Y

    # Gradients de la perte moyenne
    gradient_W = (X.T @ ecarts) / nb_images
    gradient_b = np.mean(ecarts, axis=0)

    return gradient_W, gradient_b


def entrainer(X, y, pas=0.1, nb_iterations=1000, afficher=True):
    W, b = initialiser_parametres(
        nb_pixels=X.shape[1],
        nb_classes=10
    )

    pertes = []

    for iteration in range(nb_iterations):
        
        scores = calculer_scores(X, W, b)
        probabilites = softmax(scores)

        
        perte = calculer_perte(probabilites, y)
        pertes.append(perte)

        
        gradient_W, gradient_b = calculer_gradients(
            X, probabilites, y
        )

        
        W = W - pas * gradient_W
        b = b - pas * gradient_b

        if afficher and iteration % 100 == 0:
            print(f"Itération {iteration} : perte = {perte:.4f}")

    return W, b, pertes