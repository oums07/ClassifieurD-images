from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modele import initialiser_parametres, calculer_scores, softmax, calculer_perte
from modele import calculer_gradients, entrainer


digits = load_digits()
X = digits.data
y = digits.target


X_restant, X_test, y_restant, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_restant,
    y_restant,
    test_size=0.25,
    random_state=42,
    stratify=y_restant
)

print("Entraînement :", X_train.shape, y_train.shape)
print("Validation   :", X_val.shape, y_val.shape)
print("Test         :", X_test.shape, y_test.shape)



scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)


X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

print("\nAprès standardisation :")
print("Entraînement :", X_train_std.shape)
print("Validation   :", X_val_std.shape)
print("Test         :", X_test_std.shape)


print("Moyenne du pixel 2 :", X_train_std[:, 2].mean())
print("Écart-type du pixel 2 :", X_train_std[:, 2].std())



W, b = initialiser_parametres(
    nb_pixels=X_train_std.shape[1],
    nb_classes=10
)

scores = calculer_scores(X_train_std, W, b)

print("\nDimensions des poids :", W.shape)
print("Dimensions des biais :", b.shape)
print("Dimensions des scores :", scores.shape)
print("Scores de la première image :", scores[0])



probabilites = softmax(scores)

print("\nDimensions des probabilités :", probabilites.shape)
print("Probabilités de la première image :", probabilites[0])
print("Somme :", probabilites[0].sum())


perte = calculer_perte(probabilites, y_train)

print("\nPerte initiale :", perte)



gradient_W, gradient_b = calculer_gradients(
    X_train_std, probabilites, y_train
)

print("\nDimensions du gradient des poids :", gradient_W.shape)
print("Dimensions du gradient des biais :", gradient_b.shape)


W, b, pertes = entrainer(
    X_train_std,
    y_train,
    pas=0.1,
    nb_iterations=1000
)


probabilites_finales = softmax(
    calculer_scores(X_train_std, W, b)
)

perte_finale = calculer_perte(
    probabilites_finales, y_train
)

print("\nPerte finale d'entraînement :", perte_finale)


#5
print("\nComparaison des paramètres sur la validation")
pas_possibles = [0.1, 0.2, 0.5, 1, 2]
iterations_possibles = [1000, 3000, 5000]
meilleure_perte = float("inf")
meilleur_pas = None
meilleur_nombre_iterations = None
meilleure_exactitude = None

for pas in pas_possibles:
    for nb_iterations in iterations_possibles:
        W_candidat, b_candidat, _ = entrainer(
            X_train_std,
            y_train,
            pas=pas,
            nb_iterations=nb_iterations,
            afficher=False
        )

        probabilites_val = softmax(
            calculer_scores(X_val_std, W_candidat, b_candidat)
        )
        perte_val = calculer_perte(probabilites_val, y_val)
        predictions_val = probabilites_val.argmax(axis=1)
        exactitude_val = (predictions_val == y_val).mean()

        print(f"Pas={pas:.2f}, itérations={nb_iterations:4d} | "f"perte validation={perte_val:.6f} | "f"exactitude={exactitude_val:.2%}")

        if perte_val < meilleure_perte:
            meilleure_perte = perte_val
            meilleur_pas = pas
            meilleur_nombre_iterations = nb_iterations
            meilleure_exactitude = exactitude_val

print("\nMeilleur réglage selon la perte de validation : "f"pas={meilleur_pas}, "f"itérations={meilleur_nombre_iterations} "f"(perte={meilleure_perte:.6f}, "f"exactitude={meilleure_exactitude:.2%})")
