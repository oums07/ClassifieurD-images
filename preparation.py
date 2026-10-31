from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modele import initialiser_parametres, calculer_scores, softmax, calculer_perte

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