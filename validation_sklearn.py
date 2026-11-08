from preparation import X_train_std, y_train, X_val_std, y_val
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

C_possibles = [0.01, 0.1, 1, 10, 100]

meilleure_perte = float("inf")
meilleur_C = None
meilleure_exactitude = None

print("\nComparaison des paramètres sur la validation")

for C in C_possibles:

    modele = LogisticRegression(C=C, max_iter=1000)

    modele.fit(X_train_std, y_train)

    probabilites = modele.predict_proba(X_val_std)
    predictions = modele.predict(X_val_std)

    perte = log_loss(y_val, probabilites)
    exactitude = accuracy_score(y_val, predictions)

    print(f"C={C} | perte={perte:.6f} | exactitude={exactitude:.2%}")

    if perte < meilleure_perte:
        meilleure_perte = perte
        meilleur_C = C
        meilleure_exactitude = exactitude
        meilleur_modele = modele

print("\nMeilleur réglage scikit-learn :")
print("C :", meilleur_C)
print("Perte :", meilleure_perte)
print("Exactitude :", f"{meilleure_exactitude:.2%}")