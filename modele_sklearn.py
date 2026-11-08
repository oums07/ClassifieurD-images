
from preparation import X_train_std, y_train, X_val_std, y_val
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


# Creation du modele
modele_sklearn = LogisticRegression(max_iter=1000)

# Entraînement du modele
modele_sklearn.fit(X_train_std, y_train)

# prediction des chiffres
predictions = modele_sklearn.predict(X_val_std)

print("\nPrédictions :", predictions[:10])
print("Vrais chiffres :", y_val[:10])

# Calcule des probabilites
probabilites = modele_sklearn.predict_proba(X_val_std)

# Calcule de  la perte et lexactitude
perte = log_loss(y_val, probabilites)
exactitude = accuracy_score(y_val, predictions)

print("\nRésultats scikit-learn :")
print("Perte de validation :", perte)
print("Exactitude de validation :", f"{exactitude:.2%}")
