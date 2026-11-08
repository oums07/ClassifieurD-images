
from preparation import X_train_std, y_train, X_val_std, y_val
from modele import entrainer, softmax, calculer_scores, calculer_perte


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
