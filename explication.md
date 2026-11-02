&#x20;

**Plan du projet :** 

1. Explorer les images et leur répartition  (DONE)
2. Séparer et standardiser les données  (DONE)
3. Programmer la régression logistique multi-classe ( DONE)
4. Entraîner notre modèle avec la descente de gradient (DONE but i didnt test it yet )
5. Étudier les paramètres sur la validation
6. Entraîner la régression logistique de scikit-learn
7. Comparaison des 2 modèles
8. FACULTATIF: ajouter une régularisation, essayer un autre modèle ou tester sur MNIST





**EN GROS IL NOUS RESTE :**

\- Sur la validation : comparer plusieurs pas d’apprentissage et nombres d’itérations pour choisir leurs valeurs.

\- Avec scikit-learn : entraîner le modèle de référence sur les mêmes données.

\- Sur le test final : comparer les performances des deux modèles, avec le taux de bonnes prédictions, les scores par chiffre et les matrices de confusion.

\- Analyser les résultats : observer les chiffres confondus, afficher quelques erreurs et discuter des limites.

\- Le poster.



**L'architecture du projet :**



**Fichier 1 :exploration.py:**



&#x20;ici j'ai tout simplement chargé et essayé de tester et voir comment ça marche le jeu de données qu'on a . Au début j'ai affiché les 10 premières lignes du jeu de données donc il affichait les chiffres de 0 à 9 .

*Les variables utilisés* sont : 

***X*** : représente les 1797 lignes( le nb d'images) et 64 colonnes( le nb de pixels), ses dimensions sont(1797,64)

***y*** : représente le target c'est à dire la vraie étiquette associé à chaque image c'est à dire le chiffre qu'elle représente ses dimensions sont (1797,)

Donc chaque image à son propre étiquette par exemple y\[0] donne la bonne réponse pour l'image X\[0]



Au total on a 1797 images , ces images sont des chiffres manuscrits allant de 0 à 9 .



Et à la fin du fichier j'ai fait une répartition des chiffres afin de savoir chaque chiffre est représenté cmb de fois par exemple le chiffre 0 a 178 images représentatives .

Jusqu'à ici on essaye juste d'explorer le modèle , il ne reconnaît pas les chiffres lui-mm

**Nb: Les valeurs des pixels vont de 0 à 16 : 0 représente le noir, 16 le blanc et les valeurs intermédiaires des niveaux de gris.**





**Fichier 2: preparation.py:** 



pour ce fichier on commence à préparer les données pour l'apprentissage en les séparant en 3 groupes : groupe d'entraînement , groupe de validation et groupe de test .

pour le groupe d'entraînement. ça sert à quoi ? 

**Entraînement (60 %) :** le modèle apprend avec ces images.

**Validation (20 %) :** on choisit les paramètres avec ces images .

**Test (20 %) :** on évalue le modèle final sur ces images gardées de côté .

Chaque image reste associée à sa bonne réponse et appartient à un seul groupe. **random\_state=42** permet de reproduire le même découpage, et **stratify** conserve approximativement les proportions des chiffres .

et là on passe à la standardisation des pixels du coup pour chaque pixel on applique xstandardisé=(x-moyenne)/écart-type

Après transformation, les colonnes non constantes de l’entraînement ont une moyenne proche de 0 et un écart-type proche de 1. Les colonnes constantes sont transformées en zéros par **StandardScaler**.

**NB: On calcule la moyenne et l’écart-type de chaque pixel sur l’entraînement uniquement, puis on les réutilise pour les trois groupes .**



**Fichier 3 : modele.py :**



Là on essaye de construire notre propre modèle , donc on commence par calculer les scores , chaque chiffre aura 10 scores . 

**Analyse de ces scores** : un score élevé indique que le modèle favorise sa classe . 

**Comment calculer ce score ?** score= x1\*w1 + x2\*w2 + … + x64\*w64 + b

&#x20;-Les x sont les 64 pixels standardisés .

&#x20;-Les w sont les coefs appelés Poids que le modèle apprendra .

&#x20;-Le b est un biais , lui aussi il sera appris et il permet de décaler le socre .



**NB:Le biais donne plus de liberté au modèle : il permet d’ajuster le score d’un chiffre sans modifier l’influence des pixels .**



Et puis on doit transformer ces scores en probabilités , c'est la fonction softmax .

Ensuite la fonction perte qui sert à mesurer à quel point notre modèle se trompe , on utilisera l'entropie croisée . 

**Calcul de perte :**  perte= -ln(proba du vrai chiffre)

&#x20;après on calcule la moyenne des pertes sur toutes les images .



**Pourquoi on calcule cette perte et la moyenne ?** Pour que l'apprentissage , il se basera sur cette moyenne et il va essayer de la réduire .



LA dernière partie c'est de **calculer le gradient** , c'est lui qui nous indique comment modifier les poids et les biais pendant l'apprentissage pour diminuer cette perte .

Ce Gradient contient une dérivée pour chaque param . Une dérivée indique comment la perte varie qd on change le poids 

et pour le calcul de la dérivée c'est tout simplement dL/dw

si cette dérivée est + , on cherche à diminuer sinon on fait l'inverse



**Entraînement avec la descente de gradient** c'est qu'on met à jour les paramètres en soustrayant leur gradient multiplié par un pas d’apprentissage :

W <- W - Pas \* gradient(w)

b <- b - pas \* gradient(b)



Le pas règle la taille des modifications. Il doit être suffisamment petit pour éviter des changements trop importants.



La boucle d’entraînement répète donc : calcul des scores → probabilités → perte → gradients → mise à jour des poids et des biais.



C’est cette répétition qui permet au modèle d’apprendre à reconnaître les chiffres à partir des images d’entraînement.









