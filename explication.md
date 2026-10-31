Projet MATH: 



y'a 2 fichiers pour l'instant : 



**exploration.py:** ici j'ai tout simplement chargé et essayé de tester et voir comment ça marche le jeu de données qu'on a . Au début j'ai affiché les 10 premières lignes du jeu de données donc il affichait les chiffres de 0 à 9 .

*Les variables utilisés* sont : 

***X*** : représente les 1797 lignes( le nb d'images) et 64 colonnes( le nb de pixels), ses dimensions sont(1797,64)

***y*** : représente le target c'est à dire la vraie étiquette associé à chaque image c'est à dire le chiffre qu'elle représente ses dimensions sont (1797,)

Donc chaque image à son propre étiquette par exemple y\[0] donne la bonne réponse pour l'image X\[0]



Au total on a 1797 images , ces images sont des chiffres manuscrits allant de 0 à 9 .



Et à la fin du fichier j'ai fait une répartition des chiffres afin de savoir chaque chiffre est représenté cmb de fois par exemple le chiffre 0 a 178 images représentatives .

Jusqu'à ici on essaye juste d'explorer le modèle , il ne reconnaît pas les chiffres lui-mm

**Nb: Les valeurs des pixels vont de 0 à 16 : 0 représente le noir, 16 le blanc et les valeurs intermédiaires des niveaux de gris.**





**preparation.py:** pour ce fichier on commence à préparer les données pour l'apprentissage en les séparant en 3 groupes : groupe d'entraînement , groupe de validation et groupe de test .

pour le groupe d'entraînement. ça sert à quoi ? 

**Entraînement (60 %) :** le modèle apprend avec ces images.

**Validation (20 %) :** on choisit les paramètres avec ces images.

**Test (20 %) :** on évalue le modèle final sur ces images gardées de côté.

Chaque image reste associée à sa bonne réponse et appartient à un seul groupe. **random\_state=42** permet de reproduire le même découpage, et **stratify** conserve approximativement les proportions des chiffres.

et là on passe à la standardisation des pixels du coup pour chaque pixel on applique xstandardisé=(x-moyenne)/écart-type

Après transformation, les colonnes non constantes de l’entraînement ont une moyenne proche de 0 et un écart-type proche de 1. Les colonnes constantes sont transformées en zéros par **StandardScaler**.

**NB: On calcule la moyenne et l’écart-type de chaque pixel sur l’entraînement uniquement, puis on les réutilise pour les trois groupes.**

