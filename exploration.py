from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()

X = digits.data
y = digits.target


print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Dimensions d'une image :", digits.images[0].shape)
print("Valeur minimale des pixels :", X.min())
print("Valeur maximale des pixels :", X.max())


chiffres, effectifs = np.unique(y, return_counts=True)

print("\nRépartition des chiffres :")

for chiffre, effectif in zip(chiffres, effectifs):
    print(f"Chiffre {chiffre} : {effectif} images")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray", vmin=0, vmax=16)
    ax.set_title(f"Vrai chiffre : {y[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()



