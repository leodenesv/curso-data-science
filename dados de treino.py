import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Dados com relação quadrática
np.random.seed(42)
X = np.linspace(0, 10, 40)
y = 2 * X**2 - 5 * X + np.random.normal(0, 8, 40)

X_plot = np.linspace(0, 10, 200)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

graus = [1, 4, 15]
titulos = ["Underfitting (grau 1)", "Bom ajuste (grau 4)", "Overfitting (grau 15)"]

for ax, grau, titulo in zip(axes, graus, titulos):
    modelo = make_pipeline(
        PolynomialFeatures(grau),
        LinearRegression()
    )
    modelo.fit(X.reshape(-1, 1), y)
    y_plot = modelo.predict(X_plot.reshape(-1, 1))

    ax.scatter(X, y, alpha=0.7, label="Dados reais")
    ax.plot(X_plot, y_plot, color="red", linewidth=2, label=f"Grau {grau}")
    ax.set_title(titulo)
    ax.set_ylim(-50, 200)
    ax.legend()

plt.tight_layout()
plt.show()
plt.close()