from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0, 10, 80).reshape(-1, 1)
y = 2 * X.ravel()**2 - 5 * X.ravel() + np.random.normal(0, 8, 80)

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Testar diferentes valores de alpha
alphas = [0.001, 0.1, 1, 10, 100]
r2_treinos = []
r2_testes  = []

for alpha in alphas:
    modelo = make_pipeline(
        PolynomialFeatures(10),
        StandardScaler(),
        Ridge(alpha=alpha)
    )
    modelo.fit(X_treino, y_treino)
    r2_treinos.append(r2_score(y_treino, modelo.predict(X_treino)))
    r2_testes.append(r2_score(y_teste,  modelo.predict(X_teste)))

# Visualizar
plt.figure(figsize=(8, 5))
plt.plot(alphas, r2_treinos, marker="o", label="R² treino")
plt.plot(alphas, r2_testes,  marker="o", label="R² teste")
plt.xscale("log")
plt.xlabel("Alpha (escala logarítmica)")
plt.ylabel("R²")
plt.title("Ridge Regression — Efeito do Alpha")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# Mostrar melhor alpha
melhor_idx = np.argmax(r2_testes)
print(f"Melhor alpha: {alphas[melhor_idx]}")
print(f"R² teste com melhor alpha: {r2_testes[melhor_idx]:.3f}")

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel()**2 - 5 * X.ravel() + np.random.normal(0, 8, 100)

modelo = make_pipeline(
    PolynomialFeatures(4),
    StandardScaler(),
    Ridge(alpha=1)
)

scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")

print("R² em cada fold:")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.3f}")

print(f"\nMédia R²:        {scores.mean():.3f}")
print(f"Desvio padrão:   {scores.std():.3f}")
print(f"\nInterpretação: modelo tem R² de {scores.mean():.3f} ± {scores.std():.3f}")