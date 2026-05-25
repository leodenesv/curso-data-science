import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel()**2 - 5 * X.ravel() + np.random.normal(0, 8, 100)

modelo = make_pipeline(
    PolynomialFeatures(4),
    StandardScaler(),
    Ridge(alpha=1)
)

# KFold com shuffle embaralha os dados antes de dividir
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=kf, scoring="r2")

print("R² em cada fold:")
for i, score in enumerate(scores):
    print(f"  Fold {i+1}: {score:.3f}")

print(f"Média R²: {scores.mean():.3f}")
print(f"Desvio padrão: {scores.std():.3f}")