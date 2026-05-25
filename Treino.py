from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Dados
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.ravel()**2 - 5 * X.ravel() + np.random.normal(0, 8, 100)

# Dividir em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Treino: {X_treino.shape[0]} amostras")
print(f"Teste:  {X_teste.shape[0]} amostras")
print()

# Testar 3 graus
for grau in [1, 4, 15]:
    modelo = make_pipeline(PolynomialFeatures(grau), LinearRegression())
    modelo.fit(X_treino, y_treino)

    r2_treino = r2_score(y_treino, modelo.predict(X_treino))
    r2_teste  = r2_score(y_teste,  modelo.predict(X_teste))
    mse_teste = mean_squared_error(y_teste, modelo.predict(X_teste))

    print(f"Grau {grau:2d} | R² treino: {r2_treino:.3f} | R² teste: {r2_teste:.3f} | MSE teste: {mse_teste:.1f}")