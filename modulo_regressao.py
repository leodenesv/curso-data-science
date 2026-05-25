import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dados simulados: volume vs retorno
np.random.seed(42)
n = 100

volume = np.random.uniform(10, 100, n)
retorno = 0.05 * volume + np.random.normal(0, 2, n)

df = pd.DataFrame({"volume": volume, "retorno": retorno})

# Visualizar relação
plt.figure(figsize=(8, 5))
plt.scatter(df["volume"], df["retorno"], alpha=0.6)
plt.xlabel("Volume (milhões)")
plt.ylabel("Retorno (%)")
plt.title("Relação Volume vs Retorno")
plt.show()
plt.close()

# Preparar dados para o sklearn
X = df[["volume"]]  # sempre 2D para sklearn
y = df["retorno"]

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Ver os coeficientes aprendidos
print(f"Intercepto (a):   {modelo.intercept_:.4f}")
print(f"Coeficiente (b):  {modelo.coef_[0]:.4f}")
print()
print(f"Fórmula aprendida: retorno = {modelo.intercept_:.4f} + {modelo.coef_[0]:.4f} × volume")

# Fazer previsões
y_pred = modelo.predict(X)

# Métricas
mse = mean_squared_error(y, y_pred)
r2  = r2_score(y, y_pred)
print()
print(f"MSE (erro médio quadrático): {mse:.4f}")
print(f"R² (coeficiente de determinação): {r2:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(df["volume"], df["retorno"], alpha=0.6, label="Dados reais")
plt.plot(df["volume"], y_pred, color="red", linewidth=2, label="Linha de regressão")
plt.xlabel("Volume (milhões)")
plt.ylabel("Retorno (%)")
plt.title(f"Regressão Linear — R²={r2:.2f}")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()