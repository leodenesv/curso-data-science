import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Baixar dados de 5 ações por 1 ano
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BBDC4.SA"]

print("Baixando dados...")
dados = {}
for ticker in tickers:
    df = yf.Ticker(ticker).history(period="1y")
    dados[ticker] = df["Close"]

# Montar DataFrame com todos os fechamentos
df = pd.DataFrame(dados)
df.columns = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBDC4"]

print(f"Shape: {df.shape}")
print(df.head())

# Passo 1 — Verificar valores ausentes
print("VALORES AUSENTES:")
print(df.isnull().sum())
print()

# Passo 2 — Estatísticas descritivas
print("ESTATÍSTICAS:")
print(df.describe().round(2))
print()

# Passo 3 — Verificar tipos de dados
print("TIPOS:")
print(df.dtypes)

# Calcular retornos diários
retornos = df.pct_change().dropna()

print("RETORNOS MÉDIOS DIÁRIOS:")
print(retornos.mean().round(4))
print()

print("VOLATILIDADE DIÁRIA:")
print(retornos.std().round(4))
print()

print("CORRELAÇÃO ENTRE AÇÕES:")
print(retornos.corr().round(2))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1 — Preços normalizados
df_norm = df / df.iloc[0] * 100
df_norm.plot(ax=axes[0, 0])
axes[0, 0].set_title("Preços normalizados (base 100)")
axes[0, 0].set_ylabel("Índice")

# Gráfico 2 — Volatilidade (desvio padrão móvel)
retornos.rolling(20).std().plot(ax=axes[0, 1])
axes[0, 1].set_title("Volatilidade móvel 20 dias")
axes[0, 1].set_ylabel("Desvio padrão")

# Gráfico 3 — Distribuição de retornos PETR4
axes[1, 0].hist(retornos["PETR4"], bins=50, edgecolor="white")
axes[1, 0].set_title("Distribuição de retornos — PETR4")
axes[1, 0].set_xlabel("Retorno diário")

# Gráfico 4 — Matriz de correlação visual
import matplotlib.colors as mcolors
corr = retornos.corr()
im = axes[1, 1].imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels(corr.columns)
axes[1, 1].set_yticklabels(corr.columns)
axes[1, 1].set_title("Mapa de correlação")
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()