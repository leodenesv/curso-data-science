import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Configuração visual padrão
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# Baixar dados reais
print("Baixando dados...")
tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BBDC4.SA"]
dados = {}
for ticker in tickers:
    df = yf.Ticker(ticker).history(period="1y")
    dados[ticker] = df["Close"]

df = pd.DataFrame(dados)
df.columns = ["PETR4", "VALE3", "ITUB4", "WEGE3", "BBDC4"]
retornos = df.pct_change().dropna()

print("Dados prontos.")
print(df.shape)

# Gráfico 1 — Série temporal
fig, ax = plt.subplots(figsize=(12, 5))

df_norm = df / df.iloc[0] * 100

for coluna in df_norm.columns:
    ax.plot(df_norm.index, df_norm[coluna], label=coluna, linewidth=1.5)

ax.set_title("Performance comparada — últimos 12 meses (base 100)", fontsize=14)
ax.set_ylabel("Índice (base 100)")
ax.set_xlabel("Data")
ax.legend(loc="upper left")
ax.axhline(y=100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig("grafico1_serie_temporal.png", dpi=150)
plt.show()
plt.close()
print("Gráfico 1 salvo.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma comparativo
for coluna in retornos.columns:
    axes[0].hist(retornos[coluna], bins=40, alpha=0.5, label=coluna)

axes[0].set_title("Distribuição de retornos diários")
axes[0].set_xlabel("Retorno diário")
axes[0].set_ylabel("Frequência")
axes[0].axvline(x=0, color="black", linewidth=1)
axes[0].legend()

# KDE — curva de densidade
for coluna in retornos.columns:
    retornos[coluna].plot.kde(ax=axes[1], linewidth=2, label=coluna)

axes[1].set_title("Densidade de retornos (KDE)")
axes[1].set_xlabel("Retorno diário")
axes[1].set_ylabel("Densidade")
axes[1].axvline(x=0, color="black", linewidth=1)
axes[1].set_xlim(-0.08, 0.08)
axes[1].legend()

plt.tight_layout()
plt.savefig("grafico2_distribuicao.png", dpi=150)
plt.show()
plt.close()
print("Gráfico 2 salvo.")


fig, ax = plt.subplots(figsize=(8, 6))

corr = retornos.corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    ax=ax
)

ax.set_title("Matriz de correlação — retornos diários", fontsize=14)
plt.tight_layout()
plt.savefig("grafico3_correlacao.png", dpi=150)
plt.show()
plt.close()
print("Gráfico 3 salvo.")


fig, ax = plt.subplots(figsize=(10, 6))

retornos_pct = retornos * 100

sns.boxplot(data=retornos_pct, ax=ax, palette="muted")

ax.set_title("Distribuição de retornos diários por ação", fontsize=14)
ax.set_ylabel("Retorno diário (%)")
ax.set_xlabel("Ação")
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

plt.tight_layout()
plt.savefig("grafico4_boxplot.png", dpi=150)
plt.show()
plt.close()
print("Gráfico 4 salvo.")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_norm = df / df.iloc[0] * 100

fig = go.Figure()

for coluna in df_norm.columns:
    fig.add_trace(go.Scatter(
        x=df_norm.index,
        y=df_norm[coluna],
        name=coluna,
        mode="lines",
        hovertemplate="%{x}<br>Índice: %{y:.1f}<extra></extra>"
    ))

fig.add_hline(
    y=100,
    line_dash="dash",
    line_color="black",
    opacity=0.5,
    annotation_text="Base 100"
)

fig.update_layout(
    title="Performance comparada — últimos 12 meses (interativo)",
    xaxis_title="Data",
    yaxis_title="Índice (base 100)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

fig.write_html("grafico5_interativo.html")
fig.show()
print("Gráfico 5 salvo como HTML.")


fig, ax = plt.subplots(figsize=(14, 6))

df_norm = df / df.iloc[0] * 100

cores = {
    "PETR4": "#1f77b4",
    "VALE3": "#ff7f0e",
    "ITUB4": "#2ca02c",
    "WEGE3": "#d62728",
    "BBDC4": "#9467bd"
}

for coluna in df_norm.columns:
    ax.plot(
        df_norm.index,
        df_norm[coluna],
        label=coluna,
        linewidth=1.5,
        color=cores[coluna]
    )

# Linha de referência
ax.axhline(y=100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

# Anotação do melhor ativo
ax.annotate(
    "VALE3: +60% no período",
    xy=(df_norm.index[-1], df_norm["VALE3"].iloc[-1]),
    xytext=(-120, 15),
    textcoords="offset points",
    fontsize=9,
    color=cores["VALE3"],
    arrowprops=dict(arrowstyle="->", color=cores["VALE3"])
)

# Anotação do pior ativo
ax.annotate(
    "WEGE3: performance negativa",
    xy=(df_norm.index[-1], df_norm["WEGE3"].iloc[-1]),
    xytext=(-160, -25),
    textcoords="offset points",
    fontsize=9,
    color=cores["WEGE3"],
    arrowprops=dict(arrowstyle="->", color=cores["WEGE3"])
)

# Título e subtítulo
ax.set_title(
    "B3 — Performance comparada nos últimos 12 meses\n"
    "VALE3 liderou com alta de 60%; WEGE3 foi o único ativo com retorno negativo",
    fontsize=13,
    loc="left"
)

ax.set_ylabel("Índice (base 100 = início do período)")
ax.set_xlabel("")
ax.legend(loc="upper left", framealpha=0.9)

# Fonte dos dados
ax.text(
    0.99, 0.01,
    "Fonte: Yahoo Finance | Elaboração própria",
    transform=ax.transAxes,
    fontsize=8,
    ha="right",
    color="gray"
)

sns.despine()
plt.tight_layout()
plt.savefig("grafico6_storytelling.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("Gráfico 6 salvo.")