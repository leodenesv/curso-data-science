import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Baixar dados da PETR4 dos últimos 6 meses
ticker = yf.Ticker("PETR4.SA")
df = ticker.history(period="6mo")

# Manter só o que importa
df = df[["Open", "High", "Low", "Close", "Volume"]]

# Renomear para português
df.columns = ["abertura", "maxima", "minima", "fechamento", "volume"]

# Resetar índice para ter data como coluna normal
df = df.reset_index()
df.columns = ["data", "abertura", "maxima", "minima", "fechamento", "volume"]

# 1 — Retorno diário
df["retorno"] = df["fechamento"].pct_change()

# 2 — Média móvel de 20 dias
df["mm20"] = df["fechamento"].rolling(20).mean()

# 3 — Média móvel de 50 dias
df["mm50"] = df["fechamento"].rolling(50).mean()

# Ver resultado
print(df[["data", "fechamento", "retorno", "mm20", "mm50"]].tail(10))

# Gráfico de fechamento com médias móveis
plt.figure(figsize=(12, 5))
plt.plot(df["data"], df["fechamento"], label="Fechamento", linewidth=1)
plt.plot(df["data"], df["mm20"], label="MM20", linewidth=1.5, linestyle="--")
plt.plot(df["data"], df["mm50"], label="MM50", linewidth=1.5, linestyle=":")
plt.title("PETR4 — Fechamento e Médias Móveis")
plt.xlabel("Data")
plt.ylabel("Preço (R$)")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()


# Resumo estatístico
retorno_medio = df["retorno"].mean()
volatilidade = df["retorno"].std()
preco_atual = df["fechamento"].iloc[-1]
preco_inicial = df["fechamento"].iloc[0]
retorno_periodo = (preco_atual - preco_inicial) / preco_inicial

print("=" * 40)
print("RELATÓRIO — PETR4 (6 meses)")
print("=" * 40)
print(f"Preço inicial:     R$ {preco_inicial:.2f}")
print(f"Preço atual:       R$ {preco_atual:.2f}")
print(f"Retorno período:   {retorno_periodo:.1%}")
print(f"Retorno médio/dia: {retorno_medio:.2%}")
print(f"Volatilidade:      {volatilidade:.2%}")
print(f"MM20 atual:        R$ {df['mm20'].iloc[-1]:.2f}")
print(f"MM50 atual:        R$ {df['mm50'].iloc[-1]:.2f}")
print("=" * 40)