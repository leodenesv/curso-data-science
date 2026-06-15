import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

np.random.seed(42)
sns.set_theme(style="whitegrid")

# Gerar dataset simulado de crédito
n = 1000

df = pd.DataFrame({
    "idade":           np.random.randint(18, 70, n),
    "renda_mensal":    np.random.uniform(1500, 15000, n),
    "divida_total":    np.random.uniform(0, 50000, n),
    "score_credito":   np.random.randint(300, 900, n),
    "meses_emprego":   np.random.randint(0, 120, n),
    "num_dependentes": np.random.randint(0, 5, n),
})

# Criar variável alvo com lógica realista
prob_inadimplencia = (
    0.3
    - 0.0003 * df["score_credito"]
    - 0.00001 * df["renda_mensal"]
    + 0.000005 * df["divida_total"]
    + np.random.normal(0, 0.1, n)
)

df["inadimplente"] = (prob_inadimplencia > 0.1).astype(int)

print(f"Shape: {df.shape}")
print(f"\nDistribuição da variável alvo:")
print(df["inadimplente"].value_counts())
print(f"\nTaxa de inadimplência: {df['inadimplente'].mean():.1%}")
print(f"\nPrimeiras linhas:")
print(df.head())


# Separar features e alvo
X = df.drop("inadimplente", axis=1)
y = df["inadimplente"]

# Dividir treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizar
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled  = scaler.transform(X_teste)

# Modelo 1 — Regressão Logística
lr = LogisticRegression(random_state=42)
lr.fit(X_treino_scaled, y_treino)
y_pred_lr = lr.predict(X_teste_scaled)

# Modelo 2 — Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_treino, y_treino)
y_pred_rf = rf.predict(X_teste)

print("=" * 50)
print("REGRESSÃO LOGÍSTICA")
print("=" * 50)
print(classification_report(y_teste, y_pred_lr))

print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print(classification_report(y_teste, y_pred_rf))


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusão
cm = confusion_matrix(y_teste, y_pred_lr)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Adimplente", "Inadimplente"],
    yticklabels=["Adimplente", "Inadimplente"],
    ax=axes[0]
)
axes[0].set_title("Matriz de Confusão — Regressão Logística")
axes[0].set_ylabel("Real")
axes[0].set_xlabel("Previsto")

# Curva ROC
y_prob_lr = lr.predict_proba(X_teste_scaled)[:, 1]
y_prob_rf = rf.predict_proba(X_teste)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_teste, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_teste, y_prob_rf)

auc_lr = roc_auc_score(y_teste, y_prob_lr)
auc_rf = roc_auc_score(y_teste, y_prob_rf)

axes[1].plot(fpr_lr, tpr_lr, label=f"Regressão Logística (AUC={auc_lr:.3f})")
axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
axes[1].plot([0,1], [0,1], "k--", label="Modelo aleatório (AUC=0.5)")
axes[1].set_title("Curva ROC")
axes[1].set_xlabel("Taxa de Falsos Positivos")
axes[1].set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
axes[1].legend()

plt.tight_layout()
plt.savefig("grafico_ml_avaliacao.png", dpi=150)
plt.show()
plt.close()
print(f"AUC Regressão Logística: {auc_lr:.3f}")
print(f"AUC Random Forest:       {auc_rf:.3f}")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Usar só duas features para visualizar
X_cluster = df[["score_credito", "renda_mensal"]].copy()

scaler2 = StandardScaler()
X_scaled = scaler2.fit_transform(X_cluster)

# Método do cotovelo — encontrar número ideal de clusters
inercias = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inercias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico do cotovelo
axes[0].plot(k_range, inercias, marker="o", linewidth=2)
axes[0].set_title("Método do Cotovelo — número ideal de clusters")
axes[0].set_xlabel("Número de clusters (k)")
axes[0].set_ylabel("Inércia")
axes[0].axvline(x=3, color="red", linestyle="--", alpha=0.7, label="k=3 (ideal)")
axes[0].legend()

# Treinar com k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

# Visualizar clusters
cores_cluster = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
for cluster in [0, 1, 2]:
    mask = df["cluster"] == cluster
    axes[1].scatter(
        df[mask]["score_credito"],
        df[mask]["renda_mensal"],
        c=cores_cluster[cluster],
        label=f"Cluster {cluster}",
        alpha=0.6,
        s=30
    )

axes[1].set_title("Segmentação de clientes — Score vs Renda")
axes[1].set_xlabel("Score de crédito")
axes[1].set_ylabel("Renda mensal (R$)")
axes[1].legend()

plt.tight_layout()
plt.savefig("grafico_clustering.png", dpi=150)
plt.show()
plt.close()

# Perfil de cada cluster
print("\nPERFIL MÉDIO POR CLUSTER:")
print(df.groupby("cluster")[["score_credito", "renda_mensal", "inadimplente"]].mean().round(2))