import sqlite3
import pandas as pd

conn = sqlite3.connect("acoes.db")
cursor = conn.cursor()

# Recriar tabela do zero
cursor.execute("DROP TABLE IF EXISTS cotacoes")
cursor.execute("""
    CREATE TABLE cotacoes (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker     TEXT NOT NULL,
        data       TEXT NOT NULL,
        fechamento REAL NOT NULL,
        volume     INTEGER NOT NULL,
        setor      TEXT NOT NULL,
        UNIQUE(ticker, data)
    )
""")

# Dados
dados = [
    ("PETR4", "2026-04-01", 38.50, 45000000, "Energia"),
    ("PETR4", "2026-04-02", 39.20, 48000000, "Energia"),
    ("VALE3", "2026-04-01", 62.30, 32000000, "Mineração"),
    ("VALE3", "2026-04-02", 61.80, 29000000, "Mineração"),
    ("ITUB4", "2026-04-01", 29.80, 21000000, "Financeiro"),
    ("ITUB4", "2026-04-02", 30.10, 23000000, "Financeiro"),
]

# Inserir com proteção contra duplicata
cursor.executemany("""
    INSERT OR IGNORE INTO cotacoes (ticker, data, fechamento, volume, setor)
    VALUES (?, ?, ?, ?, ?)
""", dados)

conn.commit()
print("Banco criado com sucesso.")

# Query 1 — Ver tudo
df = pd.read_sql_query("SELECT * FROM cotacoes", conn)
print("\nTODOS OS DADOS:")
print(df)

# Query 2 — Filtrar PETR4
df_petr4 = pd.read_sql_query(
    "SELECT * FROM cotacoes WHERE ticker = 'PETR4'", conn)
print("\nAPENAS PETR4:")
print(df_petr4)

# Query 3 — Resumo por ticker
df_resumo = pd.read_sql_query("""
    SELECT ticker,
           AVG(fechamento) as preco_medio,
           SUM(volume)     as volume_total
    FROM cotacoes
    GROUP BY ticker
    ORDER BY preco_medio DESC
""", conn)
print("\nRESUMO POR TICKER:")
print(df_resumo)

# Criar segunda tabela com informações das empresas
cursor.execute("DROP TABLE IF EXISTS empresas")
cursor.execute("""
    CREATE TABLE empresas (
        ticker     TEXT PRIMARY KEY,
        nome       TEXT NOT NULL,
        setor      TEXT NOT NULL,
        founded    INTEGER
    )
""")

empresas = [
    ("PETR4", "Petrobras", "Energia", 1953),
    ("VALE3", "Vale", "Mineração", 1942),
    ("ITUB4", "Itaú Unibanco", "Financeiro", 1924),
]

cursor.executemany("""
    INSERT OR IGNORE INTO empresas (ticker, nome, setor, founded)
    VALUES (?, ?, ?, ?)
""", empresas)

conn.commit()

# JOIN — unir cotacoes com empresas
df_join = pd.read_sql_query("""
    SELECT c.ticker,
           e.nome,
           c.data,
           c.fechamento,
           c.volume
    FROM cotacoes c
    JOIN empresas e ON c.ticker = e.ticker
    ORDER BY c.data, c.ticker
""", conn)

print("COTAÇÕES COM NOME DA EMPRESA:")
print(df_join)

conn.close()

