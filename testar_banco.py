import sqlite3
import pandas as pd

print("🔍 Inspecionando o Banco de Dados...\n")

conn = sqlite3.connect('sistema_preditivo.db')

# Busca alunos que têm distorção maior que zero
query = """
SELECT codigo_sgde, ano_letivo, media_geral_ano, distorcao_idade_serie 
FROM tb_abt_preditiva 
WHERE distorcao_idade_serie > 0 
LIMIT 10
"""

df_teste = pd.read_sql_query(query, conn)
conn.close()

if df_teste.empty:
    print("⚠️ A coluna existe, mas todos os alunos estão com distorção 0 (ou o cálculo falhou).")
else:
    print("✅ SUCESSO! Veja os alunos com atraso escolar encontrados:")
    print("-" * 50)
    print(df_teste)
    print("-" * 50)
    