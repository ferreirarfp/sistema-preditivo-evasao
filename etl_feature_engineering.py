import sqlite3
import pandas as pd

conn = sqlite3.connect('sistema_preditivo.db')

print("Iniciando a Engenharia de Atributos (Feature Engineering)...")

# 1. A Query Mágica: Agrupa os dados e calcula as estatísticas por aluno/ano
query_features = """
SELECT 
    codigo_sgde,
    ano_letivo,
    MAX(situacao_final) as situacao_final,
    ROUND(AVG(media_anual), 2) as media_geral_ano,
    SUM(total_faltas) as total_faltas_ano,
    COUNT(disciplina) as qtd_disciplinas
FROM tb_historico_academico
GROUP BY codigo_sgde, ano_letivo
"""

# Executa a query e carrega no Pandas
df_features = pd.read_sql_query(query_features, conn)

print(f"Transformação concluída! 6.536 registros de matérias foram reduzidos para {len(df_features)} resumos (1 por aluno/ano).")

# 2. Salva o resultado em uma nova tabela pronta para o algoritmo
df_features.to_sql('tb_features_ml', conn, if_exists='replace', index=False)
print("✅ Tabela 'tb_features_ml' criada com sucesso no banco de dados!")

conn.close()