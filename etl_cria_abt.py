import sqlite3
import pandas as pd

conn = sqlite3.connect('sistema_preditivo.db')

print("Iniciando a criação da ABT 2.0 (A Tabela Mestra Enriquecida)...")

# A Query SQL que une o desempenho acadêmico com a realidade socioeconômica
query_abt = """
SELECT 
    f.codigo_sgde,
    f.ano_letivo,
    e.trabalha,
    e.carga_horaria,
    e.esforco_fisico,
    e.rotina_deslocamento,
    e.horas_sono,
    e.alimentacao,
    e.problemas_transporte,
    e.objetivo_pos_medio,
    e.horario_saida_trabalho,
    f.media_geral_ano,
    f.total_faltas_ano,
    f.qtd_disciplinas,
    f.situacao_final
FROM tb_features_ml f
LEFT JOIN tb_estudantes e ON f.codigo_sgde = e.codigo_sgde
"""

try:
    df_abt = pd.read_sql_query(query_abt, conn)
    
    # Salva a tabela final, sobrescrevendo a versão antiga
    df_abt.to_sql('tb_abt_preditiva', conn, if_exists='replace', index=False)
    
    print(f"✅ SUCESSO! A Tabela Base Analítica foi atualizada com {len(df_abt)} registros.")
    print("\nUma espiadinha nas colunas que vão alimentar a Inteligência Artificial:")
    print(df_abt.columns.tolist())
    
except Exception as e:
    print(f"❌ Erro no SQL: {e}")

conn.close()