import sqlite3
import pandas as pd

# 1. Conexão com o banco
conn = sqlite3.connect('sistema_preditivo.db')
cursor = conn.cursor()

# 2. Criação da tabela com o esquema EXATO do seu Excel (em minúsculas por padrão SQL)
cursor.execute('''
CREATE TABLE IF NOT EXISTS tb_estudantes (
    codigo_sgde INTEGER PRIMARY KEY,
    situacao_matricula TEXT,
    data_nascimento TEXT,
    idade INTEGER,
    trabalhador TEXT,
    sexo TEXT,
    cor_raca TEXT,
    necessidades_especificas TEXT,
    endereco_anonimo TEXT,
    bairro TEXT,
    municipio TEXT
)
''')
conn.commit()

# --- IMPORTAÇÃO DOS DADOS ---

try:
    # Carrega o Excel
    df_fichas = pd.read_excel('dados_fichas_higienizados.xlsx')
    
    # PADRONIZAÇÃO: Transformamos os nomes das colunas para minúsculo 
    # para bater com o banco de dados e removemos espaços extras
    df_fichas.columns = [c.lower().strip() for c in df_fichas.columns]
    
    # Inserimos no banco
    # 'if_exists=replace' é melhor para esta fase de testes, pois ele recria a tabela
    # Se usar 'append', ele dará erro de "Primary Key" se você rodar duas vezes o mesmo aluno.
    df_fichas.to_sql('tb_estudantes', conn, if_exists='replace', index=False)
    
    print("✅ Sucesso! Banco de dados criado e dados importados.")
    
    # Teste rápido: contar quantos alunos entraram
    cursor.execute("SELECT COUNT(*) FROM tb_estudantes")
    total = cursor.fetchone()[0]
    print(f"📊 Total de alunos cadastrados no SQL: {total}")

except Exception as e:
    print(f"⚠️ Erro na importação: {e}")

finally:
    conn.close()