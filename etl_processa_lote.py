import sqlite3
import pandas as pd
import numpy as np
import os
import re

# 1. Configurações Iniciais
PASTA_DADOS = 'dados_historicos'
BANCO_DADOS = 'sistema_preditivo.db'

# Conectar ao banco
conn = sqlite3.connect(BANCO_DADOS)
cursor = conn.cursor()

# Garantir que a tabela existe
cursor.execute('''
CREATE TABLE IF NOT EXISTS tb_historico_academico (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_sgde INTEGER,
    disciplina TEXT,
    nota_bim1 REAL,
    nota_bim2 REAL,
    nota_bim3 REAL,
    nota_bim4 REAL,
    media_anual REAL,
    total_faltas INTEGER,
    situacao_final TEXT,
    ano_letivo INTEGER,
    FOREIGN KEY (codigo_sgde) REFERENCES tb_estudantes (codigo_sgde)
)
''')
conn.commit()

# 2. Listar todos os arquivos Excel na pasta
arquivos_excel = [f for f in os.listdir(PASTA_DADOS) if f.endswith('.xls') or f.endswith('.xlsx')]

if not arquivos_excel:
    print(f"Nenhum arquivo Excel encontrado na pasta '{PASTA_DADOS}'.")
    exit()

print(f"Iniciando processamento de {len(arquivos_excel)} arquivos...\n")

df_acumulado = pd.DataFrame()

# 3. Processar cada arquivo
for arquivo in arquivos_excel:
    caminho_completo = os.path.join(PASTA_DADOS, arquivo)
    print(f"Processando: {arquivo}")
    
    # Extrair o Ano Letivo do nome do arquivo usando Expressão Regular
    # Procura por 4 dígitos seguidos no nome do arquivo (ex: "2024")
    ano_match = re.search(r'20\d{2}', arquivo)
    ano_letivo = int(ano_match.group()) if ano_match else 0
    
    try:
        # Lendo o arquivo pulando as 6 primeiras linhas de cabeçalho
        df = pd.read_excel(caminho_completo, sheet_name=0, skiprows=6)
        
        # Limpeza básica de linhas/colunas vazias
        df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
        
        # Padronizar nomes das colunas
        df.columns = ['codigo_sgde', 'n_chamada', 'disciplina', 'bim_1', 'bim_2', 'bim_3', 
                      'bim_4', 'media_anual', 'exame_final', 'media_final', 'total_faltas', 
                      'nota_necessaria', 'situacao']
        
        # Filtros de validade
        df = df[df['codigo_sgde'].notna()]
        df = df[df['codigo_sgde'] != 'Cod. Estudante']
        df['situacao'] = df['situacao'].astype(str).str.strip()
        df['ano_letivo'] = ano_letivo
        
        # Empilhar no DataFrame principal
        df_acumulado = pd.concat([df_acumulado, df], ignore_index=True)
        
    except Exception as e:
        print(f"  ❌ Erro ao ler {arquivo}: {e}")

print(f"\nLeitura concluída. Total de {len(df_acumulado)} registros encontrados em todos os arquivos.")

# 4. Tratamento Fino dos Dados Acumulados
def limpa_numeros(valor):
    try:
        return float(valor)
    except:
        return None

colunas_numericas = ['codigo_sgde', 'bim_1', 'bim_2', 'bim_3', 'bim_4', 'media_anual', 'total_faltas']
for col in colunas_numericas:
    df_acumulado[col] = df_acumulado[col].apply(limpa_numeros)

# Eliminar possíveis linhas que ficaram com SGDE nulo após a conversão
df_acumulado = df_acumulado.dropna(subset=['codigo_sgde'])
df_acumulado['codigo_sgde'] = df_acumulado['codigo_sgde'].astype(int)

# 5. A Grande Solução: Garantir que todos os SGDEs existam na tb_estudantes
print("\nVerificando a integridade dos alunos no banco de dados...")
sgdes_nos_relatorios = df_acumulado['codigo_sgde'].unique()

# Buscar os alunos que já existem no banco
cursor.execute("SELECT codigo_sgde FROM tb_estudantes")
alunos_banco = [row[0] for row in cursor.fetchall()]

# Descobrir quais são os "Fantasmas" (estão no Excel mas não no Banco)
alunos_fantasmas = [sgde for sgde in sgdes_nos_relatorios if sgde not in alunos_banco]

if alunos_fantasmas:
    print(f"Encontrados {len(alunos_fantasmas)} alunos novos. Criando perfis básicos...")
    for sgde in alunos_fantasmas:
        # Insere apenas o Código SGDE, o resto fica vazio (NULL)
        cursor.execute("INSERT INTO tb_estudantes (codigo_sgde) VALUES (?)", (sgde,))
    conn.commit()
else:
    print("Todos os alunos já possuem cadastro prévio na base.")

# 6. Preparar colunas finais e inserir no Banco
df_final = df_acumulado[['codigo_sgde', 'disciplina', 'bim_1', 'bim_2', 'bim_3', 'bim_4', 'media_anual', 'total_faltas', 'situacao', 'ano_letivo']]
df_final.columns = ['codigo_sgde', 'disciplina', 'nota_bim1', 'nota_bim2', 'nota_bim3', 'nota_bim4', 'media_anual', 'total_faltas', 'situacao_final', 'ano_letivo']

print("\nInjetando os dados históricos no banco SQLite...")
# Usamos 'replace' aqui para sempre recriar a tabela limpa caso você rode o script várias vezes
df_final.to_sql('tb_historico_academico', conn, if_exists='replace', index=False)

print(f"✅ SUCESSO! Banco de dados atualizado com o histórico de {len(df_final)} matérias.")
conn.close()