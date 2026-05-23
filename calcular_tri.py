import pandas as pd
import numpy as np
import sqlite3
import warnings
from girth import twopl_mml # Modelo Logístico de 2 Parâmetros da TRI

warnings.filterwarnings('ignore')

print("Iniciando o Motor da Teoria de Resposta ao Item (TRI)...")

# 1. Carregar os relatórios do Evalbee
arquivo_simulado1 = r'D:\1_Arquivos\Trabalho\UNIVESP\resultado simulado 1.xlsx'
arquivo_simulado2 = r'D:\1_Arquivos\Trabalho\UNIVESP\resultado simulado 1.xlsx'

try:
    df_s1 = pd.read_excel(arquivo_simulado1)
    df_s2 = pd.read_excel(arquivo_simulado2)
    print("Ficheiros do Evalbee carregados com sucesso!")
except FileNotFoundError:
    print("Arquivos CSV não encontrados. Verifique os nomes.")
    exit()

# 2. Higienização e Preparação da Matriz
# No Evalbee, o SGDE do aluno geralmente fica na coluna 'Roll No'
coluna_id = 'Núm. da lista' if 'Núm. da lista' in df_s1.columns else df_s1.columns[0]

# Extrair apenas os alunos e formatar o SGDE
df_s1['codigo_sgde'] = df_s1[coluna_id].astype(str).str.replace(r'\.0$', '', regex=True)
df_s2['codigo_sgde'] = df_s2[coluna_id].astype(str).str.replace(r'\.0$', '', regex=True)

# Identificar as colunas que são questões (Geralmente Q1, Q2, ou apenas números)
# Vamos pegar as colunas de pontuação (que têm 1 para acerto e 0 para erro)
questoes_s1 = [col for col in df_s1.columns if col.startswith('Q') or col.isdigit()]
questoes_s2 = [col for col in df_s2.columns if col.startswith('Q') or col.isdigit()]

# Juntando os dois simulados pelo SGDE (Inner Join para pegar quem fez as duas provas)
df_tri = pd.merge(df_s1[['codigo_sgde'] + questoes_s1], 
                  df_s2[['codigo_sgde'] + questoes_s2], 
                  on='codigo_sgde', 
                  how='inner',
                  suffixes=('_S1', '_S2'))

# Separando apenas a matriz de respostas (1 = acerto, 0 = erro)
# Converte tudo para numérico garantindo que seja binário
matriz_respostas = df_tri.drop('codigo_sgde', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
matriz_respostas = (matriz_respostas > 0).astype(int)

print(f"Matriz criada: {matriz_respostas.shape[0]} alunos x {matriz_respostas.shape[1]} questões.")

# 3. Rodando o Algoritmo da TRI (Modelo de 2 Parâmetros: Dificuldade e Discriminação)
# A biblioteca girth exige que a matriz seja [itens x participantes] (transposta)
matriz_girth = matriz_respostas.values.T

print("Calculando o Theta (Proficiência) de cada aluno... Isso pode levar alguns segundos.")
try:
    # O twopl_mml calcula a habilidade do aluno isolando o peso do chute e da dificuldade da questão
    resultados_tri = twopl_mml(matriz_girth)
    
    # Extraindo a Habilidade (Theta)
    # Valores de Theta geralmente variam de -3 (muita dificuldade) a +3 (alta proficiência)
    thetas = resultados_tri['Ability']
    
    # Adicionando o Theta de volta ao nosso dataframe de alunos
    df_tri['proficiencia_tri'] = thetas
    print("Proficiência (Theta) calculada com sucesso!")
    
except Exception as e:
    print(f"Erro ao calcular TRI: {e}")
    print("Dica: A TRI precisa de variância. Se todos os alunos acertaram ou erraram tudo, o cálculo falha.")
    exit()

# 4. Injetando no Banco de Dados SQLite
print("Conectando ao sistema_preditivo.db para atualizar os perfis...")
conn = sqlite3.connect('sistema_preditivo.db')
cursor = conn.cursor()

# Adiciona a coluna na tabela caso ela ainda não exista
try:
    cursor.execute("ALTER TABLE tb_abt_preditiva ADD COLUMN proficiencia_tri REAL DEFAULT 0.0")
    print("Nova coluna 'proficiencia_tri' criada no banco!")
except sqlite3.OperationalError:
    pass # Coluna já existe

alunos_atualizados = 0

for index, row in df_tri.iterrows():
    sgde = row['codigo_sgde']
    # Arredondamos o Theta para 3 casas decimais
    theta = round(row['proficiencia_tri'], 3) 
    
    cursor.execute("""
        UPDATE tb_abt_preditiva 
        SET proficiencia_tri = ?
        WHERE codigo_sgde = ? AND ano_letivo = 2026
    """, (theta, sgde))
    
    if cursor.rowcount > 0: 
        alunos_atualizados += 1

conn.commit()
conn.close()

print("--- RESUMO DA OPERAÇÃO ---")
print(f"Alunos de 2026 enriquecidos com nota TRI: {alunos_atualizados}")
print("Próximo passo: Atualizar o Dashboard e treinar o XGBoost!")