import pandas as pd
import sqlite3
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("⚙️ Iniciando Engenharia de Atributos: Distorção Idade-Série...")

# ==========================================
# 1. CONFIGURAÇÕES 
CAMINHO_PLANILHA = 'dados_fichas_higienizados.xlsx' 

# Nome das colunas na planilha
COL_SGDE = 'CODIGO_SGDE' 
COL_NASCIMENTO = 'DATA_NASCIMENTO' 
COL_SERIE = 'Turma' 
# ==========================================

try:
    df_cad = pd.read_excel(CAMINHO_PLANILHA)
    print("✅ Planilha cadastral carregada com sucesso!")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {CAMINHO_PLANILHA}")
    exit()

# 2. Higienização e Cálculo da Idade
# Convertendo para texto e removendo '.0' do código SGDE
df_cad['codigo_sgde'] = df_cad[COL_SGDE].astype(str).str.replace(r'\.0$', '', regex=True)

# Garantindo que a data de nascimento é uma data real
df_cad[COL_NASCIMENTO] = pd.to_datetime(df_cad[COL_NASCIMENTO], errors='coerce')

# Calculando a idade do aluno no ano atual (2026)
ano_referencia = 2026
df_cad['idade_atual'] = ano_referencia - df_cad[COL_NASCIMENTO].dt.year

# 3. Engenharia da Variável (Feature Engineering)
# Dicionário da LDB para o Ensino Médio
idade_esperada_ldb = {
    '1': 15, # 1º Ano
    '2': 16, # 2º Ano
    '3': 17  # 3º Ano
}

def calcular_distorcao(linha):
    idade = linha['idade_atual']
    serie = str(linha[COL_SERIE])
    
    if pd.isna(idade):
        return 0 # Se não tiver data de nascimento, assume 0 para não quebrar o modelo
        
    # Identifica a série (procura por '1', '2' ou '3' no texto da turma)
    idade_ideal = None
    if '1' in serie:
        idade_ideal = idade_esperada_ldb['1']
    elif '2' in serie:
        idade_ideal = idade_esperada_ldb['2']
    elif '3' in serie:
        idade_ideal = idade_esperada_ldb['3']
        
    if idade_ideal is not None:
        distorcao = idade - idade_ideal
        # Só nos interessa a distorção positiva (atraso). Se for adiantado, zeramos.
        return max(0, distorcao) 
    return 0

# Aplicando a fórmula linha por linha
df_cad['distorcao_idade_serie'] = df_cad.apply(calcular_distorcao, axis=1)

print("🧠 Distorção Idade-Série calculada para todos os alunos.")

# 4. Injetando no Banco de Dados SQLite
print("💾 Conectando ao sistema_preditivo.db...")
conn = sqlite3.connect('sistema_preditivo.db')
cursor = conn.cursor()

# Adiciona a coluna na tabela caso ela ainda não exista
try:
    cursor.execute("ALTER TABLE tb_abt_preditiva ADD COLUMN distorcao_idade_serie INTEGER DEFAULT 0")
    print("✨ Nova coluna 'distorcao_idade_serie' criada no banco de dados!")
except sqlite3.OperationalError:
    # A coluna já existe, ignoramos o erro
    pass

alunos_atualizados = 0

for index, row in df_cad.iterrows():
    sgde = row['codigo_sgde']
    distorcao = int(row['distorcao_idade_serie'])
    
    # Atualiza TODOS os anos letivos desse aluno com a distorção calculada
    cursor.execute("""
        UPDATE tb_abt_preditiva 
        SET distorcao_idade_serie = ?
        WHERE codigo_sgde = ?
    """, (distorcao, sgde))
    
    # rowcount diz quantas linhas foram alteradas
    if cursor.rowcount > 0: 
        alunos_atualizados += 1

conn.commit()
conn.close()

print("--- RESUMO DA OPERAÇÃO ---")
print(f"🔄 Alunos atualizados com a nova métrica: {alunos_atualizados}")
print("✅ Banco de dados enriquecido com sucesso! Prontos para o XGBoost.")