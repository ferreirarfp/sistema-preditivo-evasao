import pandas as pd
import sqlite3
import warnings

warnings.filterwarnings('ignore')

print("🚀 Iniciando processamento do 1º Bimestre de 2026...")

caminho_arquivo = 'dados_historicos/relatorio_3D_2026_1BIM_noturno_atual.xls'

try:
    df_bruto = pd.read_excel(caminho_arquivo, skiprows=2)
    print(f"✅ Planilha carregada! Total de linhas brutas: {len(df_bruto)}")
except FileNotFoundError:
    print(f"❌ Arquivo não encontrado: {caminho_arquivo}")
    exit()
except Exception as e:
    print(f"❌ Erro ao ler a planilha: {e}")
    exit()

# --- CORREÇÃO AQUI ---
# Pega o nome exato da PRIMEIRA coluna do arquivo (índice 0)
coluna_codigo = df_bruto.columns[0]
print(f"🔍 Usando a coluna '{coluna_codigo}' como identificador SGDE.")

df_limpo = df_bruto.dropna(subset=[coluna_codigo]).copy()
df_limpo['codigo_sgde'] = df_limpo[coluna_codigo].astype(str).str.replace(r'\.0$', '', regex=True)

colunas_notas = [col for col in df_limpo.columns if col.upper().startswith('N') and len(col) <= 3]
colunas_faltas = [col for col in df_limpo.columns if col.upper().startswith('F') and len(col) <= 3]

for col in colunas_notas + colunas_faltas:
    df_limpo[col] = pd.to_numeric(df_limpo[col], errors='coerce')

df_limpo['media_geral_ano'] = df_limpo[colunas_notas].mean(axis=1)
df_limpo['total_faltas_ano'] = df_limpo[colunas_faltas].sum(axis=1)

df_agrupado = df_limpo.groupby('codigo_sgde', as_index=False).agg({
    'media_geral_ano': 'mean',
    'total_faltas_ano': 'sum'
})

print(f"🧠 Dados processados. Alunos únicos encontrados: {len(df_agrupado)}")

print("💾 Conectando ao sistema_preditivo.db...")
conn = sqlite3.connect('sistema_preditivo.db')
cursor = conn.cursor()

alunos_atualizados = 0
alunos_inseridos = 0

for index, row in df_agrupado.iterrows():
    sgde = row['codigo_sgde']
    media = round(row['media_geral_ano'], 2) if pd.notna(row['media_geral_ano']) else 0.0
    faltas = int(row['total_faltas_ano']) if pd.notna(row['total_faltas_ano']) else 0
    ano = 2026
    
    cursor.execute("SELECT 1 FROM tb_abt_preditiva WHERE codigo_sgde = ? AND ano_letivo = ?", (sgde, ano))
    existe = cursor.fetchone()
    
    if existe:
        cursor.execute("""
            UPDATE tb_abt_preditiva 
            SET media_geral_ano = ?, total_faltas_ano = ?
            WHERE codigo_sgde = ? AND ano_letivo = ?
        """, (media, faltas, sgde, ano))
        alunos_atualizados += 1
    else:
        cursor.execute("""
            INSERT INTO tb_abt_preditiva (codigo_sgde, ano_letivo, media_geral_ano, total_faltas_ano)
            VALUES (?, ?, ?, ?)
        """, (sgde, ano, media, faltas))
        alunos_inseridos += 1

conn.commit()
conn.close()

print("--- RESUMO DA OPERAÇÃO ---")
print(f"🔄 Alunos atualizados: {alunos_atualizados}")
print(f"➕ Novos alunos inseridos: {alunos_inseridos}")
print("✅ Banco de dados atualizado com sucesso!")