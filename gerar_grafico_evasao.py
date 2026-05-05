import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# 1. Conectar ao banco
conn = sqlite3.connect('sistema_preditivo.db')

# 2. Extrair dados agrupando alunos únicos por situação e ano
query = """
SELECT ano_letivo, situacao_final, COUNT(DISTINCT codigo_sgde) as total_alunos
FROM tb_historico_academico
WHERE ano_letivo IN (2024, 2025) AND situacao_final IS NOT NULL 
GROUP BY ano_letivo, situacao_final
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 3. Filtrar apenas o cenário de Evasão ("Transferido")
df_evasao = df[df['situacao_final'].str.contains('Transferido', na=False, case=False)]

# 4. Criar o gráfico
plt.figure(figsize=(8, 5))
cores = ['#e74c3c', '#c0392b'] # Tons de vermelho para alerta
barras = plt.bar(df_evasao['ano_letivo'].astype(str), df_evasao['total_alunos'], color=cores)

# Formatação do visual
plt.title('Alunos Evadidos (Transferidos) no Período Noturno', fontsize=14, pad=15)
plt.xlabel('Ano Letivo', fontsize=12)
plt.ylabel('Quantidade de Alunos', fontsize=12)

# Colocar o número exato em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, altura + 1, int(altura), 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Salvar e mostrar
plt.tight_layout()
plt.savefig('grafico_evasao_24_25.png', dpi=300) # dpi=300 alta qualidade para o relatório
print("✅ Gráfico salvo com sucesso como 'grafico_evasao_24_25.png'")
plt.show()