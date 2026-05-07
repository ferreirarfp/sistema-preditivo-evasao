import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Conectar e Extrair os Dados (Apenas 2024 e 2025)
conn = sqlite3.connect('sistema_preditivo.db')

query = """
SELECT media_geral_ano, total_faltas_ano, situacao_final
FROM tb_abt_preditiva
WHERE ano_letivo IN (2024, 2025)
  AND situacao_final IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Total de registros históricos carregados: {len(df)}")

# 2. Preparação do "Gabarito" (Target)
# Vamos transformar texto em números: 1 para quem Evadiu (Transferido/Desistente), 0 para quem Ficou
def classifica_evasao(situacao):
    situacao = str(situacao).upper()
    if 'TRANSFERIDO' in situacao or 'DESISTENTE' in situacao or 'ABANDONO' in situacao:
        return 1 # Evadiu (Risco Alto)
    return 0 # Permaneceu

df['alvo_evasao'] = df['situacao_final'].apply(classifica_evasao)

# Variáveis Preditivas (X) e Alvo (y)
X = df[['media_geral_ano', 'total_faltas_ano']]
y = df['alvo_evasao']

# Tratar possíveis valores nulos nas notas (preencher com a média)
X = X.fillna(X.mean())

# 3. Separar em Treino (80%) e Teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTreinando o modelo de Floresta Aleatória (Random Forest)...")

# 4. Criar e Treinar a Inteligência Artificial
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
modelo_rf.fit(X_treino, y_treino)

# 5. Fazer Previsões e Avaliar
previsoes = modelo_rf.predict(X_teste)
acuracia = accuracy_score(y_teste, previsoes)

print(f"\n✅ ACURÁCIA DO MODELO BASELINE: {acuracia * 100:.2f}%")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_teste, previsoes, target_names=['Permaneceu (0)', 'Evadiu (1)']))

# 6. O Bônus Visual: Gráfico de Importância das Variáveis
importancias = modelo_rf.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importancias, y=features, palette='viridis')
plt.title('O que pesa mais na Evasão? (Importância das Variáveis)')
plt.xlabel('Peso de Importância (0 a 1)')
plt.ylabel('Variável')
plt.tight_layout()
plt.savefig('importancia_variaveis_baseline.png', dpi=300)
print("\nGráfico de importância salvo como 'importancia_variaveis_baseline.png'.")
plt.show()