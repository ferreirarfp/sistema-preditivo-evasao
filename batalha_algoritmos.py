import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("🥊 Preparando a arena: Random Forest vs XGBoost...")

# 1. Carregar os Dados Históricos (Onde sabemos quem realmente evadiu)
conn = sqlite3.connect('sistema_preditivo.db')
query = """
SELECT media_geral_ano, total_faltas_ano, distorcao_idade_serie, trabalha, situacao_final 
FROM tb_abt_preditiva 
WHERE ano_letivo IN (2024, 2025) AND situacao_final IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Preparação das Variáveis
# Criação do Alvo (1 = Evadiu, 0 = Ficou)
df['alvo_evasao'] = df['situacao_final'].apply(
    lambda x: 1 if any(p in str(x).upper() for p in ['TRANSFERIDO', 'DESISTENTE', 'ABANDONO']) else 0
)

# Tratando valores nulos (XGBoost lida bem com isso, mas a RF não)
df = df.fillna(0)

# Separando as Variáveis (X) do Alvo (y)
features = ['media_geral_ano', 'total_faltas_ano', 'distorcao_idade_serie', 'trabalha']

# Se a coluna 'trabalha' estiver como SIM/NAO, transformamos em 1/0
if df['trabalha'].dtype == 'O':
    df['trabalha'] = df['trabalha'].apply(lambda x: 1 if str(x).upper() == 'SIM' else 0)

X = df[features]
y = df['alvo_evasao']

# Separando em Treino (70%) e Teste (30%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"📊 Dados divididos: {len(X_treino)} para treino, {len(X_teste)} para teste.")

# 3. Treinamento dos Modelos
# Modelo 1: Random Forest
print("🌲 Treinando Random Forest...")
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
modelo_rf.fit(X_treino, y_treino)
pred_rf = modelo_rf.predict(X_teste)

# Modelo 2: XGBoost
print("🚀 Treinando XGBoost...")
modelo_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
modelo_xgb.fit(X_treino, y_treino)
pred_xgb = modelo_xgb.predict(X_teste)

# 4. Avaliação (O Foco é no RECALL - Capacidade de acertar quem vai evadir)
# Para a gestão escolar, é melhor um "falso alarme" do que deixar um aluno evadir sem aviso.
metricas = {
    'Algoritmo': ['Random Forest', 'XGBoost'],
    'Acurácia (Geral)': [accuracy_score(y_teste, pred_rf), accuracy_score(y_teste, pred_xgb)],
    'Recall (Acerto Evasão)': [recall_score(y_teste, pred_rf), recall_score(y_teste, pred_xgb)],
    'Precisão (Acerto Positivo)': [precision_score(y_teste, pred_rf, zero_division=0), precision_score(y_teste, pred_xgb, zero_division=0)]
}

df_metricas = pd.DataFrame(metricas)

print("\n🏆 RESULTADO DA BATALHA:")
print("-" * 50)
print(df_metricas.to_string(index=False))
print("-" * 50)

# 5. Visualização (Gráfico para o TCC)
plt.figure(figsize=(8, 5))
sns.barplot(x='Algoritmo', y='Recall (Acerto Evasão)', data=df_metricas, palette=['#8C8C8C', '#004C8A'])
plt.title('Comparativo de Modelos: Detecção de Alunos em Risco (Recall)', fontsize=14)
plt.ylabel('Taxa de Acerto na Evasão (%)', fontsize=12)
plt.ylim(0, 1.1)

# Adicionando as porcentagens em cima das barras
for index, row in df_metricas.iterrows():
    plt.text(index, row['Recall (Acerto Evasão)'] + 0.02, f"{row['Recall (Acerto Evasão)']*100:.1f}%", 
             color='black', ha="center", fontweight='bold', fontsize=12)

plt.tight_layout()
nome_grafico = 'grafico_batalha_modelos.png'
plt.savefig(nome_grafico)
print(f"\n📸 Gráfico salvo com sucesso: {nome_grafico} (Use no seu trabalho escrito!)")