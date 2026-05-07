import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. Configurações da Página
st.set_page_config(page_title="Sistema Preditivo", layout="wide")
st.title("🎓 Sistema Preditivo de Evasão Escolar")
st.markdown("---")

# 2. Funções de Carga e Treinamento (Com Cache para velocidade)
@st.cache_data
def carregar_dados():
    conn = sqlite3.connect('sistema_preditivo.db')
    df = pd.read_sql_query("SELECT * FROM tb_abt_preditiva", conn)
    conn.close()
    return df

@st.cache_resource
def treinar_modelo(df):
    # Filtra apenas o histórico (2024 e 2025) para treinar
    df_treino = df[(df['ano_letivo'].isin([2024, 2025])) & (df['situacao_final'].notnull())].copy()
    
    # Prepara o alvo (1 para evasão, 0 para permanência)
    df_treino['alvo_evasao'] = df_treino['situacao_final'].apply(
        lambda x: 1 if any(p in str(x).upper() for p in ['TRANSFERIDO', 'DESISTENTE', 'ABANDONO']) else 0
    )
    
    X = df_treino[['media_geral_ano', 'total_faltas_ano']].fillna(df_treino[['media_geral_ano', 'total_faltas_ano']].mean())
    y = df_treino['alvo_evasao']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    modelo.fit(X, y)
    
    return modelo, X.columns

# 3. Inicializando Dados e IA
df_abt = carregar_dados()
modelo_rf, features = treinar_modelo(df_abt)

# 4. Criando as Abas na Tela
aba1, aba2 = st.tabs(["📊 Visão da Gestão (Panorama)", "🔎 Busca Individual (Previsão)"])

# ================= ABA 1: VISÃO DA GESTÃO =================
with aba1:
    st.header("Panorama Histórico de Evasão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("O que pesa mais no abandono?")
        importancias = modelo_rf.feature_importances_
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=importancias, y=features, palette='viridis', ax=ax)
        ax.set_xlabel('Peso de Importância')
        ax.set_ylabel('Variável')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Base de Dados Atual (ABT 2.0)")
        st.dataframe(df_abt[['codigo_sgde', 'ano_letivo', 'media_geral_ano', 'total_faltas_ano', 'trabalha']], use_container_width=True)

# ================= ABA 2: BUSCA INDIVIDUAL =================
with aba2:
    st.header("Alerta Precoce por Aluno")
    
    # Barra lateral de busca
    sgde_busca = st.text_input("Digite o Código SGDE do aluno para avaliar o risco:")
    
    if sgde_busca:
        # CORREÇÃO: Convertemos a coluna 'codigo_sgde' para string para comparar com o input
        aluno_df = df_abt[df_abt['codigo_sgde'].astype(str) == sgde_busca]
        
        if aluno_df.empty:
            st.error(f"Aluno com SGDE {sgde_busca} não encontrado. Verifique se o número está correto.")
        else:
            # Pega o registro mais recente (última linha)
            aluno_atual = aluno_df.iloc[-1:]
            
            st.markdown(f"**Analisando o SGDE:** `{sgde_busca}` (Ano: {aluno_atual['ano_letivo'].values[0]})")
            
            # Dados reais do Forms / Sistema
            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("Média Geral", round(aluno_atual['media_geral_ano'].values[0], 2))
            col_info2.metric("Total de Faltas", aluno_atual['total_faltas_ano'].values[0])
            col_info3.metric("Situação de Trabalho", aluno_atual['trabalha'].values[0] if pd.notnull(aluno_atual['trabalha'].values[0]) else "Não informada")
            
            # Previsão da IA (Probabilidade)
            X_aluno = aluno_atual[['media_geral_ano', 'total_faltas_ano']].fillna(0)
            probabilidade = modelo_rf.predict_proba(X_aluno)[0][1] * 100  # Pega a propabilidade da classe 1 (Evasão)
            
            st.markdown("---")
            st.subheader("🧠 Diagnóstico da Inteligência Artificial")
            
            if probabilidade < 30:
                st.success(f"Risco Baixo de Evasão: {probabilidade:.1f}%")
                st.progress(int(probabilidade))
            elif probabilidade < 60:
                st.warning(f"Risco Moderado de Evasão: {probabilidade:.1f}% - Atenção requerida.")
                st.progress(int(probabilidade))
            else:
                st.error(f"⚠️ RISCO ALTO DE EVASÃO: {probabilidade:.1f}% - Intervenção imediata recomendada!")
                st.progress(int(probabilidade))