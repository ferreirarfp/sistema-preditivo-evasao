import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. Configurações da Página
st.set_page_config(page_title="Sistema Preditivo - Ferreira Soluti", layout="wide")
st.title("🎓 Sistema Preditivo de Evasão Escolar")
st.markdown("---")

# 2. Funções de Carga e Treinamento
@st.cache_data
def carregar_dados():
    conn = sqlite3.connect('sistema_preditivo.db')
    # CORREÇÃO: O 'SELECT *' já vai trazer todas as colunas automaticamente!
    query = "SELECT * FROM tb_abt_preditiva"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_resource
def treinar_modelo(df):
    # Treino com dados históricos (2024/2025)
    df_treino = df[(df['ano_letivo'].isin([2024, 2025])) & (df['situacao_final'].notnull())].copy()
    
    df_treino['alvo_evasao'] = df_treino['situacao_final'].apply(
        lambda x: 1 if any(p in str(x).upper() for p in ['TRANSFERIDO', 'DESISTENTE', 'ABANDONO']) else 0
    )
    
    # ATUALIZAÇÃO: Incluindo a distorção como variável para o modelo
    features_list = ['media_geral_ano', 'total_faltas_ano', 'distorcao_idade_serie']
    X = df_treino[features_list].fillna(0)
    y = df_treino['alvo_evasao']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    modelo.fit(X, y)
    
    return modelo, features_list

# 3. Inicializando Dados e IA
df_abt = carregar_dados()
modelo_rf, features = treinar_modelo(df_abt)

# 4. Criando as Abas na Tela
aba1, aba2 = st.tabs(["📊 Visão da Gestão (Panorama)", "🔎 Busca Individual (Previsão)"])

# ================= ABA 1: VISÃO DA GESTÃO =================
with aba1:
    st.header("Panorama Histórico de Evasão")
    
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.markdown("**O que pesa mais no abandono?**")
        importancias = modelo_rf.feature_importances_
        fig, ax = plt.subplots(figsize=(4, 3)) 
        sns.barplot(x=importancias, y=features, palette='viridis', ax=ax)
        ax.set_xlabel('Peso')
        ax.set_ylabel('')
        sns.despine(left=True, bottom=True) 
        st.pyplot(fig)
        
    with col2:
        st.markdown("**Base de Dados Atual (ABT 2.0)**")
        # ATUALIZAÇÃO: A distorção agora aparece na tabela principal
        colunas_exibicao = ['codigo_sgde', 'ano_letivo', 'media_geral_ano', 'total_faltas_ano', 'trabalha', 'distorcao_idade_serie']
        st.dataframe(
            df_abt[colunas_exibicao], 
            use_container_width=True, 
            height=350 
        )

# ================= ABA 2: BUSCA INDIVIDUAL =================
with aba2:
    st.header("🔎 Diagnóstico Detalhado por Estudante")
    
    sgde_busca = st.text_input("Digite o Código SGDE do aluno:")
    
    if sgde_busca:
        aluno_df = df_abt[df_abt['codigo_sgde'].astype(str) == sgde_busca]
        
        if aluno_df.empty:
            st.error("SGDE não localizado.")
        else:
            aluno_atual = aluno_df.iloc[-1:]
            col_esq, col_dir = st.columns([1.5, 1])
            
            with col_esq:
                st.subheader("Veredito da IA")
                # ATUALIZAÇÃO: Previsão agora usa os 3 fatores (Nota, Falta, Distorção)
                X_aluno = aluno_atual[features].fillna(0)
                probabilidade = modelo_rf.predict_proba(X_aluno)[0][1] * 100
                
                if probabilidade >= 60:
                    st.error(f"RISCO ALTO: {probabilidade:.1f}%")
                elif probabilidade >= 30:
                    st.warning(f"RISCO MÉDIO: {probabilidade:.1f}%")
                else:
                    st.success(f"RISCO BAIXO: {probabilidade:.1f}%")
                st.progress(int(probabilidade))
                
                st.markdown("---")
                st.subheader("📋 Prontuário e Plano de Ação")
                
                aluno_nota = aluno_atual['media_geral_ano'].values[0]
                aluno_faltas = aluno_atual['total_faltas_ano'].values[0]
                aluno_distorcao = aluno_atual['distorcao_idade_serie'].values[0]
                
                motivos = []
                if aluno_faltas > df_abt['total_faltas_ano'].mean() * 1.5:
                    motivos.append("Absenteísmo Crítico")
                if aluno_nota < df_abt['media_geral_ano'].mean():
                    motivos.append("Defasagem Pedagógica")
                
                # ATUALIZAÇÃO: Alerta de distorção no prontuário
                if aluno_distorcao > 0:
                    motivos.append(f"Distorção Idade-Série: {int(aluno_distorcao)} ano(s) de atraso")

                if motivos:
                    with st.warning("Fatores de Risco Detectados:"):
                        for m in motivos:
                            st.markdown(f"- {m}")
                    
                    st.markdown("**Intervenção Recomendada:**")
                    if aluno_distorcao >= 2:
                        st.info("👉 **Foco Social:** Aluno com alta distorção. Priorizar acolhimento focado no Projeto de Vida e terminalidade específica.")
                    elif aluno_nota < 5:
                        st.info("👉 **Foco Pedagógico:** Encaminhar para nivelamento de competências básicas.")
                else:
                    st.success("O aluno apresenta indicadores estáveis.")

            with col_dir:
                st.subheader("Comparativo com a Turma")
                # Gráfico de termômetro mantido conforme o modelo aprovado
                fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3.5))
                
                ax1.bar(['Média'], [df_abt['media_geral_ano'].mean()], width=0.6, color='#E5E5E5')
                ax1.bar(['Média'], [aluno_nota], width=0.3, color='#BBDDF0', edgecolor='#004C8A', linewidth=1.5)
                ax1.set_ylim(0, 10)
                ax1.text(0, aluno_nota + 0.3, f'{aluno_nota:.1f}', ha='center', fontweight='bold')
                
                max_f = max(df_abt['total_faltas_ano'].mean(), aluno_faltas) * 1.2
                ax2.bar(['Faltas'], [df_abt['total_faltas_ano'].mean()], width=0.6, color='#E5E5E5')
                ax2.bar(['Faltas'], [aluno_faltas], width=0.3, color='#8C8C8C')
                ax2.set_ylim(0, max_f)
                ax2.text(0, aluno_faltas + (max_f * 0.05), f'{int(aluno_faltas)}', ha='center', fontweight='bold')
                
                for ax in [ax1, ax2]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])
                
                plt.tight_layout()
                st.pyplot(fig_comp)