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
            
            # --- COLUNA DE RESULTADO E RISCO ---
            col_esq, col_dir = st.columns([1, 2])
            
            with col_esq:
                st.subheader("Veredito da IA")
                X_aluno = aluno_atual[['media_geral_ano', 'total_faltas_ano']].fillna(0)
                probabilidade = modelo_rf.predict_proba(X_aluno)[0][1] * 100
                
                if probabilidade >= 60:
                    st.error(f"RISCO ALTO: {probabilidade:.1f}%")
                elif probabilidade >= 30:
                    st.warning(f"RISCO MÉDIO: {probabilidade:.1f}%")
                else:
                    st.success(f"RISCO BAIXO: {probabilidade:.1f}%")
                
                st.progress(int(probabilidade))
                
                # Exibe dados do Forms para contexto humano
                st.info(f"**Situação de Trabalho:** {aluno_atual['trabalha'].values[0]}")
                st.info(f"**Sono:** {aluno_atual['horas_sono'].values[0]}")

            with col_dir:
                st.subheader("Por que o risco é esse? (Comparativo)")
                
                # Cálculo de médias da turma para comparação
                media_turma = df_abt['media_geral_ano'].mean()
                faltas_turma = df_abt['total_faltas_ano'].mean()
                
                aluno_nota = aluno_atual['media_geral_ano'].values[0]
                aluno_faltas = aluno_atual['total_faltas_ano'].values[0]

                # Gráfico de comparação simples
                fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
                categorias = ['Média Acadêmica', 'Volume de Faltas']
                valores_aluno = [aluno_nota, aluno_faltas]
                valores_turma = [media_turma, faltas_turma]

                x = range(len(categorias))
                ax_comp.bar(x, valores_turma, width=0.4, label='Média da Turma', align='edge', color='lightgray')
                ax_comp.bar(x, valores_aluno, width=-0.4, label='Este Aluno', align='edge', color='#1f77b4')
                
                ax_comp.set_xticks(x)
                ax_comp.set_xticklabels(categorias)
                ax_comp.legend()
                st.pyplot(fig_comp)

            # --- PARTE NOVA: PLANO DE INTERVENÇÃO ---
            st.markdown("---")
            st.subheader("📋 Plano de Intervenção Sugerido")
            
            # Lógica simples de "Explainable AI" para a gestão
            motivos = []
            if aluno_faltas > faltas_turma * 1.5:
                motivos.append("- **Absenteísmo Crítico:** O aluno falta muito acima da média da turma.")
            if aluno_nota < media_turma:
                motivos.append("- **Defasagem Pedagógica:** O desempenho está abaixo da média esperada.")
            if "8 horas ou mais" in str(aluno_atual['carga_horaria'].values[0]):
                motivos.append("- **Sobrecarga de Trabalho:** A jornada de trabalho integral compromete o tempo de estudo.")
            if "Menos de 5 horas" in str(aluno_atual['horas_sono'].values[0]):
                motivos.append("- **Privação de Sono:** O cansaço físico pode ser o gatilho da desmotivação.")

            if motivos:
                st.write("A análise detectou os seguintes fatores que elevam o risco:")
                for m in motivos:
                    st.write(m)
                
                st.markdown("**Ação recomendada para a coordenação:**")
                if "Trabalho" in str(motivos):
                    st.write("👉 Chamar o aluno para alinhar flexibilização de horários ou regime especial de estudos.")
                else:
                    st.write("👉 Encaminhar para reforço escolar focado nas habilidades da BNCC com maior dificuldade.")
            else:
                st.write("O aluno apresenta indicadores estáveis no momento.")