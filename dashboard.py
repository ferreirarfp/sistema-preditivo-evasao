import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import re
import plotly.graph_objects as go
import plotly.express as px


# Configurações da Página e Tema Estético
st.set_page_config(page_title="Dashboard Preditivo - Ferreira Soluti", layout="wide")
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

st.title("🎓 Sistema Analítico Preditivo de Evasão Escolar (Versão Executiva 2.5)")
st.markdown("---")

# Função auxiliar para extrair números do texto de horas de sono
def tratar_sono(x):
    if pd.isna(x) or str(x).strip() == '':
        return 6.0
    nums = re.findall(r'\d+', str(x))
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    elif len(nums) == 1:
        return float(nums[0])
    else:
        return 6.0

# Funções de Carga e Treinamento
@st.cache_data
def carregar_dados():
    conn = sqlite3.connect('sistema_preditivo.db')
    query = "SELECT * FROM tb_abt_preditiva"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_resource
def treinar_modelo(df):
    df_treino = df[(df['ano_letivo'].isin([2024, 2025])) & (df['situacao_final'].notnull())].copy()
    
    df_treino['alvo_evasao'] = df_treino['situacao_final'].apply(
        lambda x: 1 if any(p in str(x).upper() for p in ['TRANSFERIDO', 'DESISTENTE', 'ABANDONO']) else 0
    )
    
    df_treino['trabalha'] = df_treino['trabalha'].apply(
        lambda x: 0 if pd.isna(x) or str(x).strip().upper() in ['NÃO', 'NAO', '0', ''] else 1
    )
    
    features_list = ['media_geral_ano', 'total_faltas_ano', 'distorcao_idade_serie', 'trabalha']
    X = df_treino[features_list].fillna(0)
    y = df_treino['alvo_evasao']
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    modelo.fit(X, y)
    
    return modelo, features_list

# Inicializando e Processando a Base Geral
df_abt = carregar_dados()
modelo_rf, features = treinar_modelo(df_abt)

# Preparando o Dataset de 2026 com as predições em lote para os gráficos globais
df_2026 = df_abt[df_abt['ano_letivo'] == 2026].copy()
df_2026['trabalha_num'] = df_2026['trabalha'].apply(lambda x: 0 if pd.isna(x) or str(x).strip().upper() in ['NÃO', 'NAO', '0', ''] else 1)
df_2026['horas_sono_num'] = df_2026['horas_sono'].apply(tratar_sono)

if 'proficiencia_tri' not in df_2026.columns:
    df_2026['proficiencia_tri'] = 0.0
else:
    df_2026['proficiencia_tri'] = df_2026['proficiencia_tri'].fillna(0.0)

# IA calcula a probabilidade em lote para o ano corrente
X_2026 = df_2026[['media_geral_ano', 'total_faltas_ano', 'distorcao_idade_serie', 'trabalha_num']].fillna(0)
X_2026.columns = features
df_2026['prob_evasao'] = modelo_rf.predict_proba(X_2026)[:, 1] * 100

def categorizar_risco(p):
    if p >= 60: return 'Alto Risco'
    elif p >= 30: return 'Médio Risco'
    else: return 'Baixo Risco'
df_2026['classe_risco'] = df_2026['prob_evasao'].apply(categorizar_risco)

# ================= SIDEBAR DE FILTROS GLOBAIS =================
st.sidebar.header("⚙️ Filtros do Painel")
st.sidebar.markdown("Filtre a visão macro da escola.")

filtro_risco = st.sidebar.multiselect(
    "Nível de Risco:",
    options=['Alto Risco', 'Médio Risco', 'Baixo Risco'],
    default=['Alto Risco', 'Médio Risco', 'Baixo Risco']
)

filtro_trabalho = st.sidebar.selectbox(
    "Situação Laboral:",
    options=["Todos os Alunos", "Apenas Trabalhadores", "Não Trabalhadores"]
)

# Aplicação dos filtros na base de dados ativa
df_filtrado = df_2026[df_2026['classe_risco'].isin(filtro_risco)].copy()
if filtro_trabalho == "Apenas Trabalhadores":
    df_filtrado = df_filtrado[df_filtrado['trabalha_num'] == 1]
elif filtro_trabalho == "Não Trabalhadores":
    df_filtrado = df_filtrado[df_filtrado['trabalha_num'] == 0]

# ================= CRIAÇÃO DAS ABAS =================
aba1, aba2, aba3 = st.tabs(["📊 Visão Macro da Gestão", "🔎 Diagnóstico Individual", "📚 Mapeamento Pedagógico (Simulado)"])


# ================= ABA 1: PANORAMA DA GESTÃO =================
with aba1:
    st.header("Análise de Indicadores Educacionais e Estatísticos (2026)")
    
    # Linha de KPIs de Alto Impacto Visual
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total de Estudantes Monitorados", len(df_2026))
    with kpi2:
        st.metric("Alunos em Risco Crítico (>=60%)", len(df_2026[df_2026['classe_risco'] == 'Alto Risco']))
    with kpi3:
        st.metric("Média de Proficiência Real (TRI)", f"{df_2026['proficiencia_tri'].mean():.2f}")
    with kpi4:
        pct_trabalha = (len(df_2026[df_2026['trabalha_num'] == 1]) / len(df_2026)) * 100
        st.metric("Taxa de Inserção no Mercado", f"{pct_trabalha:.1f}%")
        
    st.markdown("---")
    
    # Primeira Linha de Gráficos: Modelagem e Volumetria de Risco
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown("#### 🌲 Pesos Importância das Variáveis (Random Forest)")
        
        df_imp = pd.DataFrame({
            'Variável': ['Média Escolar', 'Faltas no Ano', 'Distorção Idade', 'Trabalha'], 
            'Peso': modelo_rf.feature_importances_
        }).sort_values(by='Peso', ascending=True) # Ascending=True para o Plotly empilhar do maior para o menor no topo
        
        fig_imp = px.bar(
            df_imp, 
            x='Peso', 
            y='Variável', 
            orientation='h', 
            text_auto='.1%', # Formata automaticamente o texto na barra como porcentagem
            color='Peso', 
            color_continuous_scale='Blues'
        )
        
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'),
            showlegend=False, 
            coloraxis_showscale=False, # Esconde a barra de cores lateral
            margin=dict(t=20, b=20, l=20, r=20), 
            height=350,
            xaxis=dict(visible=False), # Esconde o eixo X inferior para um visual mais limpo
            yaxis=dict(title='')
        )
        
        fig_imp.update_traces(
            textfont=dict(color='white', size=12), 
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Peso no Algoritmo: %{x:.1%}<extra></extra>"
        )
        
        st.plotly_chart(fig_imp, use_container_width=True)

    with g2:
        st.markdown("#### 🚨 Proporção das Classes de Risco")
        
        # Prepara os dados
        contagem_risco = df_filtrado['classe_risco'].value_counts().reset_index()
        contagem_risco.columns = ['Classe', 'Total']
        
        # Cria o Gráfico de Rosca (Donut) interativo
        fig_donut = px.pie(
            contagem_risco, 
            names='Classe', 
            values='Total', 
            hole=0.6, # Define a espessura da "rosca"
            color='Classe',
            color_discrete_map={'Baixo Risco': '#2ecc71', 'Médio Risco': '#f1c40f', 'Alto Risco': '#e74c3c'}
        )
        
        # Formata o texto para aparecer fora do gráfico e sem legenda lateral para poupar espaço
        fig_donut.update_traces(
            textposition='outside', 
            textinfo='label+percent', # Mostra o nome e a porcentagem diretamente
            hovertemplate="<b>%{label}</b><br>Alunos: %{value}<extra></extra>",
            textfont=dict(color='white', size=12)
        )
        
        # Estilização do fundo e adição do número total no centro
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            annotations=[dict(text=f"Total<br><b>{len(df_filtrado)}</b>", x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color='white'))]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    
    # Segunda Linha de Gráficos: Análise Avançada e Científica (Cruzamentos)
    g3, g4 = st.columns(2)
    
    with g3:
        st.markdown("#### 🎯 Dispersão Pedagógica: Média do Boletim vs. Proficiência TRI")
        
        fig_scatter = px.scatter(
            df_filtrado, 
            x='media_geral_ano', 
            y='proficiencia_tri', 
            color='classe_risco', 
            color_discrete_map={'Baixo Risco': '#2ecc71', 'Médio Risco': '#f1c40f', 'Alto Risco': '#e74c3c'},
            hover_data=['codigo_sgde'], # Mostra o código SGDE ao passar o rato no ponto!
            labels={
                'media_geral_ano': 'Média Geral do Boletim', 
                'proficiencia_tri': 'Proficiência Latente (Theta TRI)', 
                'classe_risco': 'Veredito da IA'
            }
        )
        
        fig_scatter.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')))
        
        # Adiciona as linhas de referência de cruzamento (Média 6.0 e TRI 0.0)
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_scatter.add_vline(x=6, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
        
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='white'),
            margin=dict(t=20, b=20, l=20, r=20), 
            height=350,
            legend=dict(
                orientation="h", # Coloca a legenda na horizontal
                yanchor="bottom", y=1.02, 
                xanchor="center", x=0.5,
                title=""
            ),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

    with g4:
        st.markdown("#### 💤 Impacto da Rotina: Distribuição de Sono por Ocupação")
        
        # Mapeia a variável para texto amigável
        df_filtrado['Trabalha?'] = df_filtrado['trabalha_num'].map({1: 'Sim / Aprendiz', 0: 'Não'})
        
        # Cria o Boxplot interativo no Plotly
        fig_box = px.box(
            df_filtrado, 
            x='Trabalha?', 
            y='horas_sono_num', 
            color='Trabalha?',
            color_discrete_map={'Sim / Aprendiz': '#f39c12', 'Não': '#3498db'}, # Laranja (Alerta) e Azul (Padrão)
            points="all", # Exibe os pontos dos alunos reais ao lado da caixa!
            labels={'Trabalha?': 'Estudante Atua no Mercado de Trabalho?', 'horas_sono_num': 'Horas de Sono Diárias'}
        )
        
        # Formatação para o Dark Mode e limpeza visual
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)') # Linhas de grade mais suaves
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
        # ... (fim do código do gráfico fig_box) ...

    st.markdown("---")
    st.subheader("📋 Lista de Atenção Prioritária (Fila de Intervenção)")
    
    # Isola os alunos de maior risco, ordena do pior pro melhor e pega os 5 primeiros
    df_criticos = df_filtrado[df_filtrado['classe_risco'] == 'Alto Risco'].sort_values(by='prob_evasao', ascending=False)
    
    col_tabela, col_export = st.columns([2, 1])
    
    with col_tabela:
        if not df_criticos.empty:
            # Mostra uma tabela limpa só com os dados essenciais
            df_display = df_criticos[['codigo_sgde', 'media_geral_ano', 'total_faltas_ano', 'proficiencia_tri', 'prob_evasao']].copy()
            df_display.columns = ['Código SGDE', 'Média Anual', 'Total de Faltas', 'Proficiência (TRI)', 'Risco IA (%)']
            df_display['Risco IA (%)'] = df_display['Risco IA (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display.head(5), use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum aluno em situação de risco crítico nos filtros atuais. Excelente trabalho!")
            
    with col_export:
        st.markdown("**Ação da Coordenação:**")
        st.markdown("Baixe o relatório completo dos alunos em alerta para encaminhamento ao conselho tutelar ou equipe psicopedagógica.")
        
        if not df_criticos.empty:
            # Transforma os dados em CSV para download
            csv_export = df_criticos[['codigo_sgde', 'media_geral_ano', 'total_faltas_ano', 'prob_evasao']].to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Baixar Relatório (CSV)",
                data=csv_export,
                file_name="relatorio_alunos_criticos.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.button("📥 Baixar Relatório (CSV)", disabled=True)


# ================= ABA 2: PRONTUÁRIO INDIVIDUAL (VERSÃO ENTERPRISE) =================
with aba2:
    st.header("🔎 Diagnóstico Detalhado por Estudante")
    sgde_busca = st.text_input("Insira o Código SGDE do aluno para consulta (ex: 1221313):")
    
    if sgde_busca:
        aluno_df = df_abt[df_abt['codigo_sgde'].astype(str) == sgde_busca]
        
        if aluno_df.empty:
            st.error("Código SGDE não localizado na base consolidada.")
        else:
            aluno_atual = df_2026[df_2026['codigo_sgde'].astype(str) == sgde_busca].iloc[-1:].copy()
            
            # --- 1. CABEÇALHO: PERFIL SOCIODEMOGRÁFICO ---
            st.markdown("### 👤 Perfil do Estudante")
            
            aluno_nota = aluno_atual['media_geral_ano'].values[0]
            aluno_faltas = aluno_atual['total_faltas_ano'].values[0]
            aluno_distorcao = aluno_atual['distorcao_idade_serie'].values[0]
            aluno_tri = aluno_atual['proficiencia_tri'].values[0]
            aluno_sono = aluno_atual['horas_sono_num'].values[0]
            trabalha = "Sim (Mercado/Aprendiz)" if aluno_atual['trabalha_num'].values[0] == 1 else "Não"
            
            # Utilizar colunas para criar "Cartões" de perfil
            c1, c2, c3, c4 = st.columns(4)
            c1.info(f"**Atraso Idade-Série:**\n\n{int(aluno_distorcao)} ano(s)")
            c2.info(f"**Inserção Laboral:**\n\n{trabalha}")
            c3.info(f"**Horas de Sono:**\n\n{aluno_sono}h / dia")
            c4.info(f"**Proficiência (TRI):**\n\n{aluno_tri:.2f}")
            
            st.markdown("---")
            
            # --- 2. DIVISÃO PRINCIPAL: ALERTAS vs GRÁFICOS ---
            col_esq, col_dir = st.columns([1.2, 1.8]) # A coluna direita (gráficos) fica um pouco maior
            
            with col_esq:
                st.subheader("Veredito Analítico da IA")
                probabilidade = aluno_atual['prob_evasao'].values[0]
                classe_atual = aluno_atual['classe_risco'].values[0]
                
                # Cores dinâmicas para a barra de progresso baseadas no risco
                cor_progresso = "red" if probabilidade >= 60 else "orange" if probabilidade >= 30 else "green"
                
                if classe_atual == 'Alto Risco':
                    st.error(f"🚨 ALERTA CRÍTICO: RISCO ALTO DE EVASÃO ({probabilidade:.1f}%)")
                elif classe_atual == 'Médio Risco':
                    st.warning(f"⚠️ ATENÇÃO: RISCO MÉDIO DE EVASÃO ({probabilidade:.1f}%)")
                else:
                    st.success(f"✅ SITUAÇÃO ESTÁVEL: RISCO BAIXO DE EVASÃO ({probabilidade:.1f}%)")
                
                st.progress(int(probabilidade))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📋 Plano de Ação Prescritivo")
                
                motivos = []
                if aluno_faltas > df_abt['total_faltas_ano'].mean() * 1.5:
                    motivos.append("Absenteísmo Crítico detetado na frequência acumulada.")
                
                if aluno_sono < 6 and aluno_atual['trabalha_num'].values[0] == 1:
                    st.error("🚨 **Alerta de Sobrecarga:** Estudante inserido no mercado de trabalho com severa privação de descanso biológico. Intervenção sugerida: Flexibilização de prazos.")
                
                if aluno_tri != 0:
                    if aluno_tri < -1.0:
                        motivos.append("Proficiência TRI Crítica: Padrão inconsistente de respostas (possível chute).")
                    elif aluno_tri > 0.5 and aluno_nota < 6:
                        st.info("💡 **Incoerência Oportuna:** Excelente habilidade cognitiva (TRI), mas baixas notas formais. Forte indício de desmotivação ou falta de tempo para entregas.")
                
                if motivos:
                    with st.warning("Fatores de Risco Mapeados:"):
                        for m in motivos:
                            st.markdown(f"- {m}")

            with col_dir:
                st.subheader("Análise Multidimensional (Radar)")
                
                # --- 3. GRÁFICO DE RADAR COM PLOTLY ---
                # Normalização simplificada dos dados (0 a 10) para o radar ter escalas compatíveis
                media_escola = df_abt['media_geral_ano'].mean()
                nota_norm = (aluno_nota / 10) * 10
                escola_nota_norm = (media_escola / 10) * 10
                
                # Para as faltas, quanto mais alto, pior. Logo, invertemos a lógica para o radar (10 = perfeito, 0 = péssimo)
                max_faltas = 200  # Assumindo 200 dias letivos
                faltas_norm = max(0, 10 - ((aluno_faltas / max_faltas) * 100)) 
                escola_faltas_norm = max(0, 10 - ((df_abt['total_faltas_ano'].mean() / max_faltas) * 100))
                
                # Sono normalizado (8h = 10)
                sono_norm = min(10, (aluno_sono / 8) * 10)
                
                # TRI normalizada (escala típica -3 a +3, ajustada para 0 a 10)
                tri_norm = max(0, min(10, ((aluno_tri + 3) / 6) * 10))
                
                categorias = ['Rendimento (Notas)', 'Assiduidade', 'Bem-Estar (Sono)', 'Cognitivo (TRI)']
                
                fig = go.Figure()
                
                # Traço da Média da Escola
                fig.add_trace(go.Scatterpolar(
                    r=[escola_nota_norm, escola_faltas_norm, 7.5, 5], # Valores médios assumidos para sono/TRI da escola
                    theta=categorias,
                    fill='toself',
                    name='Média da Escola',
                    line_color='rgba(169, 169, 169, 0.5)',
                    fillcolor='rgba(169, 169, 169, 0.2)'
                ))
                
                # Traço do Estudante Atual
                fig.add_trace(go.Scatterpolar(
                    r=[nota_norm, faltas_norm, sono_norm, tri_norm],
                    theta=categorias,
                    fill='toself',
                    name='Estudante Consultado',
                    line_color='rgba(46, 204, 113, 0.8)' if classe_atual == 'Baixo Risco' else 'rgba(231, 76, 60, 0.8)'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=True,
                    margin=dict(l=40, r=40, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)', # Fundo transparente para combinar com o Dark Mode
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white")
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # ======== BUSCA NO HISTÓRICO DE DEFASAGENS E NOTAS (INTEGRAÇÃO COM A ABA 3) ========
            try:
                conn_busca = sqlite3.connect('sistema_preditivo.db')
                
                # TRATAMENTO DE DADOS: Remove espaços em branco que vêm do Excel do Evalbee
                sgde_limpo = str(sgde_busca).strip()
                
                # Busca flexível ignorando formatação de números inteiros vs decimais
                df_notas_indiv = pd.read_sql_query(f"SELECT * FROM tb_notas_aluno WHERE CAST(codigo_sgde AS TEXT) LIKE '%{sgde_limpo}%'", conn_busca)
                df_defasagens_aluno = pd.read_sql_query(f"SELECT * FROM tb_defasagens_aluno WHERE CAST(codigo_sgde AS TEXT) LIKE '%{sgde_limpo}%'", conn_busca)
                conn_busca.close()
                
               # --- NOVO: GRÁFICOS DE RADAR POR DISCIPLINA (HISTÓRICO COMPLETO) ---
                if not df_notas_indiv.empty:
                    st.markdown("---")
                    st.subheader("🎯 Desempenho Curricular Específico (Simulados)")
                    
                    # Pega TODAS as combinações de Bimestre e Prova que o aluno já fez
                    provas_realizadas = df_notas_indiv[['bimestre', 'prova']].drop_duplicates().sort_values(['bimestre', 'prova'])
                    
                    # Cria as colunas dinamicamente com base no número total de provas encontradas
                    cols_radar = st.columns(len(provas_realizadas))
                    
                    # Varre e desenha um radar para cada prova existente no histórico do aluno
                    for idx, row in provas_realizadas.reset_index().iterrows():
                        bim = row['bimestre']
                        prv = row['prova']
                        
                        with cols_radar[idx]:
                            st.markdown(f"**Desempenho: {prv} ({bim}º Bim)**")
                            df_p = df_notas_indiv[(df_notas_indiv['bimestre'] == bim) & (df_notas_indiv['prova'] == prv)]
                            
                            fig_rad = go.Figure()
                            fig_rad.add_trace(go.Scatterpolar(
                                r=df_p['taxa_acerto'].tolist(),  # Removida a duplicação manual
                                theta=df_p['disciplina'].tolist(), # Removida a duplicação manual
                                fill='toself', 
                                name='Desempenho', 
                                line_color='#00cec9', 
                                fillcolor='rgba(0, 206, 201, 0.4)'
                            ))
                            
                            fig_rad.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.2)'), 
                                    angularaxis=dict(type='category', gridcolor='rgba(255,255,255,0.2)') # Forçado tipo 'category'
                                ), 
                                showlegend=False, 
                                paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)', 
                                font=dict(color='white'), 
                                margin=dict(t=30, b=30, l=40, r=40), 
                                height=300
                            )
                            st.plotly_chart(fig_rad, use_container_width=True)
                # --- A LISTA DE HABILIDADES ZERADAS ---
                if not df_defasagens_aluno.empty:
                    with st.expander("🚨 Ver Habilidades com Defasagem (Erros no Simulado)", expanded=False):
                        for disc in df_defasagens_aluno['disciplina'].unique():
                            habs = df_defasagens_aluno[df_defasagens_aluno['disciplina'] == disc]['habilidade'].unique()
                            st.markdown(f"- **{disc}:** Falhou na aquisição da(s) habilidade(s) `{', '.join(habs)}`.")
            except Exception as e:
                st.warning("⚠️ Os gráficos de radar individuais ainda não estão disponíveis. Vá à Aba 3 e oficialize os relatórios do Evalbee novamente para gerar esta base de dados.")
                
            # ================= NOVO: GERADOR DE PARECER AUTOMÁTICO =================
            st.markdown("---")
            st.subheader("📝 Gerador de Parecer Pedagógico Formal")
            st.markdown("Utilize a Inteligência Artificial do sistema para redigir o relatório bimestral do estudante.")
            
            if st.button("✨ Gerar Parecer Automático para o Diário de Classe", type="secondary"):
                
                # Lógica de construção do texto
                texto_risco = "apresenta um perfil estável com baixo risco de abandono escolar" if classe_atual == 'Baixo Risco' else ("inspira cuidados devido ao risco médio de evasão" if classe_atual == 'Médio Risco' else "encontra-se em SITUAÇÃO DE ALTO RISCO de abandono escolar, necessitando de intervenção imediata")
                
                texto_sono = f"Ressalva-se que o aluno trabalha e relata privação de sono ({aluno_sono}h diárias), o que pode estar a comprometer a sua atenção em sala de aula." if (aluno_sono < 6 and aluno_atual['trabalha_num'].values[0] == 1) else ""
                
                texto_tri = f"A sua proficiência cognitiva (TRI) foi medida em {aluno_tri:.2f}, revelando " + ("excelente domínio das habilidades estruturais." if aluno_tri >= 0.5 else ("lacunas profundas, com indícios de respostas ao acaso nas avaliações." if aluno_tri < -0.5 else "um desempenho dentro da média esperada."))
                
                # Montagem final do Parecer
                parecer_final = f"""
                **PARECER DESCRITIVO CONSOLIDADO:**
                
                O(a) estudante (Código SGDE: {sgde_busca}) obteve uma média geral de {aluno_nota:.1f} e regista um total de {int(aluno_faltas)} faltas acumuladas neste período. De acordo com o modelo analítico e cruzamento de variáveis socioeducativas, o discente {texto_risco}. 
                
                {texto_tri}
                
                {texto_sono}
                
                *Sugestão de Encaminhamento:* {"Recomenda-se acompanhamento por parte da coordenação e eventual flexibilização de prazos curriculares para mitigação da fadiga." if classe_atual == 'Alto Risco' else "Incentiva-se a manutenção da rotina atual de estudos."}
                """
                
                st.info(parecer_final)
                st.toast('Parecer gerado com sucesso! Pode copiar o texto.', icon='✅')

            # ================= NOVO: SIMULADOR DE CENÁRIOS (WHAT-IF ANALYSIS) =================
            st.markdown("---")
            st.subheader("🔮 Simulador de Intervenção Pedagógica (What-If)")
            st.markdown("Arraste os controlos abaixo para simular como uma intervenção da escola alteraria o risco estatístico deste estudante.")
            
            with st.expander("Clique para abrir o Simulador de Cenários", expanded=False):
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                
                with col_sim1:
                    sim_media = st.slider("Média Geral (Simulada)", 0.0, 10.0, float(aluno_nota), 0.1)
                with col_sim2:
                    sim_faltas = st.slider("Total de Faltas (Simulado)", 0, 200, int(aluno_faltas), 1)
                with col_sim3:
                    sim_trabalha = st.selectbox(
                        "Situação Laboral (Simulada)", 
                        [1, 0], 
                        index=0 if aluno_atual['trabalha_num'].values[0] == 1 else 1,
                        format_func=lambda x: "Trabalha / Aprendiz" if x == 1 else "Não Trabalha"
                    )
                
                # O sistema cria um "aluno fantasma" com os dados alterados nos sliders
                X_simulado = pd.DataFrame({
                    'media_geral_ano': [sim_media],
                    'total_faltas_ano': [sim_faltas],
                    'distorcao_idade_serie': [aluno_distorcao], # A distorção mantém-se constante (é fixa no ano)
                    'trabalha': [sim_trabalha]
                })
                
                # A Inteligência Artificial recalcula o risco instantaneamente
                novo_risco = modelo_rf.predict_proba(X_simulado)[0][1] * 100
                diferenca = novo_risco - probabilidade
                
                st.markdown(f"**Risco Atual:** `{probabilidade:.1f}%` ➔ **Risco Pós-Intervenção:** `{novo_risco:.1f}%`")
                
                # Feedback visual do impacto
                if diferenca < -0.1:
                    st.success(f"📉 Sucesso! Esta intervenção reduziria o risco de abandono em **{abs(diferenca):.1f}%**.")
                elif diferenca > 0.1:
                    st.error(f"📈 Atenção! Este cenário agravaria o risco de abandono em **{diferenca:.1f}%**.")
                else:
                    st.info("O risco permaneceria estatisticamente inalterado com estas configurações.")
# ================= ABA 3: MAPEAMENTO PEDAGÓGICO (HISTÓRICO E EVOLUÇÃO) =================
with aba3:
    st.header("📚 Diagnóstico Curricular e Evolução (Evalbee)")
    st.markdown("Processe relatórios separados por Prova (1 e 2) e acompanhe a evolução histórica das habilidades da turma.")
    
    # 1. Conexão dedicada para o histórico de simulados (Versão 2)
    conn_simulados = sqlite3.connect('sistema_preditivo.db')
    cursor_simulados = conn_simulados.cursor()
    
    cursor_simulados.execute('''
        CREATE TABLE IF NOT EXISTS tb_historico_simulados_v2 (
            bimestre INTEGER,
            prova TEXT,
            questao TEXT,
            rotulo TEXT,
            disciplina TEXT,
            habilidade TEXT,
            dificuldade TEXT,
            taxa_acerto REAL
        )
    ''')

    cursor_simulados.execute('''CREATE TABLE IF NOT EXISTS tb_notas_aluno (codigo_sgde TEXT, bimestre INTEGER, prova TEXT, disciplina TEXT, taxa_acerto REAL)''')
    
    # --- ACRESCENTADO: Tabela de defasagens individuais ---
    cursor_simulados.execute('''CREATE TABLE IF NOT EXISTS tb_defasagens_aluno (codigo_sgde TEXT, bimestre INTEGER, prova TEXT, disciplina TEXT, habilidade TEXT, dificuldade TEXT)''')

    conn_simulados.commit()

    # 2. Definição das Matrizes Detalhadas
    matriz_p1 = pd.DataFrame({
        'Questão': [f'Q {i} Marks' for i in range(1, 31)],
        'Rótulo': [f'Q{i}' for i in range(1, 31)],
        'Disciplina': ['Língua Portuguesa']*10 + ['Matemática']*10 + ['Itinerário Formativo']*10,
        'Habilidade': ['MS.EM13LP06']*5 + ['MS.EM13LP48 / MS.EM13LP49']*5 +
                      ['MS.EM13MAT312', 'MS.EM13MAT312', 'MS.EM13MAT311', 'MS.EM13MAT101', 'MS.EM13MAT311', 'MS.EM13MAT311', 'MS.EM13MAT101', 'MS.EF07MA31', 'MS.EM13MAT101', 'MS.EM13MAT312'] +
                      ['EMIFCG04']*4 + ['EMIFCG04 / EMIFCG05']*4 + ['EMIFCG04']*2,
        'Dificuldade': ['Fácil']*5 + ['Fácil a Média']*5 +
                       ['Média', 'Média', 'Fácil', 'Fácil', 'Fácil', 'Fácil', 'Fácil', 'Fácil', 'Média', 'Média'] +
                       ['Fácil']*10
    })

    matriz_p2 = pd.DataFrame({
        'Questão': [f'Q {i} Marks' for i in range(1, 28)],
        'Rótulo': [f'Q{i}' for i in range(1, 28)],
        'Disciplina': ['Física']*3 + ['Química']*3 + ['Filosofia']*3 + ['Biologia']*3 + ['Arte']*3 + ['Inglês']*3 + ['Sociologia']*3 + ['Ed. Física']*3 + ['Geografia']*3,
        'Habilidade': ['MS.EM13CNT105']*3 + ['MS.EM13CNT104']*3 + ['MS.EM13CHS101']*3 + ['MS.EM13CNT301']*3 +
                      ['MS.EM13LGG601']*3 + ['MS.EM13LGG402']*3 + ['MS.EM13CHS502 / MS.EM13CHS504']*3 +
                      ['MS.EM13LGG501 / MS.EM13LGG502']*3 + ['MS.EM13CHS302 / MS.EM13CHS101']*3,
        'Dificuldade': ['Média']*9 + ['Fácil a Média']*3 + ['Média']*3 + ['Fácil']*3 + ['Média']*3 + ['Fácil']*6
    })

    # --- SESSÃO A: PROCESSAMENTO E INSERÇÃO DE DADOS ---
    st.markdown("### 📥 1. Processar Novo Simulado")
    
    col_bim, col_prova, col_up = st.columns([1, 1, 2])
    with col_bim:
        bimestre_selecionado = st.selectbox("Qual o Bimestre?", [1, 2, 3, 4], format_func=lambda x: f"{x}º Bimestre")
    with col_prova:
        prova_selecionada = st.selectbox("Qual a Prova?", ["Prova 1", "Prova 2"])
    with col_up:
        arquivo_evalbee = st.file_uploader("Upload do relatório bruto do Evalbee", type=['csv', 'xlsx'])
    
    if arquivo_evalbee:
        try:
            if arquivo_evalbee.name.endswith('.csv'):
                df_eval = pd.read_csv(arquivo_evalbee, encoding='utf-8', on_bad_lines='skip')
            else:
                df_eval = pd.read_excel(arquivo_evalbee)
                
            colunas_questoes = [col for col in df_eval.columns if 'Marks' in col and 'Q' in col]
            
            if not colunas_questoes:
                st.error("Erro: O arquivo não possui o padrão de colunas do Evalbee (ex: 'Q 1 Marks').")
            else:
                df_respostas = df_eval[colunas_questoes].apply(pd.to_numeric, errors='coerce')
                taxa_acerto = (df_respostas > 0).mean() * 100
                df_taxas = pd.DataFrame({'Questão': taxa_acerto.index, 'taxa_acerto': taxa_acerto.values})
                
                # Escolhe a matriz correta baseada na seleção
                df_matriz_alvo = matriz_p1 if prova_selecionada == "Prova 1" else matriz_p2
                df_final = pd.merge(df_matriz_alvo, df_taxas, on='Questão', how='inner')
                
                if df_final.empty:
                    st.error("As questões do arquivo não bateram com a matriz selecionada. Verifique se escolheu a Prova correta.")
                else:
                    st.success(f"✅ Arquivo lido com sucesso! Pressione o botão abaixo para oficializar os dados como {prova_selecionada} do {bimestre_selecionado}º Bimestre.")
                    
                    if st.button(f"💾 Salvar Resultados: {prova_selecionada} | {bimestre_selecionado}º Bim", type="primary"):
                        cursor_simulados.execute("DELETE FROM tb_historico_simulados_v2 WHERE bimestre = ? AND prova = ?", (bimestre_selecionado, prova_selecionada))
                        df_save = df_final.rename(columns={'Questão': 'questao', 'Rótulo': 'rotulo', 'Disciplina': 'disciplina', 'Habilidade': 'habilidade', 'Dificuldade': 'dificuldade'})
                        df_save['bimestre'] = bimestre_selecionado
                        df_save['prova'] = prova_selecionada
                        df_save.to_sql('tb_historico_simulados_v2', conn_simulados, if_exists='append', index=False)
                        
                        # --- ACRESCENTADO: Lógica para processar alunos individuais ---
                        col_aluno = next((col for col in ['Código do ALUNO', 'Código ALUNO', 'Student ID'] if col in df_eval.columns), None)
                        if col_aluno:
                            df_alunos_notas = df_eval[[col_aluno] + colunas_questoes].copy()
                            df_alunos_notas.rename(columns={col_aluno: 'codigo_sgde'}, inplace=True)
                            df_melt = df_alunos_notas.melt(id_vars=['codigo_sgde'], var_name='Questão', value_name='nota')
                            df_melt['nota'] = pd.to_numeric(df_melt['nota'], errors='coerce').fillna(0)
                            
                            df_melt_matriz = pd.merge(df_melt, df_matriz_alvo, on='Questão', how='inner')
                            
                            # 1. Salva Notas para o Radar Individual
                            df_notas_aluno = df_melt_matriz.groupby(['codigo_sgde', 'Disciplina'])['nota'].mean().reset_index()
                            df_notas_aluno.rename(columns={'Disciplina': 'disciplina', 'nota': 'taxa_acerto'}, inplace=True)
                            df_notas_aluno['taxa_acerto'] = df_notas_aluno['taxa_acerto'] * 100
                            df_notas_aluno['bimestre'] = bimestre_selecionado
                            df_notas_aluno['prova'] = prova_selecionada
                            cursor_simulados.execute("DELETE FROM tb_notas_aluno WHERE bimestre = ? AND prova = ?", (bimestre_selecionado, prova_selecionada))
                            df_notas_aluno.to_sql('tb_notas_aluno', conn_simulados, if_exists='append', index=False)
                            
                            # 2. Salva Defasagens (Erros) Individuais
                            df_erros = df_melt_matriz[df_melt_matriz['nota'] <= 0]
                            df_defasagens = df_erros[['codigo_sgde', 'Disciplina', 'Habilidade', 'Dificuldade']].copy()
                            df_defasagens.rename(columns={'Disciplina': 'disciplina', 'Habilidade': 'habilidade', 'Dificuldade': 'dificuldade'}, inplace=True)
                            df_defasagens['bimestre'] = bimestre_selecionado
                            df_defasagens['prova'] = prova_selecionada
                            cursor_simulados.execute("DELETE FROM tb_defasagens_aluno WHERE bimestre = ? AND prova = ?", (bimestre_selecionado, prova_selecionada))
                            df_defasagens.to_sql('tb_defasagens_aluno', conn_simulados, if_exists='append', index=False)
                        # --------------------------------------------------------------

                        st.success("Dados registrados no Banco de Dados com sucesso! A página será atualizada.")
                        st.rerun() 
                        
        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            
    st.markdown("---")

    # --- SESSÃO B: VISUALIZAÇÃO DOS DADOS HISTÓRICOS ---
    df_hist = pd.read_sql_query("SELECT * FROM tb_historico_simulados_v2 ORDER BY bimestre, prova, rotulo", conn_simulados)
    conn_simulados.close()
    
    if not df_hist.empty:
        st.markdown("### 📊 2. Painel de Acompanhamento Curricular")
        
        bimestres_disponiveis = df_hist['bimestre'].unique()
        
        if len(bimestres_disponiveis) > 1:
            st.markdown("#### 📈 Evolução Longitudinal por Disciplina")
            df_evolucao = df_hist.groupby(['bimestre', 'disciplina'])['taxa_acerto'].mean().reset_index()
            
            fig_evo = px.line(
                df_evolucao, x='bimestre', y='taxa_acerto', color='disciplina', markers=True,
                title="Média de Acertos (%) da Turma ao longo dos Bimestres",
                labels={'bimestre': 'Bimestre', 'taxa_acerto': 'Taxa Média de Acerto (%)', 'disciplina': 'Área de Conhecimento'}
            )
            fig_evo.update_xaxes(tickvals=bimestres_disponiveis, ticktext=[f"{b}º Bim" for b in bimestres_disponiveis])
            fig_evo.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_evo, use_container_width=True)
            st.markdown("---")

        st.markdown("#### 🗺️ Mapa de Calor de Aprendizagem")
        col_fbim, col_fprova = st.columns(2)
        with col_fbim:
            bimestre_view = st.radio("Selecione o Bimestre para o Mapa de Calor:", bimestres_disponiveis, horizontal=True)
        with col_fprova:
            provas_disp = df_hist[df_hist['bimestre'] == bimestre_view]['prova'].unique()
            if len(provas_disp) > 0:
                prova_view = st.radio("Selecione a Prova para o Mapa de Calor:", provas_disp, horizontal=True)
            else:
                prova_view = None
        
        if prova_view:
            df_view = df_hist[(df_hist['bimestre'] == bimestre_view) & (df_hist['prova'] == prova_view)].copy()
            
            # Garantir a ordenação correta das questões de 1 a 30 no eixo X
            df_view['ordem_q'] = df_view['rotulo'].str.replace('Q', '').astype(int)
            df_view = df_view.sort_values('ordem_q')
            
            heatmap_data = df_view.pivot(index='disciplina', columns='rotulo', values='taxa_acerto')
            # Reordenar as colunas do heatmap explicitamente
            cols_ordenadas = sorted(heatmap_data.columns, key=lambda x: int(x.replace('Q', '')))
            heatmap_data = heatmap_data[cols_ordenadas]
            
            texto_customizado = [
                [f"{val:.0f}%" if pd.notna(val) else "" for val in row] 
                for row in heatmap_data.values
            ]
            
            fig_heat = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=[[0, '#e74c3c'], [0.6, '#f1c40f'], [1, '#2ecc71']],
                zmin=0, zmax=100,
                text=texto_customizado,       # Injetamos a nossa matriz de texto limpa
                texttemplate="%{text}",       # Exibimos o texto exatamente como foi processado
                textfont={"color": "white", "size": 12},
                hoverongaps=False,
                hovertemplate="Questão: %{x}<br>Disciplina: %{y}<br>Acerto: %{z:.1f}%<extra></extra>"
            ))
            
            fig_heat.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'), 
                yaxis=dict(autorange="reversed"), 
                height=400
            )
            
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig_heat, use_container_width=True)
            
            defasagens = df_view[df_view['taxa_acerto'] < 60]
            if not defasagens.empty:
                st.error(f"🚨 **Focos Críticos ({prova_view} - {bimestre_view}º Bimestre):** {len(defasagens)} habilidade(s) abaixo da meta de 60%.")
                st.dataframe(defasagens[['rotulo', 'disciplina', 'habilidade', 'dificuldade', 'taxa_acerto']].sort_values(by='taxa_acerto'), use_container_width=True, hide_index=True)
            else:
                st.success("🎉 Todas as habilidades superaram a meta de 60%!")
        else:
            st.info("Não há dados cadastrados para a prova e bimestre selecionados.")

    else:
        st.info("Nenhum dado de simulado cadastrado. Selecione a Prova e o Bimestre acima, faça o upload e salve o primeiro arquivo!")