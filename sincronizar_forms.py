import sqlite3
import pandas as pd

# 1. Configurações
ARQUIVO_EXCEL = 'respostas_forms_rotina.xlsx'
BANCO_DADOS = 'sistema_preditivo.db'

conn = sqlite3.connect(BANCO_DADOS)
cursor = conn.cursor()

# 2. Preparar o banco para os novos "Insights Preciosos"
colunas_novas = [
    ('trabalha', 'TEXT'),
    ('carga_horaria', 'TEXT'),
    ('esforco_fisico', 'TEXT'),
    ('rotina_deslocamento', 'TEXT'),
    ('horas_sono', 'TEXT'),
    ('alimentacao', 'TEXT'),
    ('problemas_transporte', 'TEXT'),
    ('objetivo_pos_medio', 'TEXT'),
    ('horario_saida_trabalho', 'TEXT')
]

for nome_col, tipo in colunas_novas:
    try:
        cursor.execute(f"ALTER TABLE tb_estudantes ADD COLUMN {nome_col} {tipo}")
    except sqlite3.OperationalError:
        pass

# 3. Ler a planilha do Forms (agora usando read_excel!)
df = pd.read_excel(ARQUIVO_EXCEL)

# Mapeamento das colunas
mapeamento = {
    'Código SGDE': 'codigo_sgde',
    '2)  Qual é a sua situação de trabalho atual? ': 'trabalha',
    '3)  Qual é a sua carga horária de trabalho diária? ': 'carga_horaria',
    '4)  Como você avalia o esforço físico exigido no seu trabalho? ': 'esforco_fisico',
    '5)  Como é a sua rotina de deslocamento para a escola? ': 'rotina_deslocamento',
    '6)  Em média, quantas horas de sono você consegue ter por noite durante a semana?  ': 'horas_sono',
    '7)  Sobre a sua alimentação antes de chegar à escola: ': 'alimentacao',
    '8)  O seu meio de transporte para a escola é sujeito a atrasos frequentes ou problemas (ex: ônibus rural que quebra, carona incerta, ônibus de linha atrasado)? ': 'problemas_transporte',
    '9)  Qual o seu principal objetivo ao concluir o Ensino Médio? ': 'objetivo_pos_medio',
    '10) Qual o horário que você sai do seu trabalho?': 'horario_saida_trabalho'
}

df = df.rename(columns=mapeamento)

print("Iniciando a sincronização das respostas reais...")

# 4. Loop de Atualização
for index, row in df.iterrows():
    sgde = row['codigo_sgde']
    
    sql_update = """
    UPDATE tb_estudantes 
    SET trabalha = ?, carga_horaria = ?, esforco_fisico = ?, 
        rotina_deslocamento = ?, horas_sono = ?, alimentacao = ?, 
        problemas_transporte = ?, objetivo_pos_medio = ?, 
        horario_saida_trabalho = ?
    WHERE codigo_sgde = ?
    """
    
    valores = (
        row['trabalha'], row['carga_horaria'], row['esforco_fisico'],
        row['rotina_deslocamento'], row['horas_sono'], row['alimentacao'],
        row['problemas_transporte'], row['objetivo_pos_medio'],
        row['horario_saida_trabalho'], sgde
    )
    
    cursor.execute(sql_update, valores)

conn.commit()
print(f"✅ Sucesso! {len(df)} perfis de alunos foram atualizados com dados reais do Forms.")
conn.close()