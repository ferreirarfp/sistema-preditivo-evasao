import pdfplumber  # A biblioteca salvadora para ler o texto dos PDFs
import re  # Regex: a arte de encontrar agulha em palheiro de texto
import pandas as pd  # Nossa "bíblia" para manipular tabelas e DataFrames


def extrair_dados_fichas(caminho_pdf):
    fichas_extraidas = []  # Lista onde vamos guardar o dicionário de cada aluno

    try:
        # Abrimos o arquivo PDF para "escaneamento" de texto
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                texto_pagina = pagina.extract_text()
                if not texto_pagina:
                    continue

                # Aqui é o pulo do gato: cada ficha começa com "NOME DO ESTUDANTE:"
                # Usamos o split para separar a página em blocos individuais por aluno
                blocos_texto = texto_pagina.split("NOME DO ESTUDANTE:")[1:]

                for bloco in blocos_texto:
                    # Reconstruímos a string para o Regex não se perder
                    texto_ficha = "NOME DO ESTUDANTE:" + bloco
                    print("--- RAIO X DO PDF ---")
                    print(texto_ficha) # Isso vai te mostrar a bagunça real do texto
                    print("---------------------")
                    dados_ficha = {}

                    # Dicionário de padrões Regex: limpamos o que é sensível (LGPD na veia!)
                    # Atualizado com as variáveis socioeconômicas e acadêmicas essenciais
                    padroes = {
                        "CODIGO_SGDE": r"CÓDIGO SGDE:\s*(\d+)",
                        "SITUACAO_MATRICULA": r"SITUAÇÃO MATRÍ(?:CULA|ULA):\s*(.*?)(?=NOME DA FILIAÇÃO|NOME DO PAI|DATA NASCIMENTO|$)",
                        "DATA_NASCIMENTO": r"DATA NASCIMENTO:\s*(.*?)(?=IDADE:|$)",
                        "IDADE": r"IDADE:\s*(\d+)",
                        "TRABALHADOR": r"TRABALHADOR\(A\):\s*(.*?)(?=SEXO:|$)",
                        "SEXO": r"SEXO:\s*(.*?)(?=PROFISSÃO:|DOADOR|COR/RAÇA:|\n|$)",
                        "COR_RACA": r"COR/RAÇA:\s*(.*?)(?=NECESSIDADES EDUCACIONAIS|$|\n)",
                        "NECESSIDADES_ESPECIFICAS": r"NECESSIDADES EDUCACIONAIS ESPECÍFICAS:\s*(.*?)(?=TIPO CERTIDÃO CIVIL:|$)",
                        "ENDERECO_ANONIMO": r"ENDEREÇO:\s*(.*?)(?=COMPLEMENTO:|$)",
                        "BAIRRO": r"BAIRRO:\s*(.*?)(?=MUNICÍPIO:|$)",
                        "MUNICIPIO": r"MUNICÍPIO:\s*(.*?)(?=TELEFONE:|$)"
                    }

                    for chave, padrao in padroes.items():
                        # O re.DOTALL é fundamental aqui porque o texto do SGDE adora pular linha
                        resultado = re.search(padrao, texto_ficha, re.IGNORECASE | re.DOTALL)
                        if resultado:
                            # Se achou, limpa os espaços em branco extras (strip)
                            valor = resultado.group(1).strip() if resultado.group(1) else None
                            dados_ficha[chave] = valor
                        else:
                            dados_ficha[chave] = None

                    fichas_extraidas.append(dados_ficha)

    except Exception as e:
        # Se o PDF estiver corrompido ou o caminho estiver errado, ele avisa
        print(f"Erro ao abrir o PDF: {e}")

    return fichas_extraidas



# --- Execução do Workflow de Data Science ---
print("🚀 Iniciando o script...")

arquivo_pdf = r"D:\1_Arquivos\Trabalho\Professor\Prof_2026\Documentos noturno\baixados.pdf"

print("Lendo o PDF e extraindo dados...")
dados_extraidos = extrair_dados_fichas(arquivo_pdf)
df_fichas = pd.DataFrame(dados_extraidos)

if not df_fichas.empty:
    print("✅ Sucesso! Primeiras linhas do DataFrame:")
    print(df_fichas.head())
else:
    print("⚠️ Vazio... O Regex não encontrou os padrões.")

caminho_saida = "dados_fichas_higienizados.xlsx"
df_fichas.to_excel(caminho_saida, index=False)

print(f"🎉 Relatório gerado com sucesso na pasta do projeto: {caminho_saida}")