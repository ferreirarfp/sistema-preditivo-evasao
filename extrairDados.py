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
                    dados_ficha = {}

                    # Dicionário de padrões Regex: limpamos o que é sensível (LGPD na veia!)
                    # Substituímos campos de identificação pessoal pelo CÓDIGO SGDE
                    padroes = {
                        "CODIGO_SGDE": r"CÓDIGO SGDE:\s*(\d+)",  # Captura apenas os números do código
                        "NATURALIDADE": r"NATURALIDADE:\s*(.*?)(?=TRABALHADOR\(A\):|IDADE:|$)",
                        "UF_NATURALIDADE": r"UF:\s*(\S+)?(?=\s*DATA EXPEDIÇÃO:|$)",
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

# Caminho do arquivo que baixamos do sistema da escola
print("🚀 Iniciando o processamento. Lendo o PDF...")
arquivo_pdf = r"D:\1_Arquivos\Trabalho\Professor\Prof_2026\Documentos noturno\baixados.pdf"

# Chamamos a função e transformamos a lista de dicionários em um DataFrame elegante
dados_extraidos = extrair_dados_fichas(arquivo_pdf)
df_fichas = pd.DataFrame(dados_extraidos)

# Verificação rápida no console para ver se os dados subiram certinho
if not df_fichas.empty:
    print("Sucesso! Primeiras linhas do DataFrame:")
    print(df_fichas.head())
else:
    print("Vazio... hora de revisar os padrões de Regex.")

# Exportação final para o Excel que vamos usar para alimentar o banco de dados
#caminho_saida = r"C:\Users\Renan\Desktop\dados_fichas_higienizados.xlsx"
caminho_saida = "dados_fichas_higienizados_simples.xlsx"
df_fichas.to_excel(caminho_saida, index=False)

print(f"Relatório gerado com sucesso em: {caminho_saida}")