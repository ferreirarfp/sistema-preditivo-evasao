# Sistema Web Preditivo de Evasão Escolar e Diagnóstico de Habilidades

Repositório destinado ao desenvolvimento do Projeto Integrador (PI) do eixo de Computação da Universidade Virtual do Estado de São Paulo (UNIVESP) - Polo Presidente Epitácio (2026).

## 🎯 Sobre o Projeto
Este projeto visa desenvolver um software para processar dados acadêmicos históricos (SGDE) e resultados de simulados nivelados. O objetivo é mapear defasagens cognitivas utilizando a Teoria de Resposta ao Item (TRI) e prever o risco de evasão escolar através de algoritmos de Machine Learning, subsidiando professores em suas intervenções pedagógicas. O ambiente de pesquisa e validação é a Escola Estadual Manoel da Costa Lima (Bataguassu-MS).

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Data Wrangling:** Pandas, pdfplumber, Expressões Regulares (Regex)
* **Machine Learning:** Scikit-Learn (Árvores de Decisão), XGBoost
* **Web Framework:** Streamlit
* **Controle de Versão:** Git e GitHub

## 🔒 Privacidade e LGPD
O script de extração de dados (`extrairDados.py`) foi estruturado com uma política rigorosa de privacidade. Campos sensíveis (Nomes, CPFs, RGs) são descartados no processamento, utilizando apenas o "Código SGDE" como chave primária. **Nenhum dado real de aluno é versionado neste repositório.**

## 👥 Integrantes do Grupo
* Renan Felipe de Paula Ferreira
* Vanderli Jose Ferreira de Pinho
* Daniela Barros Lemes
* Luis Gustavo dos Santos Henares Rodrigues
