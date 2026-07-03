# 🔮 Criador de Modelos Preditivos

Aplicação web (Streamlit) que cria modelos preditivos a partir de uma planilha, com
**tratamento automático de dados**, **expurgo de outliers**, **seleção de variáveis com
explicação**, **guia de modelos com prós e contras**, **simulador** e **download do modelo
treinado**.

## Formato da planilha

- **1ª coluna:** datas (ex.: `31/12/2024`)
- **Demais colunas:** parâmetros (números ou categorias)
- Uma das colunas é a **variável alvo** (o que você quer prever) — você escolhe qual no app
- Formatos aceitos: `.csv`, `.xlsx`, `.xls`

## Instalação

Requer Python 3.10 ou superior.

```bash
pip install -r Prophet/requirements.txt
```

## Como executar

A partir da raiz do repositório:

```bash
streamlit run Prophet/app.py
```

O navegador abrirá automaticamente (padrão: http://localhost:8501).

> Não tem uma planilha à mão? O app traz dois exemplos embutidos (botões na aba 1).
> Para regerar os arquivos de exemplo: `python Prophet/sample_data.py`

## Guia de uso (aba a aba)

### 1. Upload e Dados
Envie a planilha (ou carregue um exemplo). O app assume a **primeira coluna como data**,
mostra uma prévia dos dados e pede que você escolha a **variável alvo**. O tipo de problema
(**regressão** = prever número; **classificação** = prever categoria) é detectado
automaticamente, com opção de ajuste manual.

### 2. Tratamento (automático)
O app trata os dados sozinho e **reporta tudo o que fez**:
- Interpreta as datas (formato brasileiro `dia/mês/ano` primeiro) e remove linhas sem data válida
- Remove linhas com a variável alvo vazia (o alvo nunca é "inventado")
- Cria variáveis derivadas da data: `mes`, `dia_semana`, `dia_do_mes`, `trimestre`, `indice_tendencia`
- **Expurga outliers automaticamente** pelo método IQR: valores fora de
  `[Q1 − 1,5×IQR, Q3 + 1,5×IQR]` são trazidos para o limite (clipping), sem excluir linhas.
  Outliers na variável alvo são apenas reportados (não alterados, para não distorcer a previsão)
- Detecta valores ausentes — o preenchimento (mediana/mais frequente) acontece dentro do
  pipeline de treino, usando só os dados de treino, para **evitar vazamento de dados**
- Remove colunas inúteis (constantes, com mais de 60% de vazios ou com excesso de categorias),
  sempre informando o motivo

### 3. Seleção de Variáveis
Cada variável recebe um **score de 0 a 100** combinando três análises complementares:

| Análise | O que mede | Peso |
|---|---|---|
| Associação (correlação/ANOVA) | Relação estatística direta com o alvo | 35% |
| Informação mútua | Relações **não lineares** com o alvo | 30% |
| Importância em Random Forest | Utilidade prática em um modelo de árvores | 35% |

A coluna **Motivo** explica cada pontuação (ex.: "forte relação com o alvo", "redundante com
X — considere usar apenas uma", "variância quase nula"). As variáveis recomendadas já vêm
pré-selecionadas, mas **você decide** quais entram no modelo.

### 4. Modelo e Treinamento
O app **sugere um modelo** com base no tamanho da sua base e mostra, para cada modelo,
**quando usar** e uma **tabela de prós e contras**. Ajuste os hiperparâmetros se quiser,
defina o percentual de teste (com opção de divisão temporal, ideal para prever o futuro) e
treine. São exibidas as métricas (R²/MAE/RMSE ou acurácia/precisão/revocação/F1 com matriz
de confusão) e gráficos (previsto vs. real, resíduos, importância das variáveis).

Depois do treino, dois botões de download:
- **`modelo_preditivo.joblib`** — pipeline completo (tratamento + modelo), pronto para usar
- **`exemplo_uso.py`** — script Python de exemplo que carrega o modelo e faz uma previsão

### 5. Simulador
Digite os valores de cada variável selecionada (os campos já vêm preenchidos com valores
típicos da sua base) e clique em **Prever**. Em classificação, além da classe prevista, é
exibida a probabilidade de cada classe.

### 6. Guia de Modelos
Referência completa com **quando usar** e **prós/contras** de todos os modelos disponíveis:

- **Regressão:** Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, KNN
- **Classificação:** Logística, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes

## Usando o modelo baixado

```python
import joblib
import pandas as pd

artifact = joblib.load("modelo_preditivo.joblib")
pipeline = artifact["pipeline"]

novos_dados = pd.DataFrame([{ ... }])  # colunas em artifact["features"]
print(pipeline.predict(novos_dados))
```

O arquivo `exemplo_uso.py` baixado do app já vem com um exemplo real da sua base, pronto
para rodar. Recomenda-se usar a mesma versão do scikit-learn registrada em
`artifact["sklearn_version"]`.

## Estrutura do código

| Arquivo | Responsabilidade |
|---|---|
| `app.py` | Interface Streamlit (abas, estado, orquestração) |
| `data_processing.py` | Carga da planilha, datas, outliers (IQR), relatório de tratamento |
| `feature_selection.py` | Score 0–100 das variáveis com explicações |
| `models_catalog.py` | Catálogo de modelos, prós/contras e sugestão automática |
| `training.py` | Pipeline sklearn, treino, métricas e gráficos |
| `simulator.py` | Aba de simulação |
| `export.py` | Download do `.joblib` e do script de exemplo |
| `sample_data.py` | Gerador das planilhas de exemplo |
