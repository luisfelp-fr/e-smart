# e-smart — Análise causal automática de parâmetros

Programa de análise de dados que **valida relações de causa e efeito** entre
parâmetros de processo e uma variável-alvo, e gera um **relatório HTML
automático** apontando os prováveis "culpados" pelo comportamento do alvo.

A análise considera **efeitos com defasagem (lag), médias móveis, percentis e
relações não-lineares** — nenhum teste isolado decide: sete linhas de evidência
independentes são combinadas num score de culpabilidade (0–100) com controle de
falsos positivos.

## Entrada

Uma tabela `.csv` ou `.xlsx` em que:

- a **primeira coluna** contém as **datas** (formatos `dd/mm/aaaa`,
  `aaaa-mm-dd` etc. são detectados automaticamente, assim como decimal com
  vírgula e separador `;`);
- as **demais colunas** são os **parâmetros numéricos**;
- o usuário informa **qual coluna é o alvo** (`--alvo`).

Exemplo:

| data       | rendimento | temp_forno | pressao | vazao_agua | ... |
|------------|-----------:|-----------:|--------:|-----------:|-----|
| 01/06/2024 | 62,4       | 84,2       | 5,1     | 118,3      | ... |
| 02/06/2024 | 58,9       | 86,0       | 4,8     | 121,7      | ... |

## Uso

```bash
pip install -r requirements.txt

python -m causal_analysis dados.csv --alvo rendimento
```

Opções principais:

```text
--alvo NOME           coluna-alvo (obrigatório)
--coluna-data NOME    coluna de datas (padrão: primeira coluna)
--max-lag N           defasagem máxima testada em períodos (padrão: 14)
--janelas W1 W2 ...   janelas das médias móveis (padrão: 3 7 14)
--alfa X              significância com correção FDR (padrão: 0.05)
--saida ARQ.html      arquivo do relatório (padrão: relatorio_causal.html)
--planilha NOME|N     planilha do Excel (padrão: primeira)
--sep ';'             separador do CSV (padrão: autodetecta)
--top-detalhe N       nº de parâmetros com seção detalhada (padrão: 5)
```

Também pode ser usado como biblioteca:

```python
from causal_analysis import run_analysis, render_report

resultado = run_analysis("dados.csv", target="rendimento", max_lag=14)
print(resultado.scores)          # ranking com scores e vereditos
render_report(resultado, "relatorio.html")
```

## O que o relatório contém

1. **Resumo executivo** — lista dos culpados prováveis/possíveis, com direção
   do efeito (positiva/negativa/não-monotônica), melhor transformação temporal
   (ex.: "lag 2", "média móvel 7") e confiança estatística.
2. **Tabela consolidada de evidências** — score, veredito e testes
   significativos de todos os parâmetros, com detalhamento expansível de todas
   as estatísticas e p-valores.
3. **Mapa de correlação** — inclusive entre parâmetros, para expor
   multicolinearidade/confusão.
4. **Análise detalhada por suspeito** — perfil de correlação por defasagem e
   por janela de média móvel, dispersão com tendência LOWESS (revela formas
   não-lineares), boxplot do alvo por quartil do parâmetro e evolução temporal
   comparada (z-score).
5. **Diagnóstico dos dados** — linhas usadas, frequência, valores ausentes e
   interpolações.
6. **Metodologia e limitações** — explicação de cada teste e os cuidados de
   interpretação (associação ≠ prova de causalidade).

## Métodos estatísticos aplicados

| Linha de evidência | Método | O que captura |
|---|---|---|
| Associação linear | Correlação de Pearson | relações proporcionais diretas |
| Associação monotônica | Spearman e Kendall | relações crescentes/decrescentes não-lineares |
| Dependência geral | Correlação de distância (Székely) + Informação mútua | formas em U, limiares, qualquer dependência |
| Efeito defasado/acumulado | Varredura de lags 0..N e médias móveis | atraso de resposta e efeitos acumulados |
| Precedência temporal | Causalidade de Granger (com teste ADF e diferenciação) | o passado do parâmetro prevê o alvo? |
| Contraste de percentis | Mann-Whitney (P75 vs P25), delta de Cliff, Kruskal-Wallis por quartis | efeito prático de "parâmetro alto vs baixo" |
| Importância preditiva | Random Forest + importância por permutação (validação temporal) | não-linearidades e interações conjuntas |

Os p-valores de cada família passam pela correção de **Benjamini-Hochberg
(FDR)**; a confiança reportada conta quantos testes sobrevivem.

## Exemplo com dados sintéticos

O repositório inclui um gerador de dados com estrutura causal conhecida
(efeito não-linear com lag, efeito via média móvel, efeito de limiar por
percentil, variável de confusão e ruídos):

```bash
python examples/generate_example_data.py
python -m causal_analysis examples/dados_exemplo.csv --alvo rendimento --saida relatorio_exemplo.html
```

O analisador recupera corretamente os quatro mecanismos plantados e não aponta
as variáveis de ruído.

## Testes

```bash
python tests/test_pipeline.py
```

## Limitações

Evidência estatística (mesmo com precedência temporal de Granger) **não é
prova definitiva de causalidade**: fatores não medidos podem confundir a
análise, e parâmetros correlacionados entre si dividem a "culpa". Use o
relatório para priorizar hipóteses e valide com experimentos controlados ou
intervenções conhecidas no processo.
