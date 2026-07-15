# e-smart — Analisador de indicadores de processo

Aplicativo **Streamlit** para engenheiros e técnicos de processo analisarem
indicadores industriais **sem precisar dominar estatística**: a interface
explica cada método com tooltips, e todos os resultados saem também em
linguagem simples.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Entrada de dados

- Planilha **CSV ou Excel** (`.xlsx/.xls/.xlsm/.ods`), carregada na barra lateral.
- **Decimal com vírgula** (`12,5`) e datas brasileiras (`dd/mm/aaaa`) são
  detectados automaticamente, assim como o separador (`;`, `,` ou tab).
- A coluna de data/hora é **opcional**: sem ela, o app usa a ordem das linhas
  como sequência temporal (1, 2, 3, ...).
- O Excel pode ter **múltiplas abas com granularidades diferentes** (ex.: uma
  em segundos, outra de 4 em 4 horas): a aba que contém o alvo define a grade
  de tempo e as demais são alinhadas a ela automaticamente.

## Módulo 1 — Análise de capabilidade

Responde: *"meu indicador é capaz de atender aos limites de atuação do
processo?"*

1. **Tratamento opcional** de outliers (IQR, Z-score ou Z-score robusto/MAD) e
   de dados faltantes — aplicável a qualquer coluna, inclusive o alvo, com os
   devidos avisos sobre o efeito nos índices.
2. **Limites de especificação** bilaterais ou únicos (só inferior / só superior).
3. **Carta de controle I-AM** (Individuais e Amplitude Móvel) com detecção de
   causas especiais (regras de Nelson selecionadas) e opção de excluir pontos
   com justificativa — com o aviso de que excepcionalidades deturpam a análise.
4. **Teste de normalidade** (Anderson-Darling, com Shapiro-Wilk de apoio):
   - **Caso 1 — dados normais**: índices clássicos Cp/Cpk/Pp/Ppk + PPM.
   - **Caso 2 — normais após transformação** (log, raiz, Box-Cox, Yeo-Johnson,
     Johnson): índices calculados na escala transformada e valores exibidos
     convertidos de volta à escala original.
   - **Caso 3 — não-normais mesmo transformando**: análise por box-plot e
     percentis empíricos, PPM contado nos dados e **sugestão de limites de
     atuação pelos quartis** (Q3 como limite inferior para "quanto maior
     melhor"; faixa Q2–Q3 para indicadores bilaterais; Q1 como limite
     superior para "quanto menor melhor"), com os percentis de cauda
     informados como referência de cobertura.

## Módulo 2 — O que impacta o alvo

Responde: *"quais indicadores mais influenciam minha variável alvo?"* —
pensado para dados industriais, que raramente são lineares ou normais.

- Séries mais finas que o alvo geram **famílias de métricas por janela**
  (média, mediana, mínimo, máximo, P10, P90, desvio e % do tempo em faixa
  alta/baixa) para capturar picos e permanências que a média esconde.
- **Efeitos com defasagem (lag)** e médias móveis são varridos para cada
  métrica; a estrutura temporal é validada por **Ljung-Box** e a precedência
  por **causalidade de Granger** (com teste ADF).
- Sete linhas de evidência independentes (Pearson/Spearman/Kendall, correlação
  de distância, informação mútua, lags, Granger, contraste de percentis e
  Random Forest com validação temporal) são combinadas num **score 0–100**
  com controle de falsos positivos (FDR).
- Resultado em dois formatos: **ranking técnico** completo e **leitura
  gerencial** em frases simples ("quando X sobe, o alvo tende a cair; o efeito
  aparece ~3 períodos depois").
- **Investigação em cadeia**: qualquer variável do ranking pode virar o novo
  alvo, para seguir a trilha até a causa raiz.

## Relatório

Toda análise tem o botão **"Adicionar ao relatório"**. Na página Relatório é
possível ver o **preview em HTML** (gráficos interativos) e **baixar em
PDF** (uma seção por análise, com textos, tabelas e imagens dos gráficos).

> Para converter os gráficos em imagem dentro do PDF, o `kaleido` (≥ v1)
> exige um Chrome/Chromium instalado — se não houver, rode
> `plotly_get_chrome` uma vez. Sem navegador, o PDF sai apenas com textos e
> tabelas. No **Streamlit Community Cloud** isso já está resolvido pelo
> `packages.txt` (instala o `chromium` via apt).

> **Deploy no Streamlit:** a branch de referência do app é a **`main`** —
> aponte o Streamlit Cloud para ela; toda melhoria é entregue lá.

## Uso por linha de comando (motor do Módulo 2)

O motor de análise causal também funciona como CLI/biblioteca:

```bash
python -m causal_analysis dados.csv --alvo rendimento
```

```python
from causal_analysis import run_analysis, render_report

resultado = run_analysis("dados.csv", target="rendimento", max_lag=14)
print(resultado.scores)          # ranking com scores e vereditos
render_report(resultado, "relatorio.html")
```

## Estrutura do código

```
app/               interface Streamlit (páginas e componentes)
capability/        motor do Módulo 1 (carta I-AM, normalidade, índices)
causal_analysis/   motor do Módulo 2 (testes, ML, scores, relatório)
shared/            leitura de planilhas, multi-aba e exportação PDF
examples/          dados sintéticos com estrutura causal conhecida
tests/             testes unitários e de integração
```

## Exemplo com dados sintéticos

```bash
python examples/generate_example_data.py
streamlit run app/streamlit_app.py   # carregue examples/dados_exemplo.csv
```

O gerador planta mecanismos conhecidos (efeito não-linear com lag, efeito por
média móvel, limiar por percentil, confusão e ruídos) — o Módulo 2 recupera os
quatro mecanismos e não aponta as variáveis de ruído.

## Testes

```bash
python -m pytest tests/
```

## Limitações

Evidência estatística **não é prova definitiva de causalidade**: fatores não
medidos podem confundir a análise, e indicadores correlacionados entre si
dividem a "culpa". Use os resultados para priorizar hipóteses e confirme com
testes controlados no processo. Nos índices de capabilidade, processos fora de
controle estatístico (causas especiais na carta) produzem índices pouco
confiáveis — trate a estabilidade primeiro.
