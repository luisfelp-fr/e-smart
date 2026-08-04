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
- O upload é feito **uma única vez**: a planilha vira a **base compartilhada de
  todos os módulos** (lida com cache, sem re-parse a cada interação) — em cada
  módulo você só seleciona a **aba** e o **indicador**.
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
- **Indício de efeito direto vs. indireto**: para o topo do ranking, a
  correlação parcial (controlando pelos demais indicadores do topo) sinaliza
  quando a associação de um indicador "some" ao descontar outro — "indireto
  (via X)" sugere mediação e prioriza X na investigação. É uma
  versão leve da ideia de *causal discovery* (independência condicional, como
  no PC/PCMCI), escolhida no lugar do PCMCI completo por custo computacional
  e robustez com indicadores correlacionados.
- **Contrafactual e repartição** (aba do Módulo 2): o ranking diz *quem*
  influencia; esta aba diz **quanto**, na unidade do alvo. Um modelo dedicado é
  consultado duas vezes — com os valores que de fato ocorreram e com o
  indicador forçado ao **valor típico** (mediana histórica) — e a diferença é o
  efeito atribuído. Entrega quatro respostas:
  - **De onde vem a variação do alvo**: fatias por indicador que somam 100% do
    que o modelo explica, mais uma fatia explícita de **"não explicado"**
    (fatores fora da planilha + ruído). A repartição usa o **valor de Shapley**
    (amostragem de permutações), a única divisão em que as fatias fecham
    exatamente o total.
  - **"E se este indicador tivesse ficado no valor típico?"**: de quanto o alvo
    se desloca, em média, por período — e no pior período.
  - **Repartição do desvio de um período**: gráfico em cascata do período
    típico até o previsto, um degrau por indicador; a distância até o valor
    observado é o que os indicadores **não** explicam.
  - **Curva de resposta**: o alvo previsto com o indicador fixado em cada nível
    que ele já assumiu — mostra a faixa boa sem supor relação linear.

  Tudo sai com **intervalo de confiança** (bootstrap por blocos, que preserva a
  autocorrelação da série; no Shapley soma-se o erro de amostragem das
  permutações). Métricas do mesmo indicador físico (média, máximo, P90 de uma
  mesma variável) são movidas **juntas**: simular "o máximo estava normal" com
  "a média como esteve" seria um cenário impossível no processo real. E cada
  contrafactual do topo passa por uma **contraprova sem modelo** — o alvo
  observado em períodos parecidos, mas com aquele indicador normal; quando não
  existem períodos parecidos, o app diz que ali o modelo está extrapolando em
  vez de devolver um número sem lastro.
- **Diagnóstico do dia** (aba do Módulo 2): escolha um dia/período e veja os
  prováveis contribuintes daquele dia — cruzamento do ranking histórico com a
  atipicidade de cada indicador no dia (percentil do valor do dia no próprio
  histórico, avaliado na melhor transformação temporal). Indicado para o uso
  "indicadores minuto a minuto + alvo diário": a aba do alvo diário define a
  grade e os minutos viram métricas por dia automaticamente.

## Aba 📈 Analytics — análises estatísticas guiadas

Réplica do **Argus Analytics** integrada ao app: uma home com cards abre 13
telas de análise dedicadas, cada uma com seleção simples de variáveis,
tooltips de glossário e **interpretação automática em linguagem simples**:

- **Visão Geral dos Dados** (prévia + diagnóstico automático dos tipos de
  coluna: numérica, categórica, data/hora, texto), **Qualidade dos Dados**
  (ausentes, duplicidades, colunas constantes) e **Outliers** (IQR, Z-score e
  MAD, com download da base filtrada e opção de **usar a base sem outliers
  nas demais análises da aba**, reversível).
- **Estatística Descritiva**, **Distribuição** (histograma, boxplot,
  densidade, assimetria/curtose) e **Teste de Normalidade** (Shapiro-Wilk e
  Anderson-Darling).
- **Correlação** (Pearson/Spearman/Kendall, heatmap e ranking de pares),
  **Regressão** (OLS simples/múltipla) e **Comparação entre Grupos** (t,
  Mann-Whitney, ANOVA ou Kruskal-Wallis, sugerido conforme a normalidade).
- **CEP** (carta de individuais com limites ±3σ), **Capabilidade Cp/Cpk**
  (fluxo guiado com transformações Box-Cox/Johnson/log/…), **Análise
  Temporal** (reamostragem + média móvel + tendência) e **Análise com Lag**
  (correlação cruzada com validação ARIMAX/Ljung-Box).

A base é a **mesma planilha da barra lateral** (basta escolher a aba no topo)
e o botão "➕ Adicionar ao relatório" alimenta o **mesmo relatório unificado**
dos Módulos 1 e 2.

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

> **Bot keep-alive:** o workflow `.github/workflows/keep-alive.yml` visita o
> app a cada 6 h com navegador headless e o acorda se estiver hibernado
> (crons rodam na branch padrão do GitHub — recomenda-se defini-la como
> `main` em Settings). O GitHub pausa crons após ~60 dias sem atividade no
> repositório; reative na aba Actions. Em falha, o dono recebe e-mail.

## Desempenho com muitos dados

O app foi endurecido para bases grandes (ex.: dezenas de indicadores minuto a
minuto por meses):

- **Módulo 1** roda sobre todos os dados; as cartas I-AM, QQ e demais gráficos
  **rarefazem os pontos apenas para desenho** (limite ~8 mil) para não travar o
  navegador — todas as estatísticas usam a série completa e o subtítulo informa
  quando houve rarefação.
- **Módulo 2** tem um **teto de linhas** (padrão 10 mil, ajustável em *Opções da
  análise*): acima dele as linhas são **agregadas pela média em blocos
  consecutivos** — equivale a reamostrar para uma grade de tempo mais grossa,
  o que **preserva os efeitos com atraso/permanência** (os lags continuam
  traduzidos para a escala de tempo correta) e evita lentidão e estouro de
  memória. Etapas caras (Random Forest, informação mútua) também subamostram/
  escalonam o esforço conforme o tamanho do problema.

Dica: no **Streamlit Community Cloud** (plano gratuito, ~1 vCPU e ~2,7 GB de
RAM), mantenha o teto do Módulo 2 em 5–10 mil linhas para respostas em poucos
minutos. O app guarda **uma análise do Módulo 2 por vez** em memória (cache de
1 entrada, sem cópias por rerun); a matriz do modelo usa float32 e a
importância por permutação roda em processo único — mudanças pensadas para o
app não estourar a RAM do plano gratuito.

## Uso por linha de comando (motor do Módulo 2)

O motor de análise causal também funciona como CLI/biblioteca:

```bash
python -m causal_analysis dados.csv --alvo rendimento

# com a seção contrafactual (repartição da variação + cenários, com IC)
python -m causal_analysis dados.csv --alvo rendimento --contrafactual
```

```python
from causal_analysis import run_analysis, render_report

resultado = run_analysis("dados.csv", target="rendimento", max_lag=14)
print(resultado.scores)          # ranking com scores e vereditos
render_report(resultado, "relatorio.html")
```

O motor contrafactual também é usável direto:

```python
from causal_analysis import (
    fit_counterfactual_model, variance_attribution,
    scenario_effects, attribute_day, response_curve,
)

modelo = fit_counterfactual_model(resultado, top=8, n_boot=8)
variance_attribution(modelo).rows   # fatias da variação do alvo, com IC
scenario_effects(modelo).rows       # "e se estivesse no valor típico", com IC
attribute_day(modelo, rotulo).rows  # reparte o desvio de um período (Shapley)
response_curve(modelo, "temperatura").rows   # curva contrafactual por nível
```

`n_boot=0` desliga o comitê de bootstrap: sai mais rápido, sem intervalos de
confiança.

## Estrutura do código

```
app/               interface Streamlit (páginas e componentes)
capability/        motor do Módulo 1 (carta I-AM, normalidade, índices)
causal_analysis/   motor do Módulo 2 (testes, ML, scores, contrafactual)
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

O **contrafactual é do modelo, não do processo**: ele responde "o que o modelo
preveria se X estivesse no valor típico, mantendo o resto como esteve". Isso
vira leitura causal apenas sob três condições — nada relevante ficou fora da
planilha, o modelo acertou a forma da relação, e o cenário simulado existe no
histórico. A terceira é medida explicitamente (a contraprova por períodos
parecidos avisa quando o cenário é extrapolação); as duas primeiras dependem do
seu conhecimento do processo. O intervalo de confiança cobre a incerteza de
amostragem e de ajuste do modelo — **não** cobre viés por fator não medido, que
nenhum intervalo consegue cobrir. Um R² fora da amostra baixo invalida a
leitura: com o modelo errando muito, a repartição descreve o modelo, não o
processo.
