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
minutos. Cada sessão guarda **uma análise do Módulo 2 por vez** em memória; a
matriz do modelo é montada em float32 e a importância por permutação roda em
processo único — mudanças pensadas para o app não estourar a RAM.

## Uso por várias pessoas ao mesmo tempo

O Streamlit atende **todas as sessões num único processo**: sem limite, várias
análises simultâneas disputam os mesmos núcleos e a mesma RAM, e o resultado
não é "mais devagar" — é o processo travando. Por isso o app tem **fila**:
quem chega além do limite espera e vê sua posição, em vez de derrubar o app.

Nada disso exige mudar código — só variável de ambiente:

| Variável | Padrão | Para que serve |
|---|---|---|
| `ESMART_MAX_HEAVY_JOBS` | 2 | análises do Módulo 2 em paralelo |
| `ESMART_MAX_RENDER_JOBS` | 1 | PDFs em paralelo (cada gráfico usa um Chromium) |
| `ESMART_QUEUE_TIMEOUT_S` | 600 | espera máxima na fila antes de avisar |
| `ESMART_RF_JOBS` | núcleos ÷ `MAX_HEAVY_JOBS` | núcleos por Random Forest (`-1` = todos) |
| `ESMART_CACHE_SHEETS` | 2 | planilhas retidas em cache (global ao processo) |
| `ESMART_CACHE_INDICATOR` | 8 | tabelas de indicador em cache |
| `ESMART_CACHE_ANALYTICS` | 4 | bases da aba Analytics em cache |
| `ESMART_UPLOAD_TTL_H` | 24 | idade máxima dos uploads em disco |
| `ESMART_MAX_REPORT_ITEMS` | 40 | teto de análises por relatório |
| `ESMART_AUDIT_LOG` | (desligado) | caminho do log de auditoria (JSON Lines) |
| `ESMART_VERSION` | commit do git | versão carimbada nos relatórios |

Os caches do Streamlit são **globais ao processo**, não por sessão: com N
pessoas usando arquivos diferentes, `ESMART_CACHE_SHEETS` perto de N evita
reparse constante — ao custo de reter N planilhas em memória.

Para dimensionar a máquina, meça o custo de uma análise no seu hardware:

```bash
python benchmarks/bench_module2.py          # grades padrão
python benchmarks/bench_module2.py 10000 40 # uma grade específica
```

> **Plano gratuito não serve para um time inteiro.** ~1 vCPU torna as análises
> estritamente sequenciais. Para dezenas de pessoas, veja
> [`docs/PARECER_ESCALABILIDADE.md`](docs/PARECER_ESCALABILIDADE.md) — com o
> dimensionamento e os caminhos de hospedagem.

### Acesso e dados

Duas perguntas diferentes, que costumam vir grudadas:

| Pergunta | Quem resolve | Depende da hospedagem? |
|---|---|---|
| **Quem entra no app** | O Streamlit, nativamente (`st.login`, OIDC, ≥ 1.42) | **Não** |
| **Onde o dado repousa** | Ninguém por padrão — é decisão de hospedagem | **Sim** |

Hoje **o app não tem login**: publicado aberto, qualquer um sobe planilha e
dispara análise. Ligar o login não espera a decisão de infraestrutura — copie
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) para
`.streamlit/secrets.toml` (no Community Cloud, cole em *Settings > Secrets*),
preencha com o registro do app no provedor da empresa e chame `st.login()`.

O arquivo real está no `.gitignore` e **nunca** deve ser versionado — segredo
commitado fica no histórico mesmo depois de apagado, e precisa ser rotacionado
no provedor.

O que o login **não** resolve: as planilhas enviadas continuam gravadas no
disco de quem hospeda o app. Se houver restrição sobre dado operacional sair
da empresa, isso é requisito de hospedagem, não de autenticação.

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

Especificamente sobre o **ranking do Módulo 2**, e vale alinhar com quem for usar:

- **Leia a ORDEM, não o valor do score.** Os pesos e limiares são heurísticos,
  escolhidos por julgamento e não calibrados — servem para ordenar candidatos;
  o número absoluto não tem interpretação probabilística.
- **As linhas de evidência se sobrepõem.** Pearson, Spearman, contraste de
  percentis e a melhor transformação medem, em boa parte, a mesma associação
  monotônica; uma associação real é contada mais de uma vez, o que infla o
  rótulo de confiança.
- **Houve busca, e ela é cobrada.** O ranking testa ~18 transformações por
  indicador e até 12 lags de Granger, ficando com o melhor. O p-valor do
  vencedor é corrigido pelo número *efetivo* de testes independentes (Šidák
  sobre autovalores, método de Li & Ji) antes da FDR entre indicadores — a
  tabela mostra o p bruto e o ajustado lado a lado.
- **A importância do modelo só pontua se ele prevê.** Com R² fora da amostra
  abaixo de 0,05 o componente de machine learning é desconsiderado, e o
  relatório diz isso.
- **Tendência, sazonalidade e turno não são controlados** nos testes
  principais; só o Granger trata estacionariedade. Séries com tendência comum
  correlacionam sem relação causal.
- **O diagnóstico do dia prioriza, não decompõe.** É *relevância histórica ×
  atipicidade*: não calcula contribuição contrafactual nem intervalo de
  confiança. E, para dias que não são o último da base, usa informação
  posterior àquela data — o app avisa quando é o caso.

Análise completa em [`docs/PARECER_ESCALABILIDADE.md`](docs/PARECER_ESCALABILIDADE.md).
