# Parecer técnico — e-smart para uso por múltiplos times

**Pergunta avaliada:** o e-smart é a ferramenta certa para vários times fazerem,
diariamente, a análise de "o que impactou o meu indicador alvo hoje"? E ele aguenta
~50 pessoas usando ao mesmo tempo?

**Data:** agosto/2026 · **Versão avaliada:** branch `claude/code-review-scalability-dmbj2i`

---

## 1. Resumo executivo

| Pergunta | Resposta |
|---|---|
| O **método** de análise é bem construído? | **Sim** — a construção é sólida e acima da média do mercado. |
| Ele **mede o que diz medir**? | **Em parte.** A força da evidência era sistematicamente superestimada — corrigido nesta entrega. |
| A **funcionalidade** cobre o caso de uso diário? | **Sim**, com ressalva importante: prioriza o que investigar, não mede quem causou. |
| Aguenta **50 pessoas simultâneas** no Streamlit Community Cloud? | **Não.** Não é questão de ajuste fino: o plano gratuito é ~1 núcleo. |
| Aguenta 50 pessoas numa **máquina adequada**? | **Sim**, com as correções desta entrega e uma VM de porte médio. |
| Há risco **além do desempenho**? | **Sim** — o app está publicado **sem nenhuma autenticação**. |

**Recomendação:** migrar da hospedagem gratuita para uma VM/contêiner corporativo com
login integrado. As correções de código — de uso simultâneo **e de validade estatística** —
já foram aplicadas nesta entrega; o que falta é decisão de infraestrutura.

> **Ressalva de leitura, para alinhar com os times antes de distribuir:** o resultado é
> uma **fila de investigação priorizada**, não uma medida de causa. Ler a ORDEM dos
> indicadores, não o valor absoluto do score. A seção 3 explica por quê.

---

## 2. A construção do método é sólida

O Módulo 2 não é um "ranking de correlação" disfarçado — ele combina **sete linhas de
evidência** e as agrega com controle estatístico:

| Linha de evidência | O que captura |
|---|---|
| Pearson / Spearman / Kendall | associação linear, monotônica e ordinal |
| Correlação de distância (Székely) | dependência não-linear (ex.: relação em U) |
| Informação mútua | dependência de qualquer forma |
| Varredura de lag e média móvel | efeito com atraso e efeito acumulado |
| Granger (com teste ADF) | **precedência temporal** — causa vem antes |
| Contraste de percentis | efeito por faixa de operação |
| Random Forest + importância por permutação | não-linearidade e interações |

Três decisões técnicas merecem destaque porque são o tipo de coisa que ferramentas
comerciais costumam errar:

1. **Correção para testes múltiplos (FDR de Benjamini-Hochberg)**
   — `causal_analysis/pipeline.py:145-162`. Testar 40 indicadores contra um alvo gera
   falsos positivos por acaso; sem essa correção, o ranking apontaria "culpados"
   inexistentes com frequência.
2. **Validação temporal (`TimeSeriesSplit`)** — `causal_analysis/modeling.py:83`.
   A importância só é medida em blocos **futuros**. Embaralhar dados de série temporal
   é vazamento clássico e infla a importância de qualquer variável.
3. **Indício de efeito direto vs. indireto** — `causal_analysis/stats_tests.py:233`.
   Correlação parcial controlando os demais indicadores do topo: quando a associação
   "some", o efeito provavelmente passa por outro indicador (mediação). É uma versão
   leve e honesta de *causal discovery*.

A aba **Diagnóstico do dia** (`causal_analysis/day_diagnosis.py`) é literalmente o
caso de uso descrito. O docstring dela diz a coisa certa, que vale repetir para os times:

> Com uma única observação do alvo não existe análise causal possível. O que ela entrega
> é *score histórico do indicador* × *quão atípico ele esteve naquele dia* — um indício
> priorizado para investigação, **não prova de causa**.

Essa honestidade é uma qualidade da ferramenta, não uma limitação a esconder. Vale
alinhar isso com os times antes de distribuir: o resultado **prioriza hipóteses**;
a confirmação vem de teste controlado no processo.

**Qualidade do código:** alta. Sem estado global mutável, guardas de memória bem pensadas
(subamostragem em dCor e informação mútua, agregação por blocos, esforço adaptativo do
Random Forest), documentação de projeto dentro dos próprios módulos e 45 testes
automatizados. Os problemas listados adiante **não são desleixo** — são consequência
de o app ter sido afinado para um cenário diferente.

---

## 3. Validade estatística: a confiança era superestimada

Esta seção é, na prática, **mais importante que a de desempenho**. Uma ferramenta lenta
atrasa 50 pessoas; uma ferramenta confiante demais faz 50 pessoas agirem sobre a causa
errada, todo dia. Os pontos abaixo foram levantados na revisão, **confirmados no código**
e corrigidos.

### 3.1 "Procurar até achar" — corrigido

O ranking testava, para cada indicador, **18 transformações temporais** (lags 0–14 e três
médias móveis) e ficava com a de maior correlação; e **até 12 lags de Granger**, ficando
com o **menor p-valor**. Esses vencedores eram entregues à correção FDR *como se fossem
um teste único*. A FDR corrigia **entre indicadores**, nunca **dentro** de cada um.

O efeito é conhecido: o mínimo de 12 p-valores não tem distribuição de p-valor. Sob a
hipótese nula, "algum lag deu significativo" acontece com frequência muito maior que α.

**Correção aplicada:** o p do vencedor passa por ajuste de Šidák pelo **número efetivo de
testes independentes**, estimado pelos autovalores da matriz de correlação
(método de Li & Ji). Bonferroni sobre 18 seria conservador demais — lag 3 e lag 4 são
quase a mesma série; tratar o vencedor como teste único era anticonservador. O número
efetivo fica no meio e é o divisor honesto. Só então a FDR entre indicadores é aplicada.

O estimador se comporta como esperado nos dados de exemplo:

| Indicador | Transformações efetivas (de 18) | Leitura |
|---|---|---|
| `vazao_agua`, `temp_forno` | **6** | efeito suave: lags vizinhos quase idênticos |
| `pressao` | **8** | intermediário |
| `dosagem_quimica` | **16** | efeito de limiar: transformações pouco redundantes |

### 3.2 O efeito medido nos dados de exemplo

Rodando sobre `examples/dados_exemplo.csv`, cujos mecanismos são **conhecidos por
construção** (o gerador planta efeito com lag, efeito por média móvel, limiar por
percentil, um confundidor e duas variáveis de puro ruído):

| Indicador | p bruto | p ajustado | Transf. efetivas | Veredito final |
|---|---|---|---|---|
| `vazao_agua` (efeito real, forte) | 1,0 × 10⁻³⁰ | 6,1 × 10⁻³⁰ | 6 de 18 | Culpado provável |
| `pressao` (efeito real, linear) | 2,8 × 10⁻¹⁰ | 2,3 × 10⁻⁹ | 8 de 18 | Culpado possível |
| `dosagem_quimica` (efeito real, limiar) | 8,4 × 10⁻⁵ | 1,3 × 10⁻³ | 16 de 18 | Culpado possível |
| `temp_forno` (efeito real, não-linear) | **0,032** | **0,178** | 6 de 18 | Influência fraca |
| **`ruido_a` (puro ruído)** | **0,025** | **0,338** | 16 de 18 | Sem evidência |

Duas linhas contam a história inteira:

**`ruido_a` é a justificativa da correção.** É uma variável de **ruído puro**, sem
qualquer relação com o alvo — e o p bruto dela era **0,025**, que passaria por
"significativo a 5%". Esse 0,025 não vinha de sinal nenhum: vinha de ter testado 18
transformações e ficado com a melhor. Ajustado, vai a **0,338** — corretamente
sem evidência. Era exatamente esse tipo de falso positivo que chegava aos times.

**`temp_forno` é o custo da correção.** É um mecanismo *real* e foi rebaixado de "Culpado
possível" para "Influência fraca". Não é defeito: o p dele era **0,032 depois de escolher
a melhor entre 18 transformações** — estatisticamente indistinguível do ruído acima. Ele
continua em **2º lugar no ranking por score**; o que mudou foi o rótulo de confiança, que
agora reflete a evidência que de fato existe.

Os efeitos fortes passaram **intactos**: `vazao_agua`, `pressao` e `dosagem_quimica`
mantiveram veredito e confiança. A correção corta a cauda fraca, não o sinal.

Para não perder informação, a tabela de ranking agora traz **os dois p-valores** e o
número efetivo de transformações. A distância entre eles mostra quanto da evidência vinha
de ter procurado — informação que antes não existia em lugar nenhum.

### 3.3 Importância do modelo pontuava mesmo sem modelo — corrigido

O `R²` fora da amostra era calculado (`modeling.py:156`) e **nunca consultado pelo
score**. Como a importância por permutação é normalizada pelo máximo do grupo,
**alguém sempre recebia o componente ML cheio (15% do score)** — inclusive quando a
floresta era pior que chutar a média e as importâncias eram puro ruído.

**Correção aplicada:** abaixo de `R² = 0,05` o componente ML não pontua, e o relatório
diz explicitamente que o modelo não conseguiu prever o alvo e foi desconsiderado.

### 3.4 O que **permanece** como limite (documentado, não corrigido)

Registrado aqui porque muda como o resultado deve ser lido:

- **As linhas de evidência não são independentes.** Pearson, Spearman, o contraste de
  percentis e a melhor transformação medem, em boa parte, a **mesma** associação
  monotônica. Uma associação real é contada várias vezes, o que infla a contagem de
  significâncias — e portanto o rótulo de confiança. Corrigir isso exige reagrupar as
  famílias e recalibrar os pesos: muda todo ranking histórico, e ficou fora desta entrega
  por decisão de escopo.
- **Pesos e limiares do score são heurísticos**, escolhidos por julgamento e não
  calibrados contra dados com causa conhecida. Servem para **ordenar** candidatos; o valor
  absoluto não tem interpretação probabilística. É por isso que a orientação é ler a
  ordem, não o número.
- **Tendência, sazonalidade, turno e mudanças operacionais não são controlados** nos
  testes principais. Só o Granger trata estacionariedade (via ADF e diferenciação). Duas
  séries com tendência comum correlacionam sem qualquer relação causal — e uma causa
  comum não medida produz exatamente o mesmo padrão que uma causa real.
- **O Módulo 2 não distingue causa comum de causa direta** além da heurística de
  correlação parcial já existente.

### 3.5 Diagnóstico do dia: heurística de priorização, não decomposição causal

O "score do dia" é literalmente `relevância histórica × atipicidade do indicador`. Ele
**não** calcula contribuição contrafactual ("o que teria acontecido se X estivesse
normal"), **não** reparte a variação do alvo entre os indicadores e **não** produz
intervalo de confiança.

Além disso há **vazamento de informação futura**: tanto o ranking histórico quanto o
percentil do dia são calculados sobre a série inteira, **incluindo dias posteriores** ao
dia diagnosticado. Para um dia recente o efeito é pequeno; para reconstituir uma decisão
tomada há meses, é sério — o diagnóstico usa dados que não existiam naquela data.

**Aplicado nesta entrega:** o app agora avisa explicitamente nos dois casos — que o score
do dia não mede contribuição, e que o dia escolhido não é o último da base (portanto usa
informação futura), com a orientação de refazer a análise com a base cortada na data para
reconstituir a decisão de época.

Ele responde bem a **"o que investigar primeiro?"**. Não responde a **"quem causou?"**.

---

## 4. Por que ele não aguenta 50 pessoas hoje

### 4.1 A causa raiz

O app foi otimizado para **um usuário com muitos dados**. A demanda nova é o inverso:
**muitos usuários com dados moderados**. Praticamente toda decisão de engenharia que era
correta no primeiro cenário **se inverte** no segundo:

| Decisão | Ótima com 1 usuário | Efeito com 50 |
|---|---|---|
| `cache_resource(max_entries=1)` | 1 análise em RAM, não estoura | cada análise **apaga a da pessoa anterior** |
| `n_jobs=-1` no Random Forest | usa a máquina toda | 5 análises = 5× mais threads que núcleos |
| `max_entries=2` no cache de planilhas | não retém planilhas demais | quase toda leitura vira reparse do zero |
| Um diretório `/tmp` compartilhado | simples e suficiente | corrida de escrita e disco que só cresce |

### 4.2 O limite físico do Streamlit Community Cloud

O ponto decisivo não é o código, é a plataforma:

- O Streamlit atende **todas as sessões num único processo Python**. Cada sessão roda
  numa thread, mas CPU é CPU.
- O plano gratuito oferece **~1 vCPU e ~2,7 GB de RAM** (valor já registrado no README
  do próprio projeto).
- Uma análise do Módulo 2 é **CPU pura por minutos** — ver medição na seção 6.

Com um núcleo, as análises são **estritamente sequenciais**. E a medição da seção 7 dá a
dimensão: **10 minutos de CPU para uma grade pequena** (2 mil linhas, 10 indicadores) numa
máquina de 4 núcleos.

Faça a conta com o uso descrito — 50 pessoas, várias análises por pessoa ao longo do dia.
Mesmo assumindo apenas 2 análises por pessoa e a grade pequena acima:

```
100 análises/dia × 10 min = 1.000 minutos = ~16,7 horas de CPU
```

Isso é **mais que o dobro de um expediente**, num único núcleo, sem contar interface, sem
contar picos e usando a menor grade medida. Grades reais (dezenas de indicadores, meses de
dados) custam muito mais.

Cinquenta pessoas nesse cenário não formam fila lenta: formam fila que não anda, e a
memória estoura antes disso. **Nenhum ajuste de parâmetro resolve** — falta máquina.

### 4.3 Risco não relacionado a desempenho: o app é público

O app está publicado sem **nenhuma** autenticação: não há `st.login`, nem OIDC, nem
allowlist, nem `secrets.toml`. Qualquer pessoa com o endereço acessa a interface, sobe
uma planilha de até 200 MB e dispara uma análise pesada. Como o fluxo é upload manual,
**cada time subiria dado operacional próprio numa URL pública**.

**Duas perguntas que costumam vir grudadas — e que têm respostas diferentes:**

| Pergunta | Quem resolve | Bloqueado pela hospedagem? |
|---|---|---|
| **Quem entra no app** | O Streamlit, nativamente e bem (`st.login`, OIDC, ≥ 1.42) | **Não** |
| **Onde o dado repousa** | Ninguém, por padrão — é decisão de hospedagem | **Sim** |

Separar as duas muda a ordem das coisas: **o login pode ser feito hoje**, inclusive no
plano gratuito, sem esperar a decisão de infraestrutura. Basta cadastrar o app no Entra ID
da empresa, preencher `client_id`, `client_secret` e `cookie_secret`, e chamar
`st.login()`. O modelo está em `.streamlit/secrets.toml.example`; no Community Cloud o
mesmo conteúdo vai em *Settings > Secrets*. Usar o tenant específico da empresa (e não
`common`) é o que restringe o acesso a quem é da organização.

O que o login **não** resolve: as planilhas enviadas continuam gravadas no disco de quem
hospeda o app. Restringir quem entra não muda a jurisdição nem o dono da infraestrutura
onde o dado fica. Se houver restrição sobre dado operacional sair da empresa, isso é
requisito de hospedagem (seção 8) e nenhuma configuração de autenticação o satisfaz.

> **Higiene de segredo, aplicada nesta entrega:** o `.gitignore` não cobria
> `.streamlit/secrets.toml`. No momento em que a autenticação fosse ligada, um `git add .`
> distraído publicaria as credenciais num repositório público. Corrigido — junto com
> `secrets.toml`, `.env` e `*.env`. Verificado: **nenhum segredo chegou a ser commitado**,
> então não há histórico a limpar nem credencial a rotacionar.

---

## 5. Gargalos encontrados

Severidade considerando 50 usuários simultâneos.

| # | Gargalo | Onde | Sev. | Status |
|---|---|---|---|---|
| 1 | Nenhum limite de trabalho pesado simultâneo | app inteiro | 🔴 | **Corrigido** |
| 2 | `n_jobs=-1`: cada análise toma todos os núcleos | `causal_analysis/modeling.py:96` | 🔴 | **Corrigido** |
| 3 | Cache global de 1 slot: análises se apagam entre usuários | `app/page_module2.py:33` | 🔴 | **Corrigido** |
| 4 | Cache devolvia o **mesmo objeto** a todas as sessões | `app/page_module2.py:33` | 🔴 | **Corrigido** |
| 5 | PDF: Chromium sem limite nem timeout, erro engolido | `shared/pdf_export.py:44` | 🔴 | **Corrigido** |
| 6 | Corrida na gravação do upload (leitura de arquivo truncado) | `app/ui_components.py:26` | 🔴 | **Corrigido** |
| 7 | Uploads nunca apagados: disco só cresce | `app/ui_components.py:12` | 🟠 | **Corrigido** |
| 8 | Hash de até 200 MB **a cada clique** | `app/ui_components.py:22` | 🟠 | **Corrigido** |
| 9 | Mudar um slider disparava recálculo de minutos sem pedir | `app/page_module2.py:114` | 🟠 | **Corrigido** |
| 10 | `engine="python"` no CSV (5–20× mais lento) | `shared/parsing.py:71,87` | 🟠 | **Corrigido** |
| 11 | Módulo 1 refazia 4 ajustes MLE a cada tecla digitada | `app/page_module1.py:125` | 🟠 | **Corrigido** |
| 12 | Excel + CSV serializados a cada rerun na tela de outliers | `analytics/modules/outliers.py:175` | 🟠 | **Corrigido** |
| 13 | Cópia de DataFrame gravada na sessão a cada rerun | `analytics/modules/outliers.py:176` | 🟠 | **Corrigido** |
| 14 | Relatório crescia sem teto na memória da sessão | `app/ui_components.py:65` | 🟠 | **Corrigido** |
| 15 | Matriz do modelo alocada em dobro (float64 → float32) | `causal_analysis/features.py:44` | 🟠 | **Corrigido** |
| 16 | 34 colunas construídas por parâmetro para usar 1 | `causal_analysis/stats_tests.py:264` | 🟡 | **Corrigido** |
| 17 | Capacidade de cache fixa, dimensionada para 1 usuário | `app/data_store.py:21,31,39` | 🟠 | **Configurável** |
| 18 | Seleção "melhor de 18 transformações" não corrigida | `causal_analysis/stats_tests.py:118` | 🔴 | **Corrigido** |
| 19 | Menor p entre 12 lags do Granger tratado como teste único | `causal_analysis/stats_tests.py:201` | 🔴 | **Corrigido** |
| 20 | Importância do modelo pontuava com `R²` ruim ou negativo | `causal_analysis/scoring.py:60` | 🔴 | **Corrigido** |
| 21 | Šidák zerava p-valores minúsculos (cancelamento float) | `causal_analysis/stats_tests.py` | 🔴 | **Corrigido** |
| 22 | Diagnóstico do dia usa informação futura sem avisar | `causal_analysis/day_diagnosis.py` | 🟠 | **Avisado** |
| 23 | Relatório sem versão do algoritmo nem hash da base | `shared/pdf_export.py` | 🟠 | **Corrigido** |
| 24 | Sem trilha de auditoria | app inteiro | 🟠 | **Corrigido** (desligado por padrão) |
| 24b | Permutação calculada mesmo quando o score a descarta | `causal_analysis/modeling.py` | 🟠 | **Corrigido** (−98,8 %, §7.2) |
| 25 | Linhas de evidência redundantes inflam a confiança | `causal_analysis/scoring.py:14` | 🟠 | **Documentado** (§3.4) |
| 26 | Tendência/sazonalidade/turno não controlados | `causal_analysis/pipeline.py` | 🟠 | **Documentado** (§3.4) |
| 27 | Estado só em `session_state`: cai a sessão, perde tudo | app inteiro | 🟠 | **Documentado** |
| 28 | **Sem autenticação** | app inteiro | 🔴 | **Pronto para ligar** (§4.3) |
| 29 | **~1 núcleo no plano gratuito** | plataforma | 🔴 | **Depende da hospedagem** |
| 30 | Onde o dado repousa (planilhas no disco de quem hospeda) | plataforma | 🔴 | **Depende da hospedagem** |

O item 28 **não** espera a infraestrutura: o modelo de configuração está em
`.streamlit/secrets.toml.example` e só falta o registro do app no provedor de identidade
da empresa — que exige credenciais que só vocês têm. Os itens 29 e 30 são os que
realmente **não se resolvem no código**.

O item 21 merece nota: foi encontrado **pelos testes escritos para validar a própria
correção**. A forma literal `1-(1-p)^n` zera qualquer p abaixo de ~1e-16 em float64,
o que teria transformado a evidência mais forte da tabela num p-valor impossível.

---

## 6. O que foi corrigido nesta entrega

### 6.1 Fila ordenada em vez de queda coletiva — `shared/limits.py` (novo)

A mudança mais importante. Sem limite, N pessoas clicando "Analisar" juntas disputam os
mesmos núcleos e a mesma RAM — e o resultado não é "N vezes mais devagar", é o processo
travando ou morrendo. Agora há dois limites separados, porque os recursos escassos são
diferentes:

- **análises simultâneas** (limitado por CPU), padrão 2;
- **geração de PDF** (limitado por memória: cada gráfico dirige um Chromium de
  ~200–400 MB), padrão 1.

Quem chega além do limite **espera e vê sua posição na fila**, em vez de derrubar a
máquina. Tudo ajustável por variável de ambiente, sem tocar em código.

### 6.2 Isolamento entre usuários — `app/page_module2.py`

O resultado da análise era guardado num cache **global ao processo com um único slot**.
Isso somava três defeitos: cada análise nova apagava a da pessoa anterior (que
recalculava tudo ao clicar em qualquer aba); pedidos iguais ficavam presos sob lock
esperando a análise de outra pessoa; e o objeto era devolvido **por referência** a todas
as sessões — bastava uma escrita distraída para corromper o resultado alheio.

Agora cada sessão guarda o **seu** resultado. Também corrigido: mexer num slider sem
clicar em "Analisar" disparava minutos de recálculo que ninguém pediu.

### 6.3 Escrita atômica do upload — `app/ui_components.py`

O padrão anterior (`if not exists(): open(...,"wb")`) **trunca o arquivo** antes de
escrever. Uma segunda sessão que subisse a mesma planilha via o arquivo "existindo" e
lia um arquivo pela metade — falha aleatória de leitura sob carga. Agora a gravação é em
arquivo temporário + rename atômico. Somado a isso: limpeza automática por idade
(padrão 24 h) e fim do hash de 200 MB a cada clique.

### 6.4 Rastreabilidade — `shared/provenance.py` e `shared/audit.py` (novos)

Todo relatório em PDF passa a carimbar **versão do algoritmo** (commit do repositório) e
**SHA-256 de cada base analisada**. Sem isso, um relatório impresso é um número sem
passado: como o ranking muda quando o método muda — e mudou nesta entrega —, dois
relatórios da mesma planilha podem discordar legitimamente, e ninguém consegue explicar
por quê. Dois arquivos com o mesmo hash são o mesmo dado, com qualquer nome.

A trilha de auditoria registra início e fim de cada análise (base, hash, parâmetros,
duração) em JSON Lines. Fica **desligada por padrão**; ligue com `ESMART_AUDIT_LOG`.

> **Limite honesto:** o campo de usuário só é preenchido quando existe autenticação na
> frente do app. Sem login, a trilha registra **o que** aconteceu, não **quem** fez — e
> trilha sem identidade não sustenta auditoria de responsabilidade. É por isso que
> autenticação vem **antes** de auditoria na ordem de implantação.

### 6.5 Demais correções

Uso conservador de núcleos no Random Forest; leitura de CSV pelo engine C com fallback;
memória por sessão no Módulo 1; serialização de downloads sob demanda; teto no tamanho do
relatório; matriz do modelo montada direto em float32; e o caminho direto/indireto
recriando apenas a série de que precisa (antes montava 34 colunas por indicador para usar
uma — e ignorava o *lag* máximo escolhido pelo usuário).

### 6.6 Verificação

- Suíte completa: **58 testes passando** (a única falha, `test_loader_ods`, é falta da
  biblioteca `odfpy` no contêiner de teste e ocorre igual no código original).
- Motor validado ponta a ponta pela CLI sobre os dados sintéticos: os efeitos fortes
  mantêm veredito e confiança, e **as duas variáveis de ruído continuam sem evidência** —
  agora inclusive no p bruto ajustado, que antes dava 0,025 para ruído puro.
- Novos testes do limitador: respeito ao limite sob concorrência real, liberação do slot
  quando o bloco falha, e contador de fila sem vazamento após timeout.
- Novos testes da correção de seleção: séries idênticas contam como 1 teste, independentes
  contam quase todas, o ajuste é monotônico no nº de tentativas, **p-valores minúsculos
  não são zerados**, e o modelo sem poder preditivo não pontua.
- Teste de equivalência garantindo que a otimização do caminho direto/indireto produz
  **exatamente** a mesma série que a implementação anterior.

---

## 7. Medição de custo por análise

Executar na máquina candidata antes de dimensionar:

```bash
python benchmarks/bench_module2.py
```

Resultado medido nesta máquina de referência (4 vCPU, 16 GB), com os padrões do app:

| Linhas | Indicadores | Features | **Tempo** | Pico de RSS |
|---|---|---|---|---|
| 2.000 | 10 | 180 | **606 s (~10 min)** | 269 MB |

Uma grade **pequena** — 2 mil linhas e 10 indicadores — custa **dez minutos de CPU**.
É o número mais importante deste parecer, e ele sozinho responde à pergunta da seção 4.

### 7.1 Onde o tempo é gasto

```
python benchmarks/profile_steps.py 2000 10
```

| Etapa | Tempo | % do total |
|---|---|---|
| **Random Forest + importância por permutação** | **502,6 s** | **98,2 %** |
| Granger (com ADF) | 8,2 s | 1,6 % |
| Varredura de lags e médias móveis | 0,3 s | 0,1 % |
| Correlação de distância, informação mútua, percentis, Ljung-Box, correlações | < 0,5 s | ~0,1 % |
| **TOTAL** | **511,6 s** | 100 % |

**Uma única etapa consome 98 % do tempo.** Toda a bateria estatística — sete linhas de
evidência, correções, testes de precedência — custa menos de 2 % do total.

O motivo está na conta da importância por permutação: ela reavalia a floresta **uma vez
por feature, por repetição, por bloco de validação**. Com 180 features, 5 repetições e
4 blocos, são **~3.600 avaliações completas** de uma floresta de 300 árvores. E o número
de features é `(1 + lag_máximo + nº de médias móveis) × indicadores` — com os padrões,
**18 por indicador**, o que faz o custo crescer muito mais rápido que o nº de indicadores.

**Consequência prática — os dois botões que realmente importam:**

1. **Reduzir a defasagem máxima** (na interface, *Opções da análise*) corta features
   proporcionalmente. Passar de lag 14 para lag 7 tira ~40 % das colunas.
2. **Reduzir o número de indicadores** por análise. Planilhas de granularidade fina
   geram **9 métricas derivadas por coluna** na agregação multi-aba — 40 colunas viram
   360 indicadores, e daí ~6.500 features.

### 7.2 Atalho de R²: a análise não paga pelo que vai descartar

Decompondo aquela etapa dominante:

| Sub-etapa | Tempo | % |
|---|---|---|
| Ajustar as florestas + prever | 4,8 s | **1,2 %** |
| Importância por permutação | 389,4 s | **98,8 %** |

Como o componente ML só pontua quando o modelo prevê de fato (`R² ≥ 0,05`, seção 3.3),
calcular a parte cara **antes** de saber disso era gastar 99 % do tempo para descartar o
resultado no fim.

`ml_importance` passou a rodar em dois passos: mede o R² (1,2 % do custo) e só paga a
permutação se ela for entrar no score. Medido numa grade de 2 mil linhas × 10 indicadores
cujo alvo é ruído puro:

| | Tempo | R² |
|---|---|---|
| Antes | 390,1 s | −0,017 |
| Depois | **4,8 s** | −0,017 |

**98,8 % de redução — 385 segundos por análise** que antes eram gastos calculando
importâncias que o score jogava fora.

Quando o modelo **prevê**, o passo 2 reajusta as florestas: custa aquele 1,2 % a mais, e
com `random_state` fixo as florestas são idênticas, então o resultado não muda —
verificado nos dados de exemplo, ranking e vereditos inalterados. Guardar os modelos
entre os passos evitaria o reajuste, mas custaria centenas de MB de RAM para poupar 1 %.

> **Nota de reprodutibilidade, descoberta ao testar isto:** com a floresta multithread
> (`ESMART_RF_JOBS > 1`), o sklearn acumula as predições das árvores em ordem que depende
> do escalonamento das threads, e soma de ponto flutuante não é associativa. **Duas
> execuções idênticas divergem ~1e-16** — comportamento anterior a esta mudança. É
> irrelevante para o ranking (as faixas de veredito estão em 20/30/45 numa escala 0–100),
> mas quem precisar de reprodutibilidade bit a bit para auditoria deve fixar
> `ESMART_RF_JOBS=1`, que torna o resultado exato.

---

## 8. Caminhos de hospedagem

> Lembre da separação da seção 4.3: **autenticação não depende desta escolha** — pode ser
> ligada hoje, em qualquer das opções. O que depende é **onde o dado repousa**.

### Opção A — Continuar no Streamlit Community Cloud
- **Custo:** zero.
- **Recursos:** ~1 vCPU, ~2,7 GB, processo único, hiberna com inatividade.
- **Quem entra:** resolvível via `st.login`/OIDC com o Entra ID da empresa.
- **Onde o dado repousa:** infraestrutura da Snowflake/Streamlit, **fora da empresa** —
  as planilhas enviadas ficam no disco efêmero deles. É o ponto que costuma inviabilizar
  o plano gratuito para dado operacional, independente de desempenho.
- **Veredito:** serve para **1 a 3 pessoas**. **Não serve para 50** — e os termos de uso
  do plano gratuito não preveem uso produtivo corporativo.

### Opção B — VM ou contêiner corporativo ✅ **recomendada**
- **O que pedir para a TI:** máquina Linux com **8 a 16 vCPU e 32 GB de RAM**, Docker,
  e autenticação corporativa na frente do app.
- **Autenticação:** Azure App Service e equivalentes têm login integrado com o Entra ID;
  em VM pura, um `oauth2-proxy` na frente resolve. O Streamlit ≥ 1.42 também traz
  `st.login()` com OIDC nativo.
- **Custo:** se houver capacidade interna, tende a zero. Em nuvem sob demanda, uma
  máquina desse porte fica na **ordem de algumas centenas de reais por mês**, caindo
  bastante com instância reservada — *confirmar com a TI, o valor varia muito por
  contrato*.
- **Veredito:** é o caminho. Com as correções desta entrega, atende os 50 usuários com
  fila em horário de pico.

### Opção C — Streamlit in Snowflake / Databricks Apps
- Só faz sentido **se a empresa já usa** essa plataforma.
- **Vantagem grande:** autenticação e governança já resolvidas, e o dado já está lá —
  o que elimina o upload manual e abre a porta para o próximo passo (seção 8).

---

## 9. Dimensionamento e próximos passos

### Fórmula

```
núcleos ≈ (runs_por_dia × segundos_por_run) / (segundos_de_expediente × 0,6)   + folga de pico
RAM     ≈ base + (sessões_ativas × resultado_por_sessão) + (runs_simultâneos × pico_por_run)
```

O fator 0,6 é folga: uma máquina planejada para 100 % de ocupação não responde à
interface. A "folga de pico" importa porque o uso descrito **não é uniforme** — se os
times analisam de manhã, a demanda do dia inteiro se concentra em poucas horas.

### Ajuste fino por ambiente

Nenhuma dessas variáveis exige mudança de código:

| Variável | Padrão | Para que serve |
|---|---|---|
| `ESMART_MAX_HEAVY_JOBS` | 2 | análises simultâneas |
| `ESMART_MAX_RENDER_JOBS` | 1 | PDFs simultâneos |
| `ESMART_QUEUE_TIMEOUT_S` | 600 | espera máxima na fila |
| `ESMART_RF_JOBS` | núcleos ÷ `MAX_HEAVY_JOBS` | núcleos por Random Forest |
| `ESMART_CACHE_SHEETS` | 2 | planilhas retidas em cache |
| `ESMART_UPLOAD_TTL_H` | 24 | idade máxima dos uploads em disco |
| `ESMART_MAX_REPORT_ITEMS` | 40 | teto de análises por relatório |

Regra prática: com N pessoas usando **arquivos diferentes**, subir `ESMART_CACHE_SHEETS`
para perto de N evita reparse constante — ao custo de reter N planilhas em memória.

### O desperdício estrutural que vale registrar

Mesmo com tudo corrigido, o fluxo continua sendo: **cada pessoa exporta a planilha,
sobe no app, escolhe o alvo e espera minutos** — todo dia. Cinquenta pessoas fazendo isso
sobre dados que se repetem é o mesmo cálculo pago dezenas de vezes.

Quando existir um **banco de dados ou pasta de rede** que o app possa ler, o desenho certo
passa a ser:

1. **Lote noturno** calcula a análise de cada indicador alvo uma vez, fora do horário;
2. **Painel leve** de leitura, onde os times abrem o resultado do dia instantaneamente;
3. O app interativo continua existindo para **investigação ad hoc**.

Isso corta o custo de CPU em uma ordem de grandeza e transforma "esperar minutos" em
"abrir e ver". **Não está no escopo desta entrega** porque hoje a única fonte de dados é
upload manual — mas é a direção certa, e vale levantar com a TI junto do pedido de
infraestrutura.

### Sequência recomendada

1. **Ligar a autenticação** (`st.login` com o Entra ID) — **não depende da hospedagem** e
   pode ser feito hoje. É bloqueante para divulgar o link, e destrava a trilha de
   auditoria (seção 6.4), que sem identidade registra o quê mas não quem.
2. **Alinhar a leitura do resultado com os times** — que é fila de investigação
   priorizada, não medida de causa (seções 3.4 e 3.5). É o passo mais barato e o de maior
   impacto: evita decisão errada tomada com confiança indevida.
3. **Definir a hospedagem** (Opção B) e rodar `benchmarks/bench_module2.py` e
   `benchmarks/profile_steps.py` na máquina escolhida, com dados reais de um time.
   Levar à TI a pergunta certa: não "onde hospedar?", mas **"dado operacional pode
   repousar fora da empresa?"** — é ela que decide entre as opções.
4. **Ajustar as variáveis de ambiente** conforme a medição do passo 3.
5. **Piloto com 2 ou 3 times** por duas semanas, observando fila, memória e — importante —
   se os vereditos mais conservadores fazem sentido para quem conhece o processo.
6. Só então abrir para os 50.
7. Em paralelo, **levantar se existe banco/historiador** acessível — é o que destrava o
   lote noturno e corta o custo de CPU em uma ordem de grandeza.

### Itens conhecidos que ficaram fora desta entrega

Registrados para decisão futura, não esquecidos. **Nada aqui é pendência esquecida** —
cada um saiu do escopo por um motivo declarado.

| Item | Por que ficou fora |
|---|---|
| **Recalibrar o score** colapsando as linhas de evidência redundantes (§3.4) | Decisão de escopo: entre "só documentar", "p ajustado por seleção" e "recalibrar tudo", foi escolhido o caminho do meio. Recalibrar muda todo ranking histórico e pede validação contra casos com causa conhecida pelo processo. |
| **Controlar tendência, sazonalidade e turno** nos testes principais | Mudança de método, não correção de defeito. Exige decidir *como* (regressão com dummies de turno? diferenciação? decomposição?) — e cada escolha muda o resultado. |
| **Persistência dos resultados** em disco | Decisão de escopo na pergunta de governança: foram escolhidas proveniência e auditoria. Faz mais sentido junto com a hospedagem definida, que decide *onde* persistir. |
| **Métricas operacionais** (fila, memória, taxa de erro) | Mesma pergunta de governança, não selecionado. |
| ~~**Pular a permutação quando o `R²` não justifica**~~ | **Implementado** — ver §7.2. Medido: 390,1 s → 4,8 s (−98,8 %) quando o modelo não prevê, sem alterar o resultado quando prevê. |
| **Autenticação de fato ligada** | Exige o registro do app no provedor de identidade da empresa, com credenciais que só vocês têm. O scaffold está pronto (§4.3). |
