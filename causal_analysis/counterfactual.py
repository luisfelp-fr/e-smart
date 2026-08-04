"""Contrafactual, repartição da variação do alvo e intervalos de confiança.

O ranking do Módulo 2 responde *quem* influencia o alvo (força da evidência,
0–100). Ele não responde três perguntas que a engenharia de processo faz logo
em seguida — e é isso que este módulo entrega:

1. **Contrafactual** — "o que teria acontecido se X estivesse normal?"
   Um modelo preditivo é reajustado e consultado duas vezes: com os valores
   que realmente ocorreram e com o indicador X forçado ao seu valor típico
   (mediana histórica). A diferença entre as duas previsões é o efeito
   atribuído a X ter saído do normal, **na unidade do alvo**.

2. **Repartição da variação** — "de toda a variação do alvo, quanto cabe a
   cada indicador?" Usa o **valor de Shapley** (teoria dos jogos) calculado
   por amostragem de permutações: é a única repartição que satisfaz a
   propriedade de *eficiência* — as fatias somam EXATAMENTE o desvio do dia
   em relação ao dia típico. Agregando as fatias no período, a covariância de
   cada fatia com a previsão reparte a variância do alvo (as fatias somam
   100% do que o modelo explica; o resto vira "não explicado").

3. **Intervalo de confiança** — todas as estimativas acima saem com IC por
   **bootstrap por blocos** (reamostra blocos de períodos consecutivos, o que
   preserva a autocorrelação da série) sobre um comitê de modelos. Para o
   Shapley soma-se ainda o erro de Monte Carlo da amostragem de permutações.

Duas decisões de modelagem importantes:

- **Agrupamento por indicador físico.** Quando uma aba fina gera família de
  métricas (média, máximo, P90, % tempo>Q3...), todas as métricas do MESMO
  indicador entram no mesmo grupo. "E se a temperatura estivesse normal"
  move a família inteira de uma vez — cenário fisicamente coerente. Mover só
  o "máximo" deixando a "média" como esteve seria impossível no processo real.
- **Matriz enxuta.** Cada indicador entra com o valor bruto e a MELHOR versão
  temporal encontrada na varredura de lags/médias móveis do ranking. Isso
  mantém a estrutura temporal relevante com uma matriz ~10× menor que a do
  Random Forest do ranking — o que torna viável rodar o comitê de bootstrap.

LIMITE HONESTO: o contrafactual é **do modelo**, não do processo. Ele só vira
resposta causal sob as hipóteses usuais (nada relevante ficou fora da
planilha, o modelo acertou a forma da relação e o cenário simulado existe no
histórico). O módulo mede a última hipótese explicitamente — a checagem por
vizinhos históricos compara o contrafactual do modelo com o que de fato
aconteceu em dias parecidos — e avisa quando o cenário é extrapolação.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from .aggregation import METRIC_FRIENDLY, metric_of
from .features import derived_features
from .pipeline import AnalysisResult

RNG_SEED = 42
OTHERS = "demais indicadores"

# amostragem de permutações do Shapley
DAY_PERMUTATIONS = 200      # decomposição de um dia (barato: 1 linha)
VAR_PERMUTATIONS = 24       # repartição da variância (muitas linhas)
VAR_ROWS = 150              # linhas avaliadas na repartição da variância

# R² fora da amostra abaixo disso: o modelo não sustenta contrafactual
MIN_R2 = 0.05


# --------------------------------------------------------------- utilidades

def _fmt(v: float, nd: int = 4) -> str:
    """Número em pt-BR ("12,34"); "—" quando ausente."""
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}g}".replace(".", ",")


def _group_key(column: str) -> str:
    """Indicador físico de uma coluna, preservando o prefixo da aba.

    "forno: temp (máximo)" -> "forno: temp"; "temp (P90)" -> "temp". Ao
    contrário de ``base_indicator``, NÃO descarta o nome da aba: duas abas
    podem ter uma coluna "temp" e são indicadores diferentes.
    """
    return column[: column.rfind(" (")] if metric_of(column) else column


def _friendly(group: str) -> str:
    """Nome do grupo para exibição ('forno: temp')."""
    return OTHERS if group == OTHERS else f"'{group}'"


def _metrics_note(cols: list[str]) -> str:
    """Descreve as métricas reunidas num grupo (para tooltip/relatório)."""
    metrics = [m for m in dict.fromkeys(metric_of(c) for c in cols) if m]
    if not metrics:
        return "valor no período"
    return "; ".join(METRIC_FRIENDLY.get(m, m) for m in metrics)


def _ci_from_boot(point: float, boot: list[float] | np.ndarray,
                  conf: float = 0.95, extra_se: float = 0.0) -> tuple[float, float]:
    """IC ao redor de ``point`` pela dispersão do comitê de bootstrap.

    Usa aproximação t (poucas réplicas), somando em quadratura um erro extra
    conhecido (ex.: o erro de Monte Carlo da amostragem de permutações do
    Shapley). Devolve (nan, nan) quando não há réplicas suficientes.
    """
    vals = np.asarray([v for v in np.ravel(boot) if np.isfinite(v)], dtype=float)
    if len(vals) < 3 or not np.isfinite(point):
        return np.nan, np.nan
    se = float(vals.std(ddof=1))
    se = float(np.sqrt(se**2 + max(0.0, extra_se) ** 2))
    if not np.isfinite(se):
        return np.nan, np.nan
    half = float(stats.t.ppf(0.5 + conf / 2.0, len(vals) - 1)) * se
    return float(point - half), float(point + half)


def _percentile_of(values: np.ndarray, v: float) -> float:
    """Percentil empírico (0–100) de ``v`` dentro de ``values``."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or not np.isfinite(v):
        return np.nan
    return float(100.0 * (x <= v).mean())


def _block_bootstrap_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma reamostra por blocos móveis (preserva autocorrelação).

    Reamostrar linhas isoladas de uma série temporal subestima a incerteza:
    observações vizinhas carregam a mesma informação. Blocos de períodos
    consecutivos mantêm essa dependência dentro da reamostra.
    """
    block = int(max(1, min(block, n)))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


# --------------------------------------------------- montagem da matriz enxuta

def _lean_matrix(
    result: AnalysisResult, max_params: int
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]], list[str]]:
    """Matriz com o valor bruto + a melhor versão temporal de cada indicador.

    Devolve (X, y, {grupo -> colunas de X}, notas). Os grupos reúnem todas as
    métricas do MESMO indicador físico (média/máximo/P90 de 'temp' viram um
    grupo só), para que o cenário contrafactual mova a família inteira.
    """
    df, target = result.df, result.target
    notes: list[str] = []

    ranked = [p for p in list(result.scores["parametro"]) if p != target]
    if len(ranked) > max_params:
        notes.append(
            f"{len(ranked)} colunas disponíveis: o modelo contrafactual usou "
            f"as {max_params} mais bem colocadas no ranking (as demais têm "
            "evidência fraca e encareceriam o comitê de bootstrap)."
        )
        ranked = ranked[:max_params]

    cols: dict[str, pd.Series] = {}
    group_cols: dict[str, list[str]] = {}
    for p in ranked:
        best = result.per_param.get(p, {}).get("best_feature") or p
        wanted = [p] if best == p else [p, best]
        fam = None
        for feat in wanted:
            if feat in cols:
                continue
            if feat == p:
                cols[feat] = df[p]
                continue
            if fam is None:
                fam = derived_features(df[p], result.max_lag, result.windows)
            if feat in fam.columns:
                cols[feat] = fam[feat]
        atual = group_cols.setdefault(_group_key(p), [])
        atual.extend(f for f in wanted if f in cols and f not in atual)

    X = pd.DataFrame(cols)
    y = df[target]
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask], group_cols, notes


# ------------------------------------------------------------------- modelo

@dataclass
class CounterfactualModel:
    """Comitê de modelos + referência típica, pronto para consultas 'e se'."""

    target: str = ""
    columns: list[str] = field(default_factory=list)
    index: pd.Index | None = None
    Xv: np.ndarray | None = None          # matriz factual (n × d), float32
    yv: np.ndarray | None = None
    baseline: np.ndarray | None = None    # linha "tudo no valor típico"
    group_order: list[str] = field(default_factory=list)   # grupos decompostos
    group_idx: dict[str, np.ndarray] = field(default_factory=dict)
    group_params: dict[str, list[str]] = field(default_factory=dict)
    col_group: np.ndarray | None = None   # coluna -> índice do grupo
    model: object | None = None
    boot_models: list = field(default_factory=list)
    r2_oos: float = np.nan
    resid_sigma: float = np.nan
    y_typical: float = np.nan             # previsão do "período totalmente típico"
    conf: float = 0.95
    notes: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    # ---- consultas de baixo nível -------------------------------------
    @property
    def ok(self) -> bool:
        return self.skipped_reason is None and self.model is not None

    def _predict(self, model, M: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(np.ascontiguousarray(M)), dtype=float)

    def typical_value(self, group: str) -> float:
        """Valor típico (mediana) da coluna representativa do grupo."""
        cols = self.group_params.get(group, [])
        if not cols:
            return np.nan
        j = self.columns.index(cols[0])
        return float(self.baseline[j])


def _fit_forest(X: np.ndarray, y: np.ndarray, n_trees: int,
                seed: int) -> RandomForestRegressor:
    mtry = "sqrt" if X.shape[1] > 200 else 0.5
    model = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=5,      # árvores menores: menos memória e superfície
        max_features=mtry,       # contrafactual mais suave (menos degraus)
        random_state=seed,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def fit_counterfactual_model(
    result: AnalysisResult,
    top: int = 8,
    n_boot: int = 8,
    n_estimators: int = 150,
    boot_estimators: int = 50,
    max_params: int = 40,
    max_fit_rows: int = 6000,
    conf: float = 0.95,
    random_state: int = RNG_SEED,
) -> CounterfactualModel:
    """Ajusta o modelo contrafactual e o comitê de bootstrap por blocos.

    ``top`` indicadores (por score) são decompostos individualmente; os demais
    entram como um único grupo "demais indicadores" — assim a repartição
    continua somando 100% do desvio sem explodir o custo.

    ``n_boot=0`` desliga o comitê: as estimativas saem sem intervalo de
    confiança (mais rápido, útil em máquinas pequenas).
    """
    cfm = CounterfactualModel(target=result.target, conf=conf)
    if result.scores is None or result.scores.empty:
        cfm.skipped_reason = "o ranking do Módulo 2 não produziu resultados."
        return cfm

    X, y, group_cols, notes = _lean_matrix(result, max_params)
    cfm.notes.extend(notes)
    if len(X) < 60:
        cfm.skipped_reason = (
            f"apenas {len(X)} períodos completos após montar as versões "
            "temporais (mínimo: 60) — sem base para contrafactual."
        )
        return cfm
    if X.shape[1] < 2:
        cfm.skipped_reason = "menos de dois indicadores utilizáveis."
        return cfm

    # ordem dos grupos pelo melhor score entre suas colunas
    score_of = dict(zip(result.scores["parametro"], result.scores["score"]))
    best_score = {
        g: max(score_of.get(c, 0.0) for c in cols) for g, cols in group_cols.items()
    }
    ranked_groups = sorted(best_score, key=lambda g: -best_score[g])
    head = ranked_groups[:top]
    tail = ranked_groups[top:]

    cfm.index = X.index
    cfm.columns = list(X.columns)
    cfm.Xv = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    cfm.yv = y.to_numpy(dtype=np.float32)
    # "valor típico" de cada coluna: a mediana do que o processo de fato fez
    cfm.baseline = np.nanmedian(cfm.Xv, axis=0).astype(np.float32)

    col_pos = {c: i for i, c in enumerate(cfm.columns)}
    cfm.group_order = list(head) + ([OTHERS] if tail else [])
    for g in head:
        cfm.group_params[g] = list(group_cols[g])
        cfm.group_idx[g] = np.array([col_pos[c] for c in group_cols[g]], dtype=int)
    if tail:
        cols_tail = [c for g in tail for c in group_cols[g]]
        cfm.group_params[OTHERS] = cols_tail
        cfm.group_idx[OTHERS] = np.array([col_pos[c] for c in cols_tail], dtype=int)

    cfm.col_group = np.empty(len(cfm.columns), dtype=int)
    for gi, g in enumerate(cfm.group_order):
        cfm.col_group[cfm.group_idx[g]] = gi

    # ---- ajuste ---------------------------------------------------------
    fit_idx = np.arange(len(X))
    if len(X) > max_fit_rows:
        fit_idx = np.linspace(0, len(X) - 1, max_fit_rows).astype(int)
        cfm.notes.append(
            f"{len(X)} períodos: o modelo foi ajustado sobre {max_fit_rows} "
            "linhas igualmente espaçadas (mantém a cobertura do período e "
            "segura o tempo de resposta)."
        )
    Xf, yf = cfm.Xv[fit_idx], cfm.yv[fit_idx]
    cfm.model = _fit_forest(Xf, yf, n_estimators, random_state)
    cfm.y_typical = float(cfm._predict(cfm.model, cfm.baseline[None, :])[0])

    # ---- poder preditivo fora da amostra (validação temporal) -----------
    n_splits = int(min(4, max(2, len(Xf) // 60)))
    preds, obs = [], []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(Xf):
        if len(te) < 10:
            continue
        m = _fit_forest(Xf[tr], yf[tr], max(30, boot_estimators), random_state)
        preds.append(cfm._predict(m, Xf[te]))
        obs.append(yf[te])
    if preds:
        p, o = np.concatenate(preds), np.concatenate(obs)
        cfm.r2_oos = float(r2_score(o, p))
        cfm.resid_sigma = float(np.std(o - p, ddof=1))

    # ---- comitê de bootstrap por blocos (fonte dos ICs) -----------------
    rng = np.random.default_rng(random_state)
    block = int(max(2, min(len(fit_idx) // 10, result.max_lag + 1)))
    if 0 < n_boot < 3:
        # abaixo de 3 réplicas não há dispersão que sustente um IC; não vale
        # pagar o ajuste dos modelos
        n_boot = 0
        cfm.notes.append(
            "Comitê de bootstrap muito pequeno: os intervalos de confiança "
            "foram suprimidos (mínimo de 3 réplicas)."
        )
    for b in range(int(max(0, n_boot))):
        bidx = _block_bootstrap_idx(len(Xf), block, rng)
        cfm.boot_models.append(
            _fit_forest(Xf[bidx], yf[bidx], boot_estimators, random_state + 1 + b)
        )

    # ---- cautelas padrão -------------------------------------------------
    if not np.isfinite(cfm.r2_oos) or cfm.r2_oos < MIN_R2:
        cfm.cautions.append(
            f"O modelo praticamente não prevê '{result.target}' fora da "
            f"amostra (R² = {_fmt(cfm.r2_oos, 2)}). Sem poder preditivo, o "
            "contrafactual é especulação: trate os números abaixo como "
            "indicativos e priorize coletar mais/outros indicadores."
        )
    cfm.cautions.append(
        "O contrafactual é do MODELO, não do processo: ele responde 'o que o "
        "modelo preveria se X estivesse no valor típico, mantendo o resto "
        "como esteve'. Isso vira leitura causal só se nada relevante ficou "
        "fora da planilha e se esse cenário existe no histórico — a checagem "
        "por períodos parecidos (na repartição de um período) testa a segunda "
        "condição; a primeira depende do seu conhecimento do processo."
    )
    if tail:
        cfm.notes.append(
            f"Os {len(tail)} indicadores fora do topo entram agrupados como "
            f"'{OTHERS}' — assim a repartição continua fechando 100%."
        )
    return cfm


# ------------------------------------------------------- Shapley por grupos

def _permutation_masks(n_groups: int, n_perm: int,
                       col_group: np.ndarray,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de coalizão para o Shapley por amostragem de permutações.

    Devolve (masks, perms):
    - ``masks[i, j]`` (booleano por COLUNA) = quais colunas já estão no valor
      factual no passo j da permutação i (passo 0 = tudo no valor típico,
      passo k = tudo factual);
    - ``perms[i, j]`` = qual grupo entrou no passo j.

    A mesma amostra de permutações é usada em todas as linhas e em todos os
    modelos do comitê: assim a diferença entre réplicas reflete incerteza do
    modelo, não ruído de sorteio.
    """
    perms = np.array([rng.permutation(n_groups) for _ in range(n_perm)])
    present = np.zeros((n_perm, n_groups + 1, n_groups), dtype=bool)
    for i in range(n_perm):
        for j in range(n_groups):
            present[i, j + 1:, perms[i, j]] = True
    return present[:, :, col_group], perms


def _shapley(cfm: CounterfactualModel, model, rows: np.ndarray,
             masks: np.ndarray, perms: np.ndarray,
             chunk_cells: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Valores de Shapley por grupo para cada linha de ``rows``.

    Devolve (phi, phi_mc_se), ambos (n_linhas × n_grupos). Por construção,
    ``phi.sum(axis=1) == f(linha) - f(linha típica)`` exatamente — é a
    propriedade de eficiência que faz as fatias fecharem o desvio.
    """
    n_perm, n_states, d = masks.shape
    k = n_states - 1
    baseline = cfm.baseline
    per_row = n_perm * n_states
    chunk = int(max(1, chunk_cells // max(1, per_row * d)))

    phi = np.zeros((len(rows), k))
    mc_se = np.zeros((len(rows), k))
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk]
        # (linhas, perms, estados, colunas) -> factual onde a máscara é True
        states = np.where(
            masks[None, :, :, :], block[:, None, None, :], baseline[None, None, None, :]
        ).astype(np.float32, copy=False)
        preds = cfm._predict(model, states.reshape(-1, d))
        preds = preds.reshape(len(block), n_perm, n_states)
        inc = np.diff(preds, axis=2)                     # (linhas, perms, k)
        contrib = np.zeros_like(inc)
        rows_i = np.arange(n_perm)[:, None]
        for r in range(len(block)):
            contrib[r][rows_i, perms] = inc[r]           # incremento -> grupo
        phi[start:start + chunk] = contrib.mean(axis=1)
        mc_se[start:start + chunk] = contrib.std(axis=1, ddof=1) / np.sqrt(n_perm)
    return phi, mc_se


# ------------------------------------------------ contrafactual "e se típico"

def _counterfactual_predictions(cfm: CounterfactualModel, model,
                                rows: np.ndarray) -> tuple[np.ndarray, dict]:
    """Previsão factual e, por grupo, a previsão com aquele grupo no típico."""
    fact = cfm._predict(model, rows)
    out = {}
    for g in cfm.group_order:
        alt = rows.copy()
        alt[:, cfm.group_idx[g]] = cfm.baseline[cfm.group_idx[g]]
        out[g] = cfm._predict(model, alt)
    return fact, out


@dataclass
class ScenarioEffects:
    """'E se este indicador tivesse ficado no valor típico o período todo?'"""

    target: str = ""
    n_periods: int = 0
    observed_mean: float = np.nan
    rows: pd.DataFrame | None = None
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def scenario_effects(cfm: CounterfactualModel, max_eval_rows: int = 2000
                     ) -> ScenarioEffects:
    """Efeito médio no alvo de cada indicador ter saído do valor típico.

    Para cada grupo, compara a previsão factual com a previsão do cenário em
    que aquele indicador ficou na mediana histórica em TODOS os períodos,
    mantendo os demais como de fato estiveram. A diferença média é o efeito
    atribuído — em unidade do alvo, com intervalo de confiança.
    """
    out = ScenarioEffects(target=cfm.target)
    if not cfm.ok:
        out.cautions.append(cfm.skipped_reason or "modelo indisponível.")
        return out

    idx = np.arange(len(cfm.Xv))
    if len(idx) > max_eval_rows:
        idx = np.linspace(0, len(idx) - 1, max_eval_rows).astype(int)
    rows = cfm.Xv[idx]
    out.n_periods = len(idx)
    out.observed_mean = float(np.mean(cfm.yv[idx]))

    def _stats(model) -> dict[str, tuple[float, float]]:
        """(efeito médio no nível, deslocamento típico) por grupo."""
        f, a = _counterfactual_predictions(cfm, model, rows)
        return {g: (float(np.mean(f - a[g])), float(np.mean(np.abs(f - a[g]))))
                for g in cfm.group_order}

    fact, alt = _counterfactual_predictions(cfm, cfm.model, rows)
    boot = [_stats(m) for m in cfm.boot_models]
    labels = cfm.index[idx]

    lines = []
    for g in cfm.group_order:
        delta = fact - alt[g]
        effect, swing = float(np.mean(delta)), float(np.mean(np.abs(delta)))
        lo, hi = _ci_from_boot(effect, [b[g][0] for b in boot], cfm.conf)
        s_lo, s_hi = _ci_from_boot(swing, [b[g][1] for b in boot], cfm.conf)
        pico = int(np.argmax(np.abs(delta)))
        # quanto tempo o indicador passou fora da faixa habitual
        cols = cfm.group_idx[g]
        z = cfm.Xv[idx][:, cols].astype(float)
        q1, q3 = np.nanpercentile(cfm.Xv[:, cols].astype(float), [25, 75], axis=0)
        fora = float(100.0 * np.mean(((z < q1) | (z > q3)).any(axis=1)))
        conclusivo = np.isfinite(lo) and (lo > 0) == (hi > 0)
        lines.append({
            "indicador": _friendly(g),
            "grupo": g,
            "deslocamento típico": swing,
            "deslocamento IC-": s_lo,
            "deslocamento IC+": s_hi,
            "alvo médio observado": out.observed_mean,
            "alvo médio se típico": float(np.mean(alt[g])),
            "efeito no nível médio": effect,
            "IC inferior": lo,
            "IC superior": hi,
            "maior deslocamento": float(delta[pico]),
            "quando": labels[pico],
            "% do tempo fora do típico": round(fora, 1),
            "efeito no nível é conclusivo": "sim" if conclusivo
            else "não (IC cruza zero)",
            "métricas reunidas": _metrics_note(cfm.group_params[g]),
        })
    tab = pd.DataFrame(lines).sort_values(
        "deslocamento típico", ascending=False).reset_index(drop=True)
    tab.index = tab.index + 1
    out.rows = tab

    # ---- frases gerenciais ---------------------------------------------
    out.findings.append(
        f"No período analisado ({out.n_periods} períodos), '{cfm.target}' "
        f"ficou em média {_fmt(out.observed_mean)}. As linhas abaixo respondem, "
        "para cada indicador: quanto o alvo se deslocou por ele ter operado "
        "fora do valor típico, período a período."
    )
    for _, r in tab.head(5).iterrows():
        if r["grupo"] == OTHERS or not np.isfinite(r["deslocamento típico"]):
            continue
        ic = (f" (IC {int(cfm.conf * 100)}%: {_fmt(r['deslocamento IC-'])} a "
              f"{_fmt(r['deslocamento IC+'])})") \
            if np.isfinite(r["deslocamento IC-"]) else ""
        out.findings.append(
            f"{r['indicador']}: manter este indicador sempre no valor típico "
            f"({_fmt(cfm.typical_value(r['grupo']))}) teria tirado do alvo, em "
            f"média, {_fmt(r['deslocamento típico'])} de oscilação por período"
            f"{ic} — e até {_fmt(abs(r['maior deslocamento']))} no pior "
            f"período. Esteve fora da faixa habitual em "
            f"{r['% do tempo fora do típico']:.0f}% do tempo."
        )
        if r["efeito no nível é conclusivo"] == "sim":
            sinal = "acima" if r["efeito no nível médio"] > 0 else "abaixo"
            out.findings.append(
                f"   ↳ além de oscilar, {r['indicador']} deixou a MÉDIA do "
                f"alvo {_fmt(abs(r['efeito no nível médio']))} {sinal} do que "
                "seria com ele sempre no típico — é um viés sistemático, não "
                "só variabilidade."
            )
    out.cautions.append(
        "O 'deslocamento típico' é um módulo (média de |efeito|), então é "
        "sempre positivo: ele mede o TAMANHO do efeito, não a evidência de "
        "que ele exista. Para saber se o efeito é real, use o IC do efeito no "
        "nível médio, a fatia da variação (com IC) e o score do ranking."
    )
    out.cautions.extend(cfm.cautions)
    if not cfm.boot_models:
        out.cautions.append(
            "Comitê de bootstrap desligado: os efeitos saíram sem intervalo "
            "de confiança."
        )
    return out


# --------------------------------------------- checagem empírica por vizinhos

# distância máxima (em desvios-padrão, por indicador) para um período contar
# como "parecido": acima disso o pareamento vira comparação com outra coisa
MATCH_MAX_DIST = 0.75


def _matched_counterfactual(cfm: CounterfactualModel, pos: int, group: str,
                            k: int = 20) -> dict:
    """Contrafactual empírico: períodos parecidos, mas com o indicador normal.

    Não usa o modelo. Procura no histórico os períodos em que o indicador
    esteve na faixa central (P35–P65) E os DEMAIS indicadores do topo
    estiveram próximos do período analisado; devolve a mediana do alvo
    observado nesses períodos. Serve de contraprova ao contrafactual do
    modelo — e, principalmente, denuncia EXTRAPOLAÇÃO: se não existem
    períodos parecidos, a resposta do modelo não tem lastro no histórico e a
    checagem se declara sem suporte (``qualidade='fraca'``).
    """
    vazio = {"n": 0, "y": np.nan, "lo": np.nan, "hi": np.nan,
             "dist": np.nan, "qualidade": "sem dados"}
    cols = cfm.group_idx[group]
    others = np.concatenate(
        [cfm.group_idx[g] for g in cfm.group_order if g != group]
    ) if len(cfm.group_order) > 1 else np.array([], dtype=int)

    X = cfm.Xv.astype(float)
    band = np.nanpercentile(X[:, cols], [35, 65], axis=0)
    normal = np.all((X[:, cols] >= band[0]) & (X[:, cols] <= band[1]), axis=1)
    if normal.sum() < 5:   # faixa central estreita demais: alarga para o IQR
        band = np.nanpercentile(X[:, cols], [25, 75], axis=0)
        normal = np.all((X[:, cols] >= band[0]) & (X[:, cols] <= band[1]), axis=1)
    normal[pos] = False
    cand = np.flatnonzero(normal)
    if len(cand) < 5:
        return {**vazio, "n": int(len(cand))}

    if len(others):
        sd = np.nanstd(X[:, others], axis=0)
        sd[sd == 0] = 1.0
        d = np.linalg.norm((X[np.ix_(cand, others)] - X[pos, others]) / sd, axis=1)
        d /= np.sqrt(len(others))          # distância média por indicador
        order = np.argsort(d)[:k]
        # só entram os realmente parecidos: um "vizinho" a 3 desvios não é
        # contraprova, é outro regime de operação
        order = order[d[order] <= MATCH_MAX_DIST]
        if len(order) < 5:
            return {**vazio, "n": int(len(order)),
                    "dist": float(np.min(d)) if len(d) else np.nan,
                    "qualidade": "fraca"}
        sel, dist = cand[order], float(np.median(d[order]))
    else:
        sel, dist = cand[:k], np.nan

    vals = cfm.yv[sel].astype(float)
    rng = np.random.default_rng(RNG_SEED)
    boots = [float(np.median(rng.choice(vals, len(vals), replace=True)))
             for _ in range(400)]
    lo, hi = np.percentile(boots, [(1 - cfm.conf) / 2 * 100,
                                   (1 + cfm.conf) / 2 * 100])
    return {"n": int(len(sel)), "y": float(np.median(vals)),
            "lo": float(lo), "hi": float(hi), "dist": dist,
            "qualidade": "boa"}


# ------------------------------------------------- repartição do desvio do dia

@dataclass
class DayAttribution:
    """Repartição do desvio de um dia entre os indicadores (soma exata)."""

    label: object = None
    target: str = ""
    y_observed: float = np.nan
    y_predicted: float = np.nan
    y_typical: float = np.nan
    deviation: float = np.nan        # previsto − típico (o que se reparte)
    residual: float = np.nan         # observado − previsto (não explicado)
    rows: pd.DataFrame | None = None
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def attribute_day(cfm: CounterfactualModel, label, n_perm: int = DAY_PERMUTATIONS,
                  check_top: int = 3) -> DayAttribution:
    """Reparte o desvio do dia ``label`` entre os indicadores (Shapley).

    As contribuições somam exatamente ``previsto − típico``. Além da fatia
    justa (Shapley), cada linha traz o contrafactual isolado ("se SÓ este
    indicador estivesse típico, o alvo teria sido...") — os dois números
    divergem quando há interação ou indicadores andando juntos.
    """
    out = DayAttribution(label=label, target=cfm.target)
    if not cfm.ok:
        out.cautions.append(cfm.skipped_reason or "modelo indisponível.")
        return out
    pos_arr = cfm.index.get_indexer(pd.Index([label]))
    if pos_arr[0] < 0:
        out.cautions.append(
            "Este período não entrou na matriz do modelo (faltam valores "
            "para montar as versões com defasagem)."
        )
        return out
    pos = int(pos_arr[0])
    row = cfm.Xv[pos:pos + 1]

    out.y_observed = float(cfm.yv[pos])
    out.y_typical = cfm.y_typical
    out.y_predicted = float(cfm._predict(cfm.model, row)[0])
    out.deviation = out.y_predicted - out.y_typical
    out.residual = out.y_observed - out.y_predicted

    rng = np.random.default_rng(RNG_SEED)
    masks, perms = _permutation_masks(len(cfm.group_order), n_perm,
                                      cfm.col_group, rng)
    phi, mc_se = _shapley(cfm, cfm.model, row, masks, perms)
    phi, mc_se = phi[0], mc_se[0]
    boot_phi = [_shapley(cfm, m, row, masks, perms)[0][0] for m in cfm.boot_models]

    fact, alt = _counterfactual_predictions(cfm, cfm.model, row)
    boot_alt = [
        {g: float(v[0]) for g, v in _counterfactual_predictions(cfm, m, row)[1].items()}
        for m in cfm.boot_models
    ]

    lines = []
    for gi, g in enumerate(cfm.group_order):
        lo, hi = _ci_from_boot(
            float(phi[gi]), [b[gi] for b in boot_phi], cfm.conf,
            extra_se=float(mc_se[gi]),
        )
        alvo_tipico = float(alt[g][0])
        isolado = out.y_predicted - alvo_tipico
        iso_lo, iso_hi = _ci_from_boot(
            isolado, [float(fact[0]) - b[g] for b in boot_alt], cfm.conf
        )
        rep = cfm.group_params[g][0]
        j = cfm.columns.index(rep)
        valor = float(cfm.Xv[pos, j])
        lines.append({
            "indicador": _friendly(g) if g != OTHERS else OTHERS,
            "grupo": g,
            "valor no dia": valor if g != OTHERS else np.nan,
            "típico": float(cfm.baseline[j]) if g != OTHERS else np.nan,
            "percentil no dia": round(
                _percentile_of(cfm.Xv[:, j], valor), 1) if g != OTHERS else np.nan,
            "contribuição": float(phi[gi]),
            "IC inferior": lo,
            "IC superior": hi,
            "% do desvio": (100.0 * phi[gi] / out.deviation
                            if abs(out.deviation) > 1e-12 else np.nan),
            "alvo se típico": alvo_tipico,
            "efeito isolado": isolado,
            "efeito isolado IC-": iso_lo,
            "efeito isolado IC+": iso_hi,
            "períodos parecidos": np.nan,
            "alvo em períodos parecidos": np.nan,
            "efeito pelos vizinhos": np.nan,
            "qualidade do pareamento": "—",
            "métricas reunidas": _metrics_note(cfm.group_params[g]),
        })

    tab = pd.DataFrame(lines)
    tab = tab.reindex(tab["contribuição"].abs().sort_values(
        ascending=False).index).reset_index(drop=True)

    # contraprova empírica só para o topo (custo O(n) por indicador)
    for i in tab.index[:check_top]:
        g = tab.at[i, "grupo"]
        if g == OTHERS:
            continue
        emp = _matched_counterfactual(cfm, pos, g)
        tab.at[i, "períodos parecidos"] = emp["n"]
        tab.at[i, "qualidade do pareamento"] = emp["qualidade"]
        if emp["qualidade"] == "boa":
            tab.at[i, "alvo em períodos parecidos"] = emp["y"]
            tab.at[i, "efeito pelos vizinhos"] = out.y_observed - emp["y"]
    tab.index = tab.index + 1
    out.rows = tab

    out.findings.extend(_day_findings(cfm, out, tab))
    out.cautions.extend(cfm.cautions)
    return out


def _day_findings(cfm: CounterfactualModel, day: DayAttribution,
                  tab: pd.DataFrame) -> list[str]:
    """Frases gerenciais da repartição do dia."""
    alvo = cfm.target
    txt = [
        f"Neste período, '{alvo}' foi {_fmt(day.y_observed)}. Um período "
        f"totalmente típico daria {_fmt(day.y_typical)}; o modelo, olhando os "
        f"indicadores que de fato ocorreram, previa {_fmt(day.y_predicted)}. "
        f"O desvio explicado pelos indicadores é {_fmt(day.deviation)} e se "
        "reparte integralmente entre as linhas abaixo."
    ]
    if abs(day.residual) > 1e-12:
        share = (abs(day.residual) /
                 max(abs(day.y_observed - day.y_typical), 1e-12))
        extra = ""
        if np.isfinite(cfm.resid_sigma) and cfm.resid_sigma > 0:
            extra = (f" Para referência, o erro típico deste modelo fora da "
                     f"amostra é ±{_fmt(cfm.resid_sigma)}.")
        txt.append(
            f"Sobra {_fmt(day.residual)} entre o observado e o previsto — "
            f"{share * 100:.0f}% do afastamento do dia NÃO é explicado pelos "
            "indicadores da planilha (variação comum do processo, medição ou "
            f"fator não medido).{extra}"
        )
    for _, r in tab.head(4).iterrows():
        pct = r["% do desvio"]
        if not np.isfinite(r["contribuição"]) or not np.isfinite(pct) or abs(pct) < 5:
            continue
        sentido = "para CIMA" if r["contribuição"] > 0 else "para BAIXO"
        ic = (f" (IC {int(cfm.conf * 100)}%: {_fmt(r['IC inferior'])} a "
              f"{_fmt(r['IC superior'])})") if np.isfinite(r["IC inferior"]) else ""
        base = (
            f"{r['indicador']}: contribuiu {_fmt(r['contribuição'])} {sentido}"
            f"{ic} — {abs(pct):.0f}% do desvio do dia."
        )
        if r["grupo"] != OTHERS and np.isfinite(r["valor no dia"]):
            base += (
                f" No dia valeu {_fmt(r['valor no dia'])} (percentil "
                f"{r['percentil no dia']:.0f} do histórico; típico "
                f"{_fmt(r['típico'])}); se estivesse no típico, o modelo "
                f"preveria {_fmt(r['alvo se típico'])}."
            )
        if r["qualidade do pareamento"] == "boa":
            base += (
                f" Contraprova sem modelo: em {int(r['períodos parecidos'])} "
                f"períodos parecidos, mas com este indicador normal, o alvo "
                f"ficou em torno de {_fmt(r['alvo em períodos parecidos'])} — "
                f"ou seja, {_fmt(r['efeito pelos vizinhos'])} de diferença "
                "para o que se observou hoje."
            )
        elif r["qualidade do pareamento"] != "—":
            base += (
                " Não há períodos parecidos com este indicador normal no "
                "histórico: aqui o modelo está extrapolando, e o número acima "
                "não tem contraprova empírica."
            )
        txt.append(base)

    checados = tab.head(4)
    div = checados[
        checados["qualidade do pareamento"].eq("boa")
        & (checados["efeito pelos vizinhos"].abs()
           > 1.5 * checados["efeito isolado"].abs())
    ]
    if len(div):
        txt.append(
            "Atenção: para "
            + ", ".join(div["indicador"].astype(str))
            + ", a contraprova pelos vizinhos aponta efeito bem maior que o "
            "do modelo. Isso é típico de dias extremos — o Random Forest "
            "encolhe extremos porque nunca extrapola além do que já viu. "
            "Use o número dos vizinhos como limite superior do efeito."
        )
    return txt


# ------------------------------------------ repartição da variação do período

@dataclass
class VarianceAttribution:
    """Quanto da variação do alvo no período cabe a cada indicador."""

    target: str = ""
    r2_oos: float = np.nan
    n_rows: int = 0
    n_perm: int = 0
    rows: pd.DataFrame | None = None
    unexplained: float = np.nan       # fração da variância não explicada
    findings: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)


def _shares_from_phi(phi: np.ndarray) -> np.ndarray:
    """Fatias da variância a partir das contribuições de Shapley.

    Como ``previsão = típico + Σ φ_g``, vale a identidade
    ``Var(previsão) = Σ_g Cov(φ_g, previsão)`` — ou seja, as covariâncias
    repartem a variância da previsão exatamente, sem sobra nem sobreposição
    (é a versão para variância da propriedade de eficiência do Shapley).
    """
    pred = phi.sum(axis=1)
    var = float(np.var(pred))
    if var <= 0:
        return np.full(phi.shape[1], np.nan)
    cov = np.array([float(np.cov(phi[:, g], pred, ddof=0)[0, 1])
                    for g in range(phi.shape[1])])
    return cov / var


def variance_attribution(cfm: CounterfactualModel, n_rows: int = VAR_ROWS,
                         n_perm: int = VAR_PERMUTATIONS) -> VarianceAttribution:
    """Reparte a variação do alvo no período entre os indicadores.

    As fatias somam 100% do que o modelo consegue explicar (R² fora da
    amostra); o restante aparece explicitamente como "não explicado".
    """
    out = VarianceAttribution(target=cfm.target, r2_oos=cfm.r2_oos,
                              n_perm=n_perm)
    if not cfm.ok:
        out.cautions.append(cfm.skipped_reason or "modelo indisponível.")
        return out

    idx = np.arange(len(cfm.Xv))
    if len(idx) > n_rows:
        idx = np.linspace(0, len(idx) - 1, n_rows).astype(int)
    rows = cfm.Xv[idx]
    out.n_rows = len(idx)

    rng = np.random.default_rng(RNG_SEED)
    masks, perms = _permutation_masks(len(cfm.group_order), n_perm,
                                      cfm.col_group, rng)
    phi, _ = _shapley(cfm, cfm.model, rows, masks, perms)
    shares = _shares_from_phi(phi)
    boot_shares = [
        _shares_from_phi(_shapley(cfm, m, rows, masks, perms)[0])
        for m in cfm.boot_models
    ]

    r2 = cfm.r2_oos if np.isfinite(cfm.r2_oos) else np.nan
    explained = float(np.clip(r2, 0.0, 1.0)) if np.isfinite(r2) else np.nan
    out.unexplained = 1.0 - explained if np.isfinite(explained) else np.nan

    lines = []
    for gi, g in enumerate(cfm.group_order):
        share = float(shares[gi])
        lo, hi = _ci_from_boot(share, [b[gi] for b in boot_shares], cfm.conf)
        scale = explained if np.isfinite(explained) else 1.0
        lines.append({
            "indicador": _friendly(g) if g != OTHERS else OTHERS,
            "grupo": g,
            "% da variação do alvo": 100.0 * share * scale,
            "IC inferior": 100.0 * lo * scale,
            "IC superior": 100.0 * hi * scale,
            "% do que o modelo explica": 100.0 * share,
        })
    tab = pd.DataFrame(lines)
    tab = tab.reindex(tab["% da variação do alvo"].abs().sort_values(
        ascending=False).index).reset_index(drop=True)
    if np.isfinite(out.unexplained):
        tab = pd.concat([tab, pd.DataFrame([{
            "indicador": "não explicado (fatores fora da planilha + ruído)",
            "grupo": "__residuo__",
            "% da variação do alvo": 100.0 * out.unexplained,
            "IC inferior": np.nan, "IC superior": np.nan,
            "% do que o modelo explica": np.nan,
        }])], ignore_index=True)
    tab.index = tab.index + 1
    out.rows = tab

    # ---- frases gerenciais ---------------------------------------------
    if np.isfinite(explained):
        out.findings.append(
            f"Os indicadores da planilha explicam {100 * explained:.0f}% da "
            f"variação de '{cfm.target}' (R² fora da amostra, validação "
            f"temporal); os {100 * out.unexplained:.0f}% restantes vêm de "
            "fatores não medidos, variação comum do processo ou erro de "
            "medição."
        )
    for _, r in tab.head(5).iterrows():
        if r["grupo"] in ("__residuo__",) or not np.isfinite(r["% da variação do alvo"]):
            continue
        ic = (f" (IC {int(cfm.conf * 100)}%: {r['IC inferior']:.0f}% a "
              f"{r['IC superior']:.0f}%)") if np.isfinite(r["IC inferior"]) else ""
        out.findings.append(
            f"{r['indicador']}: responde por {r['% da variação do alvo']:.0f}% "
            f"da variação do alvo{ic}."
        )
    neg = tab[tab["% da variação do alvo"] < -1e-9]
    if len(neg):
        out.cautions.append(
            "Fatias negativas indicam indicadores que, no conjunto, se movem "
            "contra a previsão (efeito supressor) — costuma acontecer quando "
            "dois indicadores muito correlacionados entram juntos."
        )
    out.cautions.extend(cfm.cautions)
    if not np.isfinite(explained) or explained < MIN_R2:
        out.cautions.append(
            "Com R² fora da amostra tão baixo, a repartição diz como o MODELO "
            "se organiza, não como o processo se comporta."
        )
    return out


# ------------------------------------------------------- curva de resposta

@dataclass
class ResponseCurve:
    """Alvo previsto conforme o indicador é forçado a cada nível histórico."""

    target: str = ""
    group: str = ""
    rows: pd.DataFrame | None = None
    typical: float = np.nan
    findings: list[str] = field(default_factory=list)


def response_curve(cfm: CounterfactualModel, group: str, n_points: int = 9,
                   max_eval_rows: int = 400) -> ResponseCurve:
    """Curva contrafactual: alvo previsto se o indicador ficasse em cada nível.

    Percorre percentis de 10 a 90 do próprio indicador (nunca extrapola além
    do que o processo já fez) e devolve a previsão média do alvo com IC —
    responde "qual é a faixa boa?" sem supor relação linear.
    """
    out = ResponseCurve(target=cfm.target, group=group)
    if not cfm.ok or group not in cfm.group_idx:
        return out
    idx = np.arange(len(cfm.Xv))
    if len(idx) > max_eval_rows:
        idx = np.linspace(0, len(idx) - 1, max_eval_rows).astype(int)
    rows = cfm.Xv[idx]
    cols = cfm.group_idx[group]
    rep = cols[0]
    qs = np.linspace(10, 90, n_points)
    levels = np.nanpercentile(cfm.Xv[:, cols].astype(float), qs, axis=0)

    lines = []
    for i, q in enumerate(qs):
        alt = rows.copy()
        alt[:, cols] = levels[i].astype(np.float32)
        point = float(np.mean(cfm._predict(cfm.model, alt)))
        boot = [float(np.mean(cfm._predict(m, alt))) for m in cfm.boot_models]
        lo, hi = _ci_from_boot(point, boot, cfm.conf)
        lines.append({
            "percentil": int(round(q)),
            "valor do indicador": float(levels[i][0]),
            "alvo previsto": point,
            "IC inferior": lo, "IC superior": hi,
        })
    out.rows = pd.DataFrame(lines)
    out.typical = float(cfm.baseline[rep])

    best = out.rows.loc[out.rows["alvo previsto"].idxmax()]
    worst = out.rows.loc[out.rows["alvo previsto"].idxmin()]
    out.findings.append(
        f"Mantendo os demais indicadores como estiveram, o alvo previsto vai "
        f"de {_fmt(worst['alvo previsto'])} (com o indicador em "
        f"{_fmt(worst['valor do indicador'])}, percentil "
        f"{int(worst['percentil'])}) a {_fmt(best['alvo previsto'])} (com o "
        f"indicador em {_fmt(best['valor do indicador'])}, percentil "
        f"{int(best['percentil'])}). Se o alvo for 'quanto maior melhor', a "
        f"faixa boa é a segunda; se for 'quanto menor melhor', a primeira."
    )
    return out
