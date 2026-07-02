"""Geração do relatório HTML autocontido (gráficos embutidos em base64)."""

from __future__ import annotations

import datetime as _dt

import numpy as np
from jinja2 import Template

from . import plots
from .features import derived_features
from .pipeline import AnalysisResult

TEMPLATE = Template(
    """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Análise causal — {{ target }}</title>
<style>
  :root {
    --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --pos: #2a78d6; --neg: #e34948; --good: #0ca30c; --warn: #fab219;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         font-size: 15px; line-height: 1.55; }
  main { max-width: 980px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  h2 { font-size: 19px; margin: 40px 0 12px; border-bottom: 1px solid var(--grid);
       padding-bottom: 6px; }
  h3 { font-size: 16px; margin: 24px 0 8px; }
  .sub { color: var(--ink2); margin: 0 0 24px; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 18px 20px; margin: 14px 0; }
  .verdict-list { margin: 8px 0 0; padding-left: 0; list-style: none; }
  .verdict-list li { padding: 6px 0; border-bottom: 1px dashed var(--grid); }
  .verdict-list li:last-child { border-bottom: none; }
  .badge { display: inline-block; border-radius: 6px; padding: 1px 8px;
           font-size: 12.5px; font-weight: 600; margin-right: 6px; }
  .b-prov { background: #fdeceb; color: #a02c2c; }
  .b-poss { background: #fdf3dc; color: #8a5a00; }
  .b-fraco { background: #eef1f5; color: #52514e; }
  .b-nao { background: #eef7ee; color: #1d6b1d; }
  .dir-pos { color: var(--pos); font-weight: 600; }
  .dir-neg { color: var(--neg); font-weight: 600; }
  .dir-nm { color: var(--ink2); font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px;
          background: var(--surface); }
  th { text-align: left; color: var(--ink2); font-weight: 600;
       border-bottom: 2px solid var(--grid); padding: 7px 10px; }
  td { border-bottom: 1px solid var(--grid); padding: 6px 10px;
       font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: none; }
  .tbl-wrap { overflow-x: auto; border: 1px solid var(--border);
              border-radius: 10px; }
  img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 720px) { .grid2 { grid-template-columns: 1fr; } }
  .note { font-size: 13px; color: var(--ink2); }
  .sig { color: var(--good); font-weight: 700; }
  .nsig { color: var(--muted); }
  details { margin: 10px 0; }
  summary { cursor: pointer; font-weight: 600; color: var(--ink2); }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
  .kpi { background: var(--surface); border: 1px solid var(--border);
         border-radius: 10px; padding: 12px 18px; min-width: 150px; flex: 1; }
  .kpi .v { font-size: 22px; font-weight: 700; }
  .kpi .l { font-size: 12px; color: var(--ink2); }
  footer { margin-top: 48px; color: var(--muted); font-size: 12.5px;
           border-top: 1px solid var(--grid); padding-top: 12px; }
</style>
</head>
<body>
<main>
  <h1>Relatório de análise causal — alvo: <em>{{ target }}</em></h1>
  <p class="sub">Gerado em {{ generated_at }} · {{ n_obs }} observações de
     {{ date_start }} a {{ date_end }} · {{ n_params }} parâmetros analisados ·
     lags 0–{{ max_lag }} · médias móveis {{ windows|join(', ') }} ·
     α = {{ alpha }} (com correção FDR)</p>

  <h2>1. Resumo executivo</h2>
  <div class="card">
    {% if culprits %}
    <p><strong>Parâmetros apontados como prováveis ou possíveis "culpados"
       pelo comportamento de {{ target }}</strong> (por ordem de evidência):</p>
    <ul class="verdict-list">
      {% for c in culprits %}
      <li>
        <span class="badge {{ c.badge }}">{{ c.veredito }}</span>
        <strong>{{ c.parametro }}</strong> — score {{ c.score }};
        relação <span class="{{ c.dir_class }}">{{ c.direcao_label }}</span>{{ c.dir_hint }};
        efeito mais forte em <strong>{{ c.melhor_transformacao }}</strong>;
        confiança estatística <strong>{{ c.confianca|lower }}</strong>
        ({{ c.testes_significativos }} testes significativos após FDR).
      </li>
      {% endfor %}
    </ul>
    {% else %}
    <p><strong>Nenhum parâmetro apresentou evidência estatística consistente de
       influência sobre {{ target }}.</strong> Considere ampliar o período de
       dados, revisar a granularidade temporal ou incluir outros parâmetros.</p>
    {% endif %}
    {% if ml_r2 is not none %}
    <p class="note">Poder preditivo conjunto (Random Forest, validação
       temporal fora da amostra): R² = {{ '%.2f' % ml_r2 }} —
       {{ ml_r2_msg }}</p>
    {% endif %}
  </div>
  <img src="data:image/png;base64,{{ fig_ranking }}" alt="Ranking de scores">

  <h2>2. Tabela de evidências consolidada</h2>
  <p class="note">O score (0–100) pondera sete linhas de evidência:
     associação linear (Pearson), monotônica (Spearman), não-linear
     (correlação de distância e informação mútua), melhor transformação
     temporal (lag/média móvel), causalidade de Granger, contraste de
     percentis (P75 vs P25) e importância no modelo Random Forest.</p>
  <div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>#</th><th>Parâmetro</th><th>Score</th><th>Veredito</th>
      <th>Direção</th><th>Melhor transformação</th><th>Confiança</th>
      <th>Testes sig.</th>
    </tr></thead>
    <tbody>
    {% for row in score_rows %}
      <tr>
        <td>{{ row.rank }}</td><td><strong>{{ row.parametro }}</strong></td>
        <td>{{ row.score }}</td>
        <td><span class="badge {{ row.badge }}">{{ row.veredito }}</span></td>
        <td class="{{ row.dir_class }}">{{ row.direcao_label }}</td>
        <td>{{ row.melhor_transformacao }}</td>
        <td>{{ row.confianca }}</td><td>{{ row.testes_significativos }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>

  <details>
    <summary>Estatísticas detalhadas por parâmetro (correlações, p-valores, Granger, percentis)</summary>
    <div class="tbl-wrap" style="margin-top:10px">
    <table>
      <thead><tr>
        <th>Parâmetro</th><th>Pearson r (p)</th><th>Spearman ρ (p)</th>
        <th>Kendall τ</th><th>dCor</th><th>MI (r-eq)</th>
        <th>Granger p (lag)</th><th>Cliff δ P75×P25 (p)</th>
        <th>Kruskal p</th><th>Import. RF</th>
      </tr></thead>
      <tbody>
      {% for d in detail_rows %}
        <tr>
          <td><strong>{{ d.name }}</strong></td>
          <td>{{ d.pearson }}</td><td>{{ d.spearman }}</td>
          <td>{{ d.kendall }}</td><td>{{ d.dcor }}</td><td>{{ d.mi }}</td>
          <td>{{ d.granger }}</td><td>{{ d.cliff }}</td>
          <td>{{ d.kruskal }}</td><td>{{ d.ml }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    </div>
    <p class="note">Valores marcados com ✓ permanecem significativos após a
       correção de Benjamini-Hochberg (FDR) em sua família de testes.</p>
  </details>

  <h2>3. Estrutura de correlação geral</h2>
  <p class="note">Correlações fortes <em>entre parâmetros</em> indicam
     possível confusão (multicolinearidade): dois parâmetros correlacionados
     entre si podem dividir a "culpa" de um mesmo mecanismo.</p>
  <img src="data:image/png;base64,{{ fig_heatmap }}" alt="Mapa de correlação">

  <h2>4. Análise detalhada dos principais suspeitos</h2>
  {% for p in param_sections %}
  <div class="card">
    <h3>{{ loop.index }}. {{ p.name }} — score {{ p.score }}
        <span class="badge {{ p.badge }}">{{ p.veredito }}</span></h3>
    <p class="note">{{ p.summary }}</p>
    <img src="data:image/png;base64,{{ p.fig_lag }}" alt="Perfil de lags de {{ p.name }}">
    <div class="grid2">
      <img src="data:image/png;base64,{{ p.fig_scatter }}" alt="Dispersão {{ p.name }}">
      <img src="data:image/png;base64,{{ p.fig_box }}" alt="Boxplot por quartis {{ p.name }}">
    </div>
    <img src="data:image/png;base64,{{ p.fig_ts }}" alt="Série temporal {{ p.name }}">
  </div>
  {% endfor %}

  <h2>5. Diagnóstico dos dados</h2>
  <div class="kpi-row">
    <div class="kpi"><div class="v">{{ n_obs }}</div>
      <div class="l">observações utilizadas (de {{ n_rows_raw }} linhas)</div></div>
    <div class="kpi"><div class="v">{{ n_params }}</div>
      <div class="l">parâmetros analisados</div></div>
    <div class="kpi"><div class="v">{{ freq }}</div>
      <div class="l">frequência detectada</div></div>
  </div>
  {% if diag_notes %}
  <ul class="note">
    {% for n in diag_notes %}<li>{{ n }}</li>{% endfor %}
  </ul>
  {% endif %}
  {% if missing_rows %}
  <details>
    <summary>Valores ausentes por coluna</summary>
    <div class="tbl-wrap" style="margin-top:10px">
    <table>
      <thead><tr><th>Coluna</th><th>% ausente</th><th>Pontos interpolados</th></tr></thead>
      <tbody>
      {% for m in missing_rows %}
        <tr><td>{{ m.col }}</td><td>{{ m.pct }}%</td><td>{{ m.interp }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
    </div>
  </details>
  {% endif %}

  <h2>6. Metodologia e limitações</h2>
  <div class="card">
    <p><strong>Como o score é construído.</strong> Cada parâmetro é avaliado por
    sete linhas de evidência complementares: (1) correlação de Pearson mede
    associação <em>linear</em>; (2) Spearman e Kendall medem associação
    <em>monotônica</em> (crescente/decrescente, mesmo que não linear);
    (3) a correlação de distância e a informação mútua detectam dependências
    <em>não-lineares e não-monotônicas</em> (ex.: efeito em forma de U, limiares);
    (4) a varredura de defasagens e médias móveis identifica efeitos com
    <em>atraso</em> ou <em>acumulados</em>; (5) o teste de causalidade de
    Granger verifica se o passado do parâmetro melhora a previsão do alvo além
    do próprio histórico do alvo (precedência temporal, com séries
    estacionarizadas via teste ADF e diferenciação); (6) a análise de
    percentis contrasta o alvo quando o parâmetro está alto (≥P75) vs. baixo
    (≤P25) com Mann-Whitney e delta de Cliff, e entre quartis com
    Kruskal-Wallis; (7) um Random Forest com validação temporal mede a
    importância preditiva por permutação considerando todos os parâmetros e
    transformações simultaneamente (capturando interações).</p>
    <p><strong>Controle de falsos positivos.</strong> Os p-valores de cada
    família de testes passam pela correção de Benjamini-Hochberg (FDR,
    α={{ alpha }}); a "confiança" reportada conta quantos testes sobrevivem à
    correção.</p>
    <p><strong>Limitações importantes.</strong></p>
    <ul>
      <li><strong>Associação e precedência temporal não são prova definitiva de
        causalidade.</strong> Um terceiro fator não medido pode dirigir tanto o
        parâmetro quanto o alvo (confusão), e parâmetros correlacionados entre
        si dividem a evidência.</li>
      <li>O teste de Granger assume relações aproximadamente lineares entre as
        séries estacionarizadas; efeitos puramente não-lineares podem escapar
        dele (por isso ele é apenas uma das sete linhas de evidência).</li>
      <li>A varredura de melhor lag/janela e o menor p-valor de Granger entre
        lags introduzem viés de seleção; a correção FDR mitiga, mas confirme os
        achados com conhecimento do processo.</li>
      <li>Recomenda-se validar os "culpados" apontados com experimentos
        controlados (mudança deliberada do parâmetro) ou análise de
        intervenções históricas.</li>
    </ul>
  </div>

  <footer>Relatório gerado automaticamente pelo pacote <code>causal_analysis</code>
  v{{ version }}. Evidência estatística ≠ prova de causalidade; use em conjunto
  com conhecimento do processo.</footer>
</main>
</body>
</html>
"""
)

_BADGE = {
    "Culpado provável": "b-prov",
    "Culpado possível": "b-poss",
    "Influência fraca": "b-fraco",
    "Sem evidência de influência": "b-nao",
}
_DIR_CLASS = {"positiva": "dir-pos", "negativa": "dir-neg"}


def _fmt_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def _sig_mark(fdr_family: dict, name: str) -> str:
    info = fdr_family.get(name)
    if info is None:
        return ""
    return " ✓" if info["significant"] else ""


def _dir_hint(row) -> str:
    if row["direcao_label"] == "positiva":
        return " (parâmetro maior → alvo maior)"
    if row["direcao_label"] == "negativa":
        return " (parâmetro maior → alvo menor)"
    if row["direcao_label"] == "não-monotônica":
        return " (efeito muda de sentido ao longo da faixa)"
    return ""


def render_report(result: AnalysisResult, output_path: str, top_detail: int = 5) -> str:
    """Monta o HTML final e grava em ``output_path``. Devolve o caminho."""
    scores = result.scores
    assert scores is not None
    df, target = result.df, result.target
    pp = result.per_param

    culprit_mask = scores["veredito"].isin(
        ["Culpado provável", "Culpado possível"]
    )
    culprits = []
    for _, row in scores[culprit_mask].iterrows():
        culprits.append(
            {
                **row.to_dict(),
                "badge": _BADGE[row["veredito"]],
                "dir_class": _DIR_CLASS.get(row["direcao_label"], "dir-nm"),
                "dir_hint": _dir_hint(row),
            }
        )

    score_rows = [
        {
            **row.to_dict(),
            "rank": rank,
            "badge": _BADGE[row["veredito"]],
            "dir_class": _DIR_CLASS.get(row["direcao_label"], "dir-nm"),
        }
        for rank, row in scores.iterrows()
    ]

    detail_rows = []
    for name in scores["parametro"]:
        r = pp[name]
        g = r.get("granger")
        pc = r.get("percentile")
        detail_rows.append(
            {
                "name": name,
                "pearson": f"{r['pearson'][0]:+.2f} ({_fmt_p(r['pearson'][1])})"
                + _sig_mark(result.fdr["Pearson"], name),
                "spearman": f"{r['spearman'][0]:+.2f} ({_fmt_p(r['spearman'][1])})"
                + _sig_mark(result.fdr["Spearman"], name),
                "kendall": f"{r['kendall'][0]:+.2f}"
                if np.isfinite(r["kendall"][0]) else "—",
                "dcor": f"{r['dcor']:.2f}" if np.isfinite(r.get("dcor", np.nan)) else "—",
                "mi": f"{r['mi']:.2f} ({r['mi_r']:.2f})"
                if np.isfinite(r.get("mi", np.nan)) else "—",
                "granger": (
                    f"{_fmt_p(g['p_value'])} (lag {g['best_lag']})"
                    + _sig_mark(result.fdr["Granger"], name)
                ) if g else "—",
                "cliff": (
                    f"{pc['cliffs_delta']:+.2f} ({_fmt_p(pc['p_mannwhitney'])})"
                    + _sig_mark(result.fdr["Mann-Whitney (P75 vs P25)"], name)
                ) if pc else "—",
                "kruskal": (
                    _fmt_p(pc["p_kruskal"])
                    + _sig_mark(result.fdr["Kruskal-Wallis (quartis)"], name)
                ) if pc else "—",
                "ml": f"{r['ml_importance']:.3f}"
                if np.isfinite(r.get("ml_importance", np.nan)) else "—",
            }
        )

    # seções detalhadas: culpados apontados (ou top N por score)
    detail_names = list(scores[culprit_mask]["parametro"].head(top_detail))
    if not detail_names:
        detail_names = list(scores["parametro"].head(min(3, len(scores))))

    param_sections = []
    for name in detail_names:
        r = pp[name]
        row = scores[scores["parametro"] == name].iloc[0]
        fam = derived_features(df[name], result.max_lag, result.windows)
        best_series = fam[r["best_feature"]] if r["best_feature"] in fam else df[name]
        g = r.get("granger")
        pc = r.get("percentile")
        bits = [
            f"Melhor associação com o alvo em <strong>{r['best_label']}</strong> "
            f"(Spearman ρ={r['best_rho']:+.2f}, p={_fmt_p(r['best_p'])})."
        ]
        if g:
            bits.append(
                f"Granger: p={_fmt_p(g['p_value'])} no lag {g['best_lag']}"
                + (f", séries diferenciadas {g['diffs_applied']}×" if g["diffs_applied"] else "")
                + "."
            )
        if pc:
            bits.append(
                f"Com o parâmetro alto (≥P75) a mediana do alvo vai a "
                f"{pc['median_high']:.3g} vs. {pc['median_low']:.3g} quando baixo "
                f"(≤P25) — deslocamento de {pc['median_shift_iqr']:+.2f} IQR "
                f"(delta de Cliff {pc['cliffs_delta']:+.2f})."
            )
        if np.isfinite(r.get("ml_importance", np.nan)):
            bits.append(
                f"Importância no Random Forest: {r['ml_importance']:.3f} "
                f"(transformação mais usada: {r['ml_top_feature']})."
            )
        param_sections.append(
            {
                "name": name,
                "score": row["score"],
                "veredito": row["veredito"],
                "badge": _BADGE[row["veredito"]],
                "summary": " ".join(bits),
                "fig_lag": plots.fig_lag_profile(
                    name, r["lag_profile"], r["rolling_profile"]
                ),
                "fig_scatter": plots.fig_scatter(
                    best_series, df[target], name, target, r["best_label"]
                ),
                "fig_box": plots.fig_quartile_box(df[name], df[target], name, target),
                "fig_ts": plots.fig_timeseries_overlay(
                    df.index, df[target], df[name], target, name
                ),
            }
        )

    ml_r2 = result.ml.r2_oos if result.ml and np.isfinite(result.ml.r2_oos) else None
    if ml_r2 is None:
        ml_r2_msg = ""
    elif ml_r2 >= 0.5:
        ml_r2_msg = ("os parâmetros explicam boa parte da variação do alvo; "
                     "o ranking acima tende a ser informativo.")
    elif ml_r2 >= 0.15:
        ml_r2_msg = ("os parâmetros explicam parte moderada da variação do alvo; "
                     "outros fatores não medidos também atuam.")
    else:
        ml_r2_msg = ("os parâmetros medidos explicam pouco da variação do alvo — "
                     "interprete os 'culpados' com cautela; fatores não medidos dominam.")

    diag = result.diagnostics
    missing_rows = [
        {
            "col": c,
            "pct": diag.missing_pct.get(c, 0.0),
            "interp": diag.interpolated.get(c, 0),
        }
        for c in diag.missing_pct
        if diag.missing_pct.get(c, 0) > 0 or diag.interpolated.get(c, 0) > 0
    ]

    from . import __version__

    html = TEMPLATE.render(
        target=target,
        generated_at=_dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
        n_obs=len(df),
        n_rows_raw=diag.n_rows_raw,
        n_params=len(result.params),
        date_start=diag.date_start,
        date_end=diag.date_end,
        max_lag=result.max_lag,
        windows=result.windows,
        alpha=result.alpha,
        culprits=culprits,
        score_rows=score_rows,
        detail_rows=detail_rows,
        param_sections=param_sections,
        fig_ranking=plots.fig_ranking(scores),
        fig_heatmap=plots.fig_corr_heatmap(df, target),
        ml_r2=ml_r2,
        ml_r2_msg=ml_r2_msg,
        freq=diag.freq or "irregular",
        diag_notes=diag.notes,
        missing_rows=missing_rows,
        version=__version__,
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path
