"""Agregação da evidência em um score de culpabilidade (0-100) por parâmetro.

O score pondera sete linhas de evidência independentes; a confiança vem da
contagem de testes que permanecem significativos após correção FDR. Score e
confiança juntos definem o veredito ("culpado provável", "possível", ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# pesos das linhas de evidência (somam 1.0)
WEIGHTS = {
    "linear": 0.10,        # |Pearson|
    "monotonic": 0.15,     # |Spearman|
    "nonlinear": 0.20,     # max(dCor, MI-equivalente)
    "temporal": 0.15,      # |Spearman| na melhor transformação (lag/média móvel)
    "granger": 0.15,       # -log10(p) do Granger, saturado em p=1e-3
    "percentile": 0.10,    # |delta de Cliff| alto-vs-baixo
    "ml": 0.15,            # importância por permutação (normalizada no grupo)
}

VERDICTS = {
    "provavel": "Culpado provável",
    "possivel": "Culpado possível",
    "fraco": "Influência fraca",
    "improvavel": "Sem evidência de influência",
}


def _nz(v: float | None) -> float:
    """NaN/None -> 0 (evidência ausente não pontua)."""
    return 0.0 if v is None or not np.isfinite(v) else float(v)


def score_parameters(per_param: dict[str, dict], fdr: dict[str, dict[str, dict]], alpha: float) -> pd.DataFrame:
    """Monta a tabela final de scores a partir dos resultados por parâmetro.

    ``per_param``: saída do pipeline com todos os testes por parâmetro.
    ``fdr``: {família de teste: {parâmetro: {p, p_adj, significant}}}.
    """
    max_ml = max(
        (_nz(r.get("ml_importance")) for r in per_param.values()), default=0.0
    )
    rows = []
    for name, r in per_param.items():
        comp = {
            "linear": min(1.0, abs(_nz(r["pearson"][0]))),
            "monotonic": min(1.0, abs(_nz(r["spearman"][0]))),
            "nonlinear": min(1.0, max(_nz(r.get("dcor")), _nz(r.get("mi_r")))),
            "temporal": min(1.0, abs(_nz(r.get("best_rho")))),
            "granger": min(1.0, -np.log10(max(r["granger"]["p_value"], 1e-12)) / 3.0)
            if r.get("granger")
            else 0.0,
            "percentile": min(1.0, abs(_nz(r["percentile"]["cliffs_delta"])))
            if r.get("percentile")
            else 0.0,
            "ml": (_nz(r.get("ml_importance")) / max_ml) if max_ml > 0 else 0.0,
        }
        score = 100.0 * sum(WEIGHTS[k] * v for k, v in comp.items())

        n_sig = sum(
            1
            for family in fdr.values()
            if family.get(name, {}).get("significant", False)
        )
        n_tested = sum(1 for family in fdr.values() if name in family)

        if n_sig >= 4:
            confidence = "Alta"
        elif n_sig >= 2:
            confidence = "Média"
        elif n_sig >= 1:
            confidence = "Baixa"
        else:
            confidence = "Nenhuma"

        if score >= 45 and n_sig >= 3:
            verdict = VERDICTS["provavel"]
        elif score >= 30 and n_sig >= 2:
            verdict = VERDICTS["possivel"]
        elif score >= 20 and n_sig >= 1:
            verdict = VERDICTS["fraco"]
        else:
            verdict = VERDICTS["improvavel"]

        # direção do efeito na melhor transformação temporal
        rho = _nz(r.get("best_rho"))
        nonlin_dominant = comp["nonlinear"] >= 0.25 and abs(rho) < 0.15
        if nonlin_dominant:
            direction, dir_label = 0, "não-monotônica"
        elif rho > 0.05:
            direction, dir_label = 1, "positiva"
        elif rho < -0.05:
            direction, dir_label = -1, "negativa"
        else:
            direction, dir_label = 0, "indefinida"

        rows.append(
            {
                "parametro": name,
                "score": round(score, 1),
                "confianca": confidence,
                "veredito": verdict,
                "direcao": direction,
                "direcao_label": dir_label,
                "melhor_transformacao": r.get("best_label", "—"),
                "testes_significativos": f"{n_sig}/{n_tested}",
                "n_sig": n_sig,
                **{f"comp_{k}": round(v, 3) for k, v in comp.items()},
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.index = out.index + 1  # ranking 1-based
    return out
