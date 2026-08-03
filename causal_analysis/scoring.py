"""Agregação da evidência em um score de culpabilidade (0-100) por parâmetro.

O score pondera sete linhas de evidência; a confiança vem da contagem de
testes que permanecem significativos após correção FDR. Score e confiança
juntos definem o veredito ("culpado provável", "possível", ...).

LIMITES CONHECIDOS DESTA AGREGAÇÃO — leia antes de tratar o score como medida:

- **As linhas não são independentes.** Pearson, Spearman, o contraste de
  percentis e a melhor transformação medem, em boa parte, a MESMA associação
  monotônica. Uma associação real é contada várias vezes, o que infla tanto o
  score quanto a contagem de significâncias.
- **Os pesos e os limiares são heurísticos**, escolhidos por julgamento, não
  calibrados contra dados com causa conhecida. Servem para ORDENAR
  candidatos; o valor absoluto não tem interpretação probabilística.
- **Há seleção dentro de cada indicador**: a melhor de ~18 transformações e o
  menor p entre os lags do Granger. Ver ``stats_tests.selection_adjusted_p``.

Por isso o resultado é uma FILA DE INVESTIGAÇÃO priorizada, não uma medida de
causalidade nem uma probabilidade.
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


# R² fora da amostra abaixo disto = o modelo não prevê nada de útil, e a
# importância por permutação dele é ruído. Zero já significa "pior que chutar
# a média"; a margem evita creditar modelos praticamente nulos.
ML_MIN_R2 = 0.05


def score_parameters(
    per_param: dict[str, dict],
    fdr: dict[str, dict[str, dict]],
    alpha: float,
    ml_r2: float | None = None,
) -> pd.DataFrame:
    """Monta a tabela final de scores a partir dos resultados por parâmetro.

    ``per_param``: saída do pipeline com todos os testes por parâmetro.
    ``fdr``: {família de teste: {parâmetro: {p, p_adj, significant}}}.
    """
    # A importância por permutação só significa alguma coisa se a floresta
    # PREVÊ o alvo fora da amostra. Com R² <= 0 o modelo é pior que chutar a
    # média, e permutar colunas de um modelo que não aprendeu nada mede ruído.
    # Pior: como a importância é normalizada pelo máximo do grupo, sem esta
    # trava ALGUÉM sempre recebia o componente ML cheio — mesmo quando não
    # havia sinal nenhum a distribuir.
    ml_usable = ml_r2 is not None and np.isfinite(ml_r2) and ml_r2 >= ML_MIN_R2
    max_ml = max(
        (_nz(r.get("ml_importance")) for r in per_param.values()), default=0.0
    ) if ml_usable else 0.0
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
                # p bruto e p corrigido pela escolha da melhor transformação:
                # a distância entre os dois mostra QUANTO da evidência vinha de
                # ter procurado. Quem só olha o bruto acha que sabe mais do que
                # sabe; quem só olha o corrigido não vê que houve busca.
                "p_melhor_transf": r.get("best_p"),
                "p_melhor_transf_ajustado": r.get("best_p_adj"),
                "transformacoes_efetivas": round(
                    _nz(r.get("n_eff_transforms")) or 1.0, 1),
                **{f"comp_{k}": round(v, 3) for k, v in comp.items()},
            }
        )
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.index = out.index + 1  # ranking 1-based
    return out
