"""Testes da correção de "procurar até achar" e da trava de R² do modelo.

Estes testes protegem decisões de VALIDADE, não de desempenho: se alguém
reverter sem querer, o app volta a declarar evidência forte onde ela é fraca —
e ninguém percebe, porque o número continua saindo.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.scoring import ML_MIN_R2, score_parameters  # noqa: E402
from causal_analysis.stats_tests import (  # noqa: E402
    effective_n_tests,
    scan_transforms,
    sidak_adjust,
)


def test_series_identicas_contam_como_um_teste():
    """18 cópias da mesma série não são 18 testes independentes."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 300)
    frame = pd.DataFrame({f"c{i}": base for i in range(18)})
    assert effective_n_tests(frame) < 2.0


def test_series_independentes_contam_quase_todas():
    """Séries sem relação entre si devem contar perto do total."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({f"c{i}": rng.normal(0, 1, 400) for i in range(10)})
    assert effective_n_tests(frame) > 7.0


def test_numero_efetivo_fica_entre_um_e_o_total():
    """Nunca menos que 1 teste, nunca mais que o número de colunas."""
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 200)
    frame = pd.DataFrame({
        "a": base,
        "b": base + rng.normal(0, 0.05, 200),   # quase igual a 'a'
        "c": rng.normal(0, 1, 200),             # independente
    })
    n_eff = effective_n_tests(frame)
    assert 1.0 <= n_eff <= 3.0


def test_sidak_penaliza_o_vencedor_de_varias_tentativas():
    """O p do vencedor precisa piorar quando houve busca."""
    assert sidak_adjust(0.01, 1) == pytest.approx(0.01)  # sem busca, sem custo
    assert sidak_adjust(0.01, 10) > 0.01                 # com busca, custa
    assert sidak_adjust(0.01, 10) < 1.0
    # monotônico no número de tentativas
    assert sidak_adjust(0.01, 5) < sidak_adjust(0.01, 20)


def test_sidak_preserva_p_valores_minusculos():
    """p ~1e-43 não pode virar zero por cancelamento de ponto flutuante.

    A forma literal 1-(1-p)^n perde toda a precisão aqui: (1-p) arredonda
    para 1.0 e o resultado zera — justamente na evidência mais forte.
    """
    p = 7.45e-43
    ajustado = sidak_adjust(p, 3.0)
    assert ajustado > 0.0, "p minúsculo foi zerado pelo ajuste"
    assert ajustado == pytest.approx(3.0 * p, rel=1e-6)  # Šidák ~ Bonferroni


def test_scan_transforms_expoe_p_ajustado_pior_que_o_bruto():
    """A varredura escolhe a melhor de ~18 transformações: o p tem de pagar."""
    rng = np.random.default_rng(3)
    n = 300
    x = pd.Series(rng.normal(0, 1, n).cumsum(), name="x")
    y = pd.Series(rng.normal(0, 1, n).cumsum(), name="y")

    scan = scan_transforms(x, y, max_lag=14, windows=[3, 7, 14])
    assert np.isfinite(scan.best_p_adj)
    assert scan.best_p_adj >= scan.best_p
    assert 1.0 <= scan.n_eff <= 18.0


def _per_param(ml_importance: float) -> dict:
    return {
        "x": {
            "pearson": (0.5, 0.001), "spearman": (0.5, 0.001),
            "dcor": 0.4, "mi_r": 0.3, "best_rho": 0.5,
            "best_p": 0.001, "best_p_adj": 0.01, "n_eff_transforms": 6.0,
            "granger": {"p_value": 0.01, "p_value_adj": 0.05},
            "percentile": {"cliffs_delta": 0.4},
            "ml_importance": ml_importance,
        }
    }


def test_modelo_sem_poder_preditivo_nao_pontua():
    """R² fora da amostra ruim: a importância do modelo não entra no score."""
    fdr = {"Pearson": {"x": {"significant": True}}}

    bom = score_parameters(_per_param(1.0), fdr, 0.05, ml_r2=0.5)
    ruim = score_parameters(_per_param(1.0), fdr, 0.05, ml_r2=-0.3)

    assert bom.loc[1, "comp_ml"] > 0.0
    assert ruim.loc[1, "comp_ml"] == 0.0
    assert ruim.loc[1, "score"] < bom.loc[1, "score"]


def test_r2_ausente_tambem_nao_pontua():
    """Sem R² conhecido não há como creditar o modelo."""
    fdr = {"Pearson": {"x": {"significant": True}}}
    for r2 in (None, float("nan"), ML_MIN_R2 - 0.01):
        out = score_parameters(_per_param(1.0), fdr, 0.05, ml_r2=r2)
        assert out.loc[1, "comp_ml"] == 0.0, f"pontuou com R²={r2}"
