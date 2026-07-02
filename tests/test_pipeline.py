"""Teste de fumaça: pipeline completo sobre dados sintéticos com causas conhecidas.

Roda com: python -m pytest tests/ -q  (ou simplesmente python tests/test_pipeline.py)
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from causal_analysis import run_analysis
from causal_analysis.report import render_report


def _make_data(path: str, n: int = 300) -> None:
    rng = np.random.default_rng(3)
    t = np.arange(n)
    causa_linear = 10 + 2 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.8, n)
    causa_lag = 5 + np.cos(2 * np.pi * t / 70) + rng.normal(0, 0.4, n)
    ruido = rng.normal(0, 1, n)
    alvo = (
        3.0 * causa_linear
        + 4.0 * pd.Series(causa_lag).shift(3).bfill().to_numpy()
        + rng.normal(0, 1.0, n)
    )
    pd.DataFrame(
        {
            "data": pd.date_range("2025-01-01", periods=n, freq="D"),
            "alvo": alvo,
            "causa_linear": causa_linear,
            "causa_lag": causa_lag,
            "ruido": ruido,
        }
    ).to_csv(path, index=False)


def test_pipeline_identifica_causas_e_gera_relatorio():
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "dados.csv")
        html = os.path.join(tmp, "relatorio.html")
        _make_data(csv)

        res = run_analysis(csv, target="alvo", max_lag=7, verbose=False)
        scores = res.scores
        assert scores is not None and len(scores) == 3

        ranking = list(scores["parametro"])
        # as duas causas verdadeiras devem vir antes do ruído
        assert ranking.index("ruido") == 2, f"ruído não ficou por último: {ranking}"
        score_of = dict(zip(scores["parametro"], scores["score"]))
        assert score_of["causa_linear"] > score_of["ruido"] + 15
        assert score_of["causa_lag"] > score_of["ruido"] + 15

        # o ruído não pode ser apontado como culpado
        verdict_of = dict(zip(scores["parametro"], scores["veredito"]))
        assert not verdict_of["ruido"].startswith("Culpado")

        out = render_report(res, html)
        assert os.path.exists(out) and os.path.getsize(out) > 50_000
        with open(out, encoding="utf-8") as fh:
            content = fh.read()
        assert "Resumo executivo" in content
        assert "causa_linear" in content


if __name__ == "__main__":
    test_pipeline_identifica_causas_e_gera_relatorio()
    print("OK: pipeline identifica as causas e gera o relatório.")
