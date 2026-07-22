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


def test_efeito_direto_vs_indireto_em_cadeia():
    # cadeia A -> B -> alvo: B deve ser "direto"; A, "indireto (via B)"
    from causal_analysis.pipeline import analyze_dataframe

    rng = np.random.default_rng(11)
    n = 500
    a = rng.normal(0, 1, n).cumsum()
    b = 0.9 * a + rng.normal(0, 0.3, n)
    alvo = 0.9 * b + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"A": a, "B": b, "alvo": alvo,
                       "ruido": rng.normal(0, 1, n)},
                      index=pd.RangeIndex(1, n + 1, name="ordem"))
    res = analyze_dataframe(df, "alvo", max_lag=5, verbose=False)
    assert "efeito" in res.scores.columns
    efeitos = dict(zip(res.scores["parametro"], res.scores["efeito"]))
    # B (elo direto) não pode ser rotulado indireto
    assert not str(efeitos["B"]).startswith("indireto")
    # A (age só através de B) deve ter indício de mediação via B
    assert str(efeitos["A"]).startswith("indireto")
    assert "B" in str(efeitos["A"])


def test_diagnostico_do_dia_aponta_contribuinte_extremo():
    # 'driver' move o alvo; no último dia ele está extremo -> deve liderar o
    # diagnóstico do dia com empurrão para CIMA; 'ruido' não deve contribuir
    from causal_analysis.day_diagnosis import diagnose_day
    from causal_analysis.pipeline import analyze_dataframe

    rng = np.random.default_rng(7)
    n = 120
    driver = rng.normal(50, 5, n)
    driver[-1] = driver.max() + 25          # dia extremo (P100)
    alvo = 2.0 * driver + rng.normal(0, 3, n)
    df = pd.DataFrame(
        {"driver": driver, "ruido": rng.normal(0, 1, n), "alvo": alvo},
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )
    res = analyze_dataframe(df, "alvo", max_lag=5, verbose=False)
    diag = diagnose_day(res, df.index[-1])

    assert diag.n_history == n
    assert diag.target_pct > 95                    # alvo do dia ficou altíssimo
    top = diag.rows.iloc[0]
    assert "driver" in top["indicador"]
    assert top["percentil no dia"] >= 99
    assert top["empurrão esperado"] == "alvo para CIMA"
    assert top["score do dia"] > 30
    # frases citam o contribuinte e a cautela está presente
    assert any("driver" in f for f in diag.findings)
    assert any("não prova causal" in c for c in diag.cautions)
    # dia típico: nada deve ser apontado com empurrão
    meio = df.index[60]
    diag2 = diagnose_day(res, meio)
    linha_driver = diag2.rows[diag2.rows["indicador"].str.contains("driver")]
    assert linha_driver.iloc[0]["score do dia"] <= diag.rows.iloc[0]["score do dia"]


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
