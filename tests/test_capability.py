"""Testes do motor de capabilidade (Módulo 1): os 3 casos e utilitários."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capability.control_chart import imr_chart, remove_special_causes
from capability.data_prep import load_indicator_table, treat_missing, treat_outliers
from capability.indices import capability_indices
from capability.nonparametric import suggested_limits
from capability.pipeline import run_capability
from capability.transforms import best_normalizing_transform, forward, inverse

RNG = np.random.default_rng(7)


def _serie(valores) -> pd.Series:
    return pd.Series(np.asarray(valores, dtype=float),
                     index=pd.RangeIndex(1, len(valores) + 1, name="ordem"))


# ------------------------------------------------------------------ Caso 1

def test_caso1_normal_cp_cpk_conhecidos():
    # N(µ=10, σ=1), limites 7 e 13 => Pp = 6/(6·1) = 1, Ppk ≈ 1
    x = _serie(RNG.normal(10, 1, 800))
    rep = run_capability(x, "ind", lsl=7, usl=13)
    assert rep.case == 1
    assert abs(rep.indices.pp - 1.0) < 0.1
    assert abs(rep.indices.ppk - 1.0) < 0.12
    assert rep.indices.sided == "bilateral"
    assert "normal" in rep.narrative.lower()


def test_caso1_unilateral_superior():
    x = _serie(RNG.normal(50, 2, 500))
    rep = run_capability(x, "ind", usl=58)
    assert rep.case == 1
    assert np.isnan(rep.indices.cp)  # Cp indefinido com um só limite
    assert abs(rep.indices.ppk - (58 - 50) / (3 * 2)) < 0.2
    assert rep.indices.sided == "superior"


# ------------------------------------------------------------------ Caso 2

def test_caso2_lognormal_transforma_e_inverte():
    x = _serie(np.exp(RNG.normal(2.0, 0.5, 800)))
    rep = run_capability(x, "ind", lsl=1.0, usl=40.0)
    assert rep.case == 2
    assert rep.transform.best is not None
    assert rep.normality_final.is_normal
    # a mediana exibida (T⁻¹(µ_t)) deve ficar perto da mediana real e^2 ≈ 7,39
    assert abs(rep.display_median - np.median(x)) < 1.0
    # a faixa exibida deve conter praticamente todos os dados
    inside = (x >= rep.display_p0135) & (x <= rep.display_p99865)
    assert inside.mean() > 0.99


def test_transformadas_inversao_exata():
    x = np.abs(RNG.normal(5, 2, 200)) + 0.1
    for nome in ("log", "sqrt", "boxcox", "yeojohnson", "johnsonsu"):
        res = best_normalizing_transform(pd.Series(np.exp(RNG.normal(0, 1, 300))))
        # inversão redonda para cada família com params sintéticos simples
    # teste direto de ida-e-volta por família
    casos = {
        "log": {"shift": 0.0},
        "sqrt": {"shift": 0.0},
        "boxcox": {"lmbda": 0.5},
        "yeojohnson": {"lmbda": 1.7},
        "johnsonsu": {"a": 0.3, "b": 1.2, "loc": 1.0, "scale": 2.0},
        "johnsonsb": {"a": 0.1, "b": 1.5, "loc": 0.0, "scale": 100.0},
    }
    for nome, params in casos.items():
        t = forward(nome, params, x)
        volta = inverse(nome, params, t)
        assert np.allclose(volta, x, rtol=1e-8, atol=1e-8), nome


def test_yeojohnson_inversa_com_negativos():
    x = RNG.normal(0, 3, 400)  # contém negativos
    for lmb in (0.0, 0.5, 1.3, 2.0):
        t = forward("yeojohnson", {"lmbda": lmb}, x)
        volta = inverse("yeojohnson", {"lmbda": lmb}, t)
        assert np.allclose(volta, x, rtol=1e-8, atol=1e-8), lmb


# ------------------------------------------------------------------ Caso 3

def test_caso3_bimodal_vai_para_percentis():
    a = RNG.normal(10, 0.5, 400)
    b = RNG.normal(20, 0.5, 400)
    x = _serie(np.concatenate([a, b]))
    rep = run_capability(x, "ind", lsl=8, usl=22)
    assert rep.case == 3
    assert rep.indices.method == "percentil"
    assert rep.empirical["ppm_total"] < 50_000  # quase tudo dentro de 8..22
    assert rep.suggested is not None


def test_sugestao_limites_regra_quartis():
    x = _serie(RNG.normal(100, 5, 1000))
    q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    # quanto maior melhor (só LIE): meta de atuação = Q3
    s_inf = suggested_limits(x, lsl=90.0, usl=None)
    assert abs(s_inf.suggested_lsl - q3) < 1e-9
    assert s_inf.suggested_usl is None
    assert s_inf.practical_lsl < q2  # referência de cobertura na cauda baixa
    # quanto menor melhor (só LSE): espelho da regra => Q1
    s_sup = suggested_limits(x, lsl=None, usl=110.0)
    assert abs(s_sup.suggested_usl - q1) < 1e-9
    assert s_sup.suggested_lsl is None
    # bilateral: faixa da mediana (Q2) ao Q3
    s_bi = suggested_limits(x, lsl=90.0, usl=110.0)
    assert abs(s_bi.suggested_lsl - q2) < 1e-9
    assert abs(s_bi.suggested_usl - q3) < 1e-9
    assert 20.0 < s_bi.coverage_pct < 30.0  # Q2..Q3 cobre ~25% dos dados


# ------------------------------------------------------- carta de controle

def test_carta_iam_sinaliza_outlier_plantado():
    vals = RNG.normal(0, 1, 120)
    vals[60] = 8.0  # causa especial evidente
    x = _serie(vals)
    res = imr_chart(x)
    assert 61 in res.violations  # índice 1-based
    assert 1 in res.violations[61]  # regra R1
    limpo = remove_special_causes(x, [61])
    assert np.isnan(limpo.loc[61])
    res2 = imr_chart(limpo)
    assert res2.sigma_within < res.sigma_within  # variação cai sem o pico


def test_indices_sigma_dentro_vs_global():
    # processo com deslocamento no meio: σ_global ≫ σ_dentro => Cpk > Ppk
    a = RNG.normal(10, 0.5, 150)
    b = RNG.normal(14, 0.5, 150)
    x = _serie(np.concatenate([a, b]))
    imr = imr_chart(x)
    idx = capability_indices(x, lsl=5, usl=19, sigma_within=imr.sigma_within)
    assert idx.sigma_overall > 2 * idx.sigma_within
    assert idx.cpk > idx.ppk


# ---------------------------------------------------------------- tratamento

def test_outliers_iqr_e_zscore():
    vals = np.concatenate([RNG.normal(0, 1, 200), [15.0, -12.0]])
    x = _serie(vals)
    for metodo in ("iqr", "zscore", "zscore_mad"):
        tratado, rel = treat_outliers(x, method=metodo)
        assert rel.n_removed >= 2, metodo
        assert np.isnan(tratado.iloc[-1]) and np.isnan(tratado.iloc[-2]), metodo
    intacto, rel0 = treat_outliers(x, method="nenhum")
    assert rel0.n_removed == 0
    assert intacto.notna().sum() == x.notna().sum()


def test_loader_sem_data_vira_sequencia(tmp_path):
    p = tmp_path / "sem_data.csv"
    p.write_text("vazao;pressao\n10,5;2,1\n11,2;2,3\n9,8;2,0\n12,1;2,4\n")
    df, diag = load_indicator_table(str(p))
    assert not diag.has_dates
    assert df.index.name == "ordem" and list(df.index) == [1, 2, 3, 4]
    assert df["vazao"].iloc[0] == 10.5  # vírgula decimal convertida


def test_loader_com_data_brasileira(tmp_path):
    p = tmp_path / "com_data.csv"
    p.write_text("data;vazao\n01/06/2024;10,5\n02/06/2024;11,2\n03/06/2024;9,8\n")
    df, diag = load_indicator_table(str(p))
    assert diag.has_dates and diag.date_col == "data"
    assert str(df.index[0].date()) == "2024-06-01"  # dd/mm


# ------------------------------------------------------- revisão de limites

def test_revisao_limites_caso1_aderente_e_revisar():
    x = _serie(RNG.normal(10, 1, 800))
    # limites folgados => aderente; recomendação = µ ± 3σ (só referência)
    rep = run_capability(x, "ind", lsl=4, usl=16)
    lr = rep.limit_review
    assert lr.situacao == "aderente"
    assert abs(lr.rec_lsl - (rep.indices.mean - 3 * rep.indices.sigma_overall)) < 1e-9
    assert abs(lr.rec_usl - (rep.indices.mean + 3 * rep.indices.sigma_overall)) < 1e-9
    assert "Revisão dos limites" in rep.narrative
    # limites apertados => revisar
    rep2 = run_capability(x, "ind", lsl=9.5, usl=10.5)
    assert rep2.limit_review.situacao == "revisar"
    assert "recomendados" in rep2.narrative


def test_revisao_limites_unilateral_so_um_lado():
    x = _serie(RNG.normal(50, 2, 500))
    rep = run_capability(x, "ind", usl=58)  # só LSE
    lr = rep.limit_review
    assert lr.rec_lsl is None          # não inventa o lado que não existe
    assert lr.rec_usl is not None


def test_revisao_limites_caso3_usa_regra_dos_quartis():
    a = RNG.normal(10, 0.5, 400)
    b = RNG.normal(20, 0.5, 400)
    x = _serie(np.concatenate([a, b]))
    rep = run_capability(x, "ind", lsl=8, usl=22)
    assert rep.case == 3
    lr = rep.limit_review
    assert abs(lr.rec_lsl - rep.suggested.suggested_lsl) < 1e-9  # Q2
    assert abs(lr.rec_usl - rep.suggested.suggested_usl) < 1e-9  # Q3
    assert "quartis" in lr.metodo


def test_loader_ods(tmp_path):
    # o uploader aceita .ods — a leitura precisa funcionar (engine odfpy)
    p = tmp_path / "dados.ods"
    pd.DataFrame({"data": ["01/06/2024", "02/06/2024", "03/06/2024", "04/06/2024"],
                  "vazao": [10.5, 11.2, 9.8, 12.1]}).to_excel(
        p, engine="odf", index=False)
    df, diag = load_indicator_table(str(p))
    assert diag.has_dates
    assert df["vazao"].notna().sum() == 4


def test_faltantes_interpolacao_e_mediana():
    vals = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, 10.0]
    x = _serie(vals)
    interp, _ = treat_missing(x, method="interpolar", limit=2)
    assert interp.iloc[2] == 3.0            # lacuna curta preenchida
    assert interp.iloc[5:9].isna().all()    # lacuna longa (4) preservada
    med, _ = treat_missing(x, method="mediana")
    assert med.notna().all()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
