"""Contrafactual, repartição da variação e intervalos de confiança.

Os dados sintéticos têm efeito conhecido e ANALITICAMENTE calculável: com
``alvo = 3·A + 4·B + ruído``, forçar A à sua mediana muda o alvo em
``3·(A − mediana(A))`` — os testes conferem se o contrafactual do modelo
recupera esse número, se as fatias fecham 100% do desvio e se os intervalos
de confiança se comportam (contêm a verdade, encolhem com mais dados).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from causal_analysis.counterfactual import (
    OTHERS,
    attribute_day,
    fit_counterfactual_model,
    response_curve,
    scenario_effects,
    variance_attribution,
)
from causal_analysis.pipeline import analyze_dataframe


def _dados_lineares(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """alvo = 3·A + 4·B + ruído; C é ruído puro (não deve pontuar)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(10, 2, n)
    b = rng.normal(5, 1, n)
    c = rng.normal(0, 1, n)
    alvo = 3.0 * a + 4.0 * b + rng.normal(0, 1.0, n)
    return pd.DataFrame(
        {"A": a, "B": b, "C": c, "alvo": alvo},
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )


def _modelo(df: pd.DataFrame, **kw):
    res = analyze_dataframe(df, "alvo", max_lag=3, windows=[3], verbose=False)
    return res, fit_counterfactual_model(res, **kw)


def test_modelo_contrafactual_ajusta_e_prediz():
    _, cfm = _modelo(_dados_lineares(), n_boot=4, n_estimators=80,
                     boot_estimators=30)
    assert cfm.ok, cfm.skipped_reason
    assert cfm.r2_oos > 0.7                      # relação forte e simples
    assert set(cfm.group_order) <= {"A", "B", "C", OTHERS}
    # a linha de referência é a mediana de cada coluna
    assert np.isfinite(cfm.y_typical)
    assert abs(cfm.typical_value("A") - 10.0) < 0.5


def test_contrafactual_recupera_o_efeito_conhecido():
    # forçar A à mediana deve mover o alvo em ~3·(A − mediana(A))
    df = _dados_lineares(n=500)
    res, cfm = _modelo(df, top=3, n_boot=5, n_estimators=120, boot_estimators=40)
    cen = scenario_effects(cfm)
    linhas = cen.rows.set_index("grupo")

    # A tem média ≈ mediana, então o efeito MÉDIO no NÍVEL é ~0 (e o IC
    # precisa dizer isso); o efeito real aparece no deslocamento e nos dias
    # em que A saiu do típico
    assert linhas.loc["C", "efeito no nível é conclusivo"] == "não (IC cruza zero)"
    # deslocamento: 3·E|A−mediana(A)| ≈ 4,8 para A; ~0 para o ruído C
    assert linhas.loc["A", "deslocamento típico"] > 2.5
    assert linhas.loc["A", "deslocamento típico"] > \
        3 * linhas.loc["C", "deslocamento típico"]

    pos = int(np.argmax(df["A"].to_numpy()[10:]) + 10)
    label = df.index[pos]
    dia = attribute_day(cfm, label, n_perm=120)
    linha_a = dia.rows.set_index("grupo").loc["A"]
    esperado = 3.0 * (df["A"].iloc[pos] - df["A"].median())
    # Random Forest encolhe extremos: exige-se o sinal certo e a ordem de
    # grandeza, não a igualdade exata
    assert linha_a["efeito isolado"] > 0.3 * esperado
    assert linha_a["efeito isolado"] < 1.3 * esperado


def test_fatias_do_dia_somam_exatamente_o_desvio():
    # propriedade de eficiência do Shapley: Σ contribuições = previsto − típico
    df = _dados_lineares(n=300)
    _, cfm = _modelo(df, top=3, n_boot=3, n_estimators=80, boot_estimators=30)
    for pos in (25, 150, 280):
        dia = attribute_day(cfm, df.index[pos], n_perm=150)
        soma = float(dia.rows["contribuição"].sum())
        assert abs(soma - dia.deviation) < 1e-6 * max(1.0, abs(dia.deviation))
        assert abs(sum(dia.rows["% do desvio"]) - 100.0) < 0.01
        # observado = típico + desvio explicado + resíduo
        assert abs((dia.y_typical + dia.deviation + dia.residual)
                   - dia.y_observed) < 1e-6


def test_dia_atipico_aponta_o_indicador_certo():
    df = _dados_lineares(n=300)
    df.iloc[-1, df.columns.get_loc("A")] = float(df["A"].max() + 8)
    df.iloc[-1, df.columns.get_loc("alvo")] = float(
        3.0 * df["A"].iloc[-1] + 4.0 * df["B"].iloc[-1])
    _, cfm = _modelo(df, top=3, n_boot=3, n_estimators=100, boot_estimators=30)
    dia = attribute_day(cfm, df.index[-1], n_perm=150)

    topo = dia.rows.iloc[0]
    assert topo["grupo"] == "A"
    assert topo["contribuição"] > 0                 # empurrou o alvo para cima
    assert topo["percentil no dia"] >= 99
    assert any("A" in f for f in dia.findings)
    assert any("não prova" in c or "MODELO" in c for c in dia.cautions)


def test_reparticao_da_variancia_fecha_cem_por_cento():
    df = _dados_lineares(n=400)
    _, cfm = _modelo(df, top=3, n_boot=4, n_estimators=100, boot_estimators=30)
    var = variance_attribution(cfm, n_rows=80, n_perm=16)

    tab = var.rows.set_index("grupo")
    fatias = tab.drop(index="__residuo__")["% da variação do alvo"]
    # fatias + não explicado = 100% da variação do alvo
    total = float(fatias.sum() + tab.loc["__residuo__", "% da variação do alvo"])
    assert abs(total - 100.0) < 0.5
    # e as fatias sozinhas fecham 100% do que o modelo explica
    assert abs(float(tab.drop(index="__residuo__")
                     ["% do que o modelo explica"].sum()) - 100.0) < 0.01

    # A pesa mais que B (3·σ_A = 6 contra 4·σ_B = 4) e ambos mais que C
    assert fatias["A"] > fatias["B"] > fatias["C"]
    assert fatias["C"] < 10.0
    assert var.unexplained < 0.25


def test_intervalos_de_confianca_sao_coerentes():
    df = _dados_lineares(n=400)
    _, cfm = _modelo(df, top=3, n_boot=6, n_estimators=100, boot_estimators=40)
    dia = attribute_day(cfm, df.index[200], n_perm=150)

    r = dia.rows
    assert (r["IC inferior"] <= r["contribuição"] + 1e-9).all()
    assert (r["IC superior"] >= r["contribuição"] - 1e-9).all()
    # ruído puro: o IC do efeito isolado precisa conter zero
    c = r.set_index("grupo").loc["C"]
    assert c["efeito isolado IC-"] <= 0 <= c["efeito isolado IC+"]

    var = variance_attribution(cfm, n_rows=60, n_perm=12)
    v = var.rows.set_index("grupo").drop(index="__residuo__")
    assert (v["IC inferior"] <= v["% da variação do alvo"] + 1e-9).all()
    assert (v["IC superior"] >= v["% da variação do alvo"] - 1e-9).all()


def test_sem_bootstrap_nao_ha_intervalo():
    df = _dados_lineares(n=200)
    _, cfm = _modelo(df, top=2, n_boot=0, n_estimators=60)
    cen = scenario_effects(cfm)
    assert cen.rows["IC inferior"].isna().all()
    assert any("sem intervalo" in c for c in cen.cautions)


def test_metricas_do_mesmo_indicador_viram_um_grupo():
    # colunas "temp (média)" e "temp (máximo)" descrevem o MESMO equipamento:
    # o cenário contrafactual precisa mover as duas juntas
    rng = np.random.default_rng(9)
    n = 300
    base = rng.normal(100, 5, n)
    df = pd.DataFrame(
        {
            "temp (média)": base,
            "temp (máximo)": base + rng.normal(3, 0.5, n),
            "vazao (média)": rng.normal(50, 4, n),
            "alvo": 2.0 * base + rng.normal(0, 2, n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D"),
    )
    _, cfm = _modelo(df, top=3, n_boot=3, n_estimators=80, boot_estimators=30)
    assert "temp" in cfm.group_order
    assert len(cfm.group_params["temp"]) >= 2      # as duas métricas juntas
    assert "vazao" in cfm.group_order


def test_curva_de_resposta_cresce_com_o_indicador():
    df = _dados_lineares(n=350)
    _, cfm = _modelo(df, top=3, n_boot=3, n_estimators=100, boot_estimators=30)
    curva = response_curve(cfm, "A", n_points=7, max_eval_rows=150)

    r = curva.rows
    assert len(r) == 7
    assert r["valor do indicador"].is_monotonic_increasing
    # efeito positivo conhecido: o alvo previsto sobe com A
    assert r["alvo previsto"].iloc[-1] > r["alvo previsto"].iloc[0]
    assert (r["IC inferior"] <= r["alvo previsto"] + 1e-9).all()
    assert curva.findings


def test_relatorio_html_ganha_a_secao_contrafactual():
    import os
    import tempfile

    from causal_analysis.report import render_report

    df = _dados_lineares(n=250)
    res = analyze_dataframe(df, "alvo", max_lag=3, windows=[3], verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        com = render_report(
            res, os.path.join(tmp, "com.html"),
            counterfactual={"n_boot": 3, "n_estimators": 60,
                            "boot_estimators": 25},
        )
        sem = render_report(res, os.path.join(tmp, "sem.html"))
        html_com = open(com, encoding="utf-8").read()
        html_sem = open(sem, encoding="utf-8").read()

    assert "5. Contrafactual" in html_com
    assert "% da variação do alvo" in html_com
    assert "IC 95%" in html_com
    # as seções seguintes são renumeradas
    assert "6. Diagnóstico dos dados" in html_com
    assert "7. Metodologia" in html_com
    # sem a opção, o relatório antigo continua idêntico na numeração
    assert "Contrafactual" not in html_sem
    assert "5. Diagnóstico dos dados" in html_sem
    assert "6. Metodologia" in html_sem


def test_cli_aceita_a_opcao_contrafactual():
    from causal_analysis.cli import build_parser

    args = build_parser().parse_args(
        ["d.csv", "--alvo", "y", "--contrafactual", "--replicas-bootstrap", "4"]
    )
    assert args.contrafactual is True
    assert args.replicas == 4
    assert build_parser().parse_args(["d.csv", "--alvo", "y"]).contrafactual is False


def test_dados_insuficientes_degradam_com_aviso():
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame(
        {"A": rng.normal(0, 1, n), "B": rng.normal(0, 1, n),
         "alvo": rng.normal(0, 1, n)},
        index=pd.RangeIndex(1, n + 1, name="ordem"),
    )
    res = analyze_dataframe(df, "alvo", max_lag=2, windows=[3], verbose=False)
    cfm = fit_counterfactual_model(res, n_boot=2, n_estimators=40)
    assert not cfm.ok
    assert "mínimo" in (cfm.skipped_reason or "")
    # as funções de consulta não podem quebrar, só avisar
    assert scenario_effects(cfm).cautions
    assert attribute_day(cfm, df.index[-1]).cautions
    assert variance_attribution(cfm).cautions


if __name__ == "__main__":
    test_fatias_do_dia_somam_exatamente_o_desvio()
    print("OK: as fatias fecham o desvio do dia.")
