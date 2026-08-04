"""Testes do texto da aba Diagnóstico do dia.

Mesmas regras do relatório gerencial: atraso em tempo real, texto corrido,
sem parêntese guardando jargão. É o texto que os times leem todo dia.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.day_diagnosis import _friendly, diagnose_day  # noqa: E402
from causal_analysis.phrasing import measured_on, measured_short  # noqa: E402
from causal_analysis.pipeline import analyze_dataframe  # noqa: E402

UMA_HORA = pd.Timedelta(hours=1)
UM_DIA = pd.Timedelta(days=1)


def _base_horaria(n: int = 600, seed: int = 4) -> pd.DataFrame:
    """Alvo que responde a Temp_Forno com 3 horas de atraso."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    x1 = pd.Series(rng.normal(0, 1, n).cumsum(), index=idx)
    x2 = pd.Series(rng.normal(0, 1, n), index=idx)
    alvo = -2.0 * x1.shift(3) + 0.5 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {"Temp_Forno": x1, "Vazao": x2, "Rendimento": alvo}
    ).dropna()


def test_versao_avaliada_sai_em_tempo_real():
    """"lag 3" com dados horários tem de virar "3 horas antes"."""
    assert measured_short("lag 3", UMA_HORA) == "3 horas antes"
    assert measured_short("média móvel 7", UM_DIA) == "acumulado de 1 semana"
    assert measured_short("bruto (sem defasagem)", UMA_HORA) == "valor do dia"
    # sem data/hora a unidade honesta é a medição, nunca "período"
    assert measured_short("lag 3", None) == "3 medições antes"


def test_frase_da_versao_avaliada_encaixa_no_texto():
    frase = measured_on("lag 3", UMA_HORA)
    assert frase == ", medido 3 horas antes"
    assert "(" not in frase and "período" not in frase
    # valor bruto não merece frase nenhuma
    assert measured_on("bruto (sem defasagem)", UMA_HORA) == ""


def test_nome_do_indicador_vira_sujeito_sem_travessao():
    nome = _friendly("forno: temp (máximo)")
    assert nome == "o pico de temp"
    assert "—" not in nome and "'" not in nome


def test_achados_sem_parenteses_travessao_ou_seta():
    """Era o que deixava a leitura truncada."""
    df = _base_horaria()
    resultado = analyze_dataframe(df, "Rendimento", verbose=False)
    diag = diagnose_day(resultado, df.index[400])

    assert diag.findings
    for frase in diag.findings + diag.cautions:
        assert "→" not in frase, frase
        assert "—" not in frase, frase
        assert "período" not in frase, frase
        assert "(" not in frase and ")" not in frase, frase


def test_achado_do_alvo_explica_o_percentil_em_portugues():
    """"percentil 12 do histórico" não diz nada a quem não é da estatística."""
    df = _base_horaria()
    resultado = analyze_dataframe(df, "Rendimento", verbose=False)
    diag = diagnose_day(resultado, df.index[400])

    primeiro = diag.findings[0]
    assert "Neste dia Rendimento valeu" in primeiro
    assert "dos dias do histórico ficaram abaixo" in primeiro
    assert "percentil" not in primeiro


def test_tabela_traz_a_versao_avaliada_em_tempo_real():
    df = _base_horaria()
    resultado = analyze_dataframe(df, "Rendimento", verbose=False)
    diag = diagnose_day(resultado, df.index[400])

    versoes = diag.rows["versão avaliada"].astype(str)
    assert not versoes.str.contains("lag|média móvel|período").any(), \
        versoes.tolist()
    # coluna interna não deve vazar rótulo cru para quem lê
    assert "transformacao" in diag.rows.columns  # usada pelas frases
