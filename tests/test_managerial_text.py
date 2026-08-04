"""Testes do texto do relatório gerencial.

O relatório é lido por quem não é da estatística e, no uso diário por vários
times, é o que a maioria vai ler. Estes testes protegem a legibilidade:
tempo real em vez de "períodos", sem entulho de parênteses e sem repetição.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.managerial_report import (  # noqa: E402
    _cadencia,
    _friendly_name,
    _lag_phrase,
    _lag_short,
    _uc_first,
)

UMA_HORA = pd.Timedelta(hours=1)
UM_MINUTO = pd.Timedelta(minutes=1)
UM_DIA = pd.Timedelta(days=1)


def test_lag_sai_em_tempo_real_e_nunca_em_periodos():
    """"3 períodos" obriga o leitor a converter de cabeça; "3 horas" não."""
    for step, esperado in ((UMA_HORA, "3 horas"), (UM_MINUTO, "3 minutos"),
                           (UM_DIA, "3 dias")):
        frase = _lag_phrase("lag 3", step)
        assert esperado in frase, frase
        assert "período" not in frase, frase


def test_media_movel_tambem_sai_em_tempo_real():
    frase = _lag_phrase("média móvel 12", UMA_HORA)
    assert "12 horas" in frase
    assert "período" not in frase


def test_sem_data_hora_fala_em_medicoes_e_nao_em_periodos():
    """Sem coluna de data/hora não dá para saber a duração — mas "período"
    continua sendo jargão. A unidade honesta é a medição."""
    frase = _lag_phrase("lag 4", None)
    assert "4 medições" in frase
    assert "período" not in frase
    assert "1 medição" in _lag_phrase("lag 1", None)  # singular concorda


def test_texto_sem_parenteses_nem_travessao():
    """Era o que tornava a leitura truncada."""
    for label in ("lag 3", "média móvel 7", "bruto (sem defasagem)"):
        for step in (UMA_HORA, None):
            frase = _lag_phrase(label, step)
            assert "(" not in frase and ")" not in frase, frase
            assert "—" not in frase, frase


def test_forma_curta_serve_para_celula_de_tabela():
    assert _lag_short("lag 3", UMA_HORA) == "3 horas depois"
    assert _lag_short("média móvel 7", UM_DIA) == "acumulado de 1 semana"
    assert _lag_short("bruto (sem defasagem)", UMA_HORA) == "imediato"


def test_metrica_derivada_vira_sujeito_da_frase():
    """"'temp' — valor central na janela (robusto a picos)" trava a leitura."""
    nome = _friendly_name("forno: temp (P90)")
    assert nome == "o teto de temp"
    assert "(" not in nome and "—" not in nome
    # coluna simples continua sendo só o nome
    assert _friendly_name("pressao") == "pressao"


def test_maiuscula_no_inicio_nao_destroi_o_nome_do_indicador():
    """``str.capitalize()`` minusculizava o resto: Temp_Forno virava temp_forno."""
    frase = "quando Temp_Forno sobe, Rendimento tende a cair"
    assert _uc_first(frase) == "Quando Temp_Forno sobe, Rendimento tende a cair"


def test_cadencia_nao_deixa_um_solto():
    """"uma medição a cada 1 dia" soa mal; "uma por dia" não."""
    assert _cadencia("1 dia") == "por dia"
    assert _cadencia("1 hora") == "por hora"
    assert _cadencia("4 horas") == "a cada 4 horas"
