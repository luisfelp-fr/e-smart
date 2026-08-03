"""Testes do limitador de trabalho simultâneo (shared/limits.py)."""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.limits import Busy, _Gate  # noqa: E402


def test_gate_limita_execucoes_simultaneas():
    """Com 2 slots, nunca há 3 threads dentro ao mesmo tempo."""
    gate = _Gate(2, "teste")
    dentro = 0
    pico = 0
    trava = threading.Lock()

    def trabalho():
        nonlocal dentro, pico
        with gate.slot():
            with trava:
                dentro += 1
                pico = max(pico, dentro)
            time.sleep(0.05)
            with trava:
                dentro -= 1

    threads = [threading.Thread(target=trabalho) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pico <= 2, f"passaram {pico} threads simultâneas com 2 slots"
    assert dentro == 0


def test_slot_liberado_quando_o_bloco_falha():
    """O Streamlit aborta o script com exceção — o slot não pode vazar."""
    gate = _Gate(1, "teste")

    for _ in range(3):
        with pytest.raises(RuntimeError, match="falha simulada"):
            with gate.slot():
                raise RuntimeError("falha simulada")

    # se o slot tivesse vazado, este acquire ficaria preso
    with gate.slot(timeout=0.5):
        pass


def test_fila_cheia_vira_busy_e_nao_trava_para_sempre():
    """Esperar demais dá mensagem clara em vez de aba pendurada."""
    gate = _Gate(1, "teste")
    liberar = threading.Event()

    def ocupa():
        with gate.slot():
            liberar.wait(timeout=5)

    t = threading.Thread(target=ocupa)
    t.start()
    time.sleep(0.1)  # garante que o slot único já foi tomado

    with pytest.raises(Busy):
        with gate.slot(timeout=0.2):
            pass

    liberar.set()
    t.join()
    assert gate.waiting == 0


def test_contador_de_fila_volta_a_zero_apos_desistencia():
    """Quem desiste por timeout precisa sair do contador da fila."""
    gate = _Gate(1, "teste")
    liberar = threading.Event()

    def ocupa():
        with gate.slot():
            liberar.wait(timeout=5)

    t = threading.Thread(target=ocupa)
    t.start()
    time.sleep(0.1)

    with pytest.raises(Busy):
        with gate.slot(timeout=0.2):
            pass

    liberar.set()
    t.join()
    assert gate.waiting == 0, "contador de fila vazou após timeout"
