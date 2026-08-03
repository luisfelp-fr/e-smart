"""Limites de recurso para uso simultâneo — o app é um processo só.

O Streamlit atende TODAS as sessões num único processo Python: cada sessão
roda seu script numa thread, mas CPU é CPU. Sem limite, N pessoas clicando
"Analisar" ao mesmo tempo disputam os mesmos núcleos e a mesma RAM, e o
resultado não é "N vezes mais devagar" — é o processo inteiro travando ou
morrendo por falta de memória.

Estes semáforos convertem "todo mundo cai junto" em "fila ordenada": quem
chega além do limite espera e vê sua posição, em vez de derrubar a máquina.

Dois slots separados porque os recursos escassos são diferentes:

- ``heavy_slot``  — análise do Módulo 2 (CPU: Random Forest, Granger, MI).
- ``render_slot`` — geração de PDF (memória: cada figura dirige um Chromium
  headless pelo kaleido, ~200-400 MB de pico por processo).

Tudo dimensionado por variável de ambiente, para a mesma imagem servir do
plano gratuito (1 vCPU) a uma VM corporativa sem alterar código.
"""

from __future__ import annotations

import contextlib
import os
import threading

# Máximo de análises pesadas simultâneas. O padrão 2 é conservador de
# propósito: numa máquina de 1 vCPU já é o limite útil, e subir isso sem
# medir só troca lentidão por estouro de memória.
MAX_HEAVY_JOBS = max(1, int(os.environ.get("ESMART_MAX_HEAVY_JOBS", "2")))

# Renderização de PDF: 1 por vez. Um relatório de 10 análises são ~30
# chamadas ao Chromium; dois relatórios simultâneos estouram a RAM antes de
# ganhar tempo.
MAX_RENDER_JOBS = max(1, int(os.environ.get("ESMART_MAX_RENDER_JOBS", "1")))

# Tempo máximo na fila antes de desistir e avisar o usuário. Melhor uma
# mensagem clara em 10 min do que uma aba pendurada para sempre.
QUEUE_TIMEOUT_S = float(os.environ.get("ESMART_QUEUE_TIMEOUT_S", "600"))

# Teto do relatório: cada item carrega tabelas e figuras Plotly, e a lista
# vive na sessão até o usuário limpar. Sem limite, uma sessão longa acumula
# memória indefinidamente — e isso multiplica pelo número de pessoas usando
# o app ao mesmo tempo.
MAX_REPORT_ITEMS = max(1, int(os.environ.get("ESMART_MAX_REPORT_ITEMS", "40")))


class Busy(RuntimeError):
    """Fila cheia por tempo demais — a página traduz isso para o usuário."""


class _Gate:
    """Semáforo com contador de espera, para informar a posição na fila."""

    def __init__(self, size: int, name: str) -> None:
        self._sem = threading.BoundedSemaphore(size)
        self._lock = threading.Lock()
        self._waiting = 0
        self.size = size
        self.name = name

    @property
    def waiting(self) -> int:
        """Quantos estão na fila agora (sem contar quem já está rodando)."""
        with self._lock:
            return self._waiting

    @contextlib.contextmanager
    def slot(self, timeout: float | None = None):
        """Ocupa um slot; libera sempre, inclusive se o Streamlit interromper.

        O Streamlit aborta o script em andamento com uma exceção quando o
        usuário mexe em outro widget — por isso a liberação vive num
        ``finally``, e não no fim do bloco.
        """
        wait = QUEUE_TIMEOUT_S if timeout is None else timeout
        with self._lock:
            self._waiting += 1
        acquired = False
        try:
            acquired = self._sem.acquire(timeout=wait)
            if not acquired:
                raise Busy(
                    f"A fila de {self.name} está cheia há mais de "
                    f"{int(wait / 60)} min. Tente novamente em alguns minutos."
                )
            with self._lock:
                self._waiting -= 1
            yield
        finally:
            if acquired:
                self._sem.release()
            else:
                with self._lock:
                    self._waiting -= 1


_HEAVY = _Gate(MAX_HEAVY_JOBS, "análises")
_RENDER = _Gate(MAX_RENDER_JOBS, "geração de PDF")


def heavy_slot(timeout: float | None = None):
    """Slot para a análise do Módulo 2 (limitado por CPU)."""
    return _HEAVY.slot(timeout)


def render_slot(timeout: float | None = None):
    """Slot para renderizar gráficos em PNG via Chromium (limitado por RAM)."""
    return _RENDER.slot(timeout)


def queue_note(gate: str = "heavy") -> str | None:
    """Frase sobre a fila, ou None quando não há espera — para a interface."""
    g = _HEAVY if gate == "heavy" else _RENDER
    n = g.waiting
    if not n:
        return None
    if n == 1:
        return "há 1 pessoa na frente na fila"
    return f"há {n} pessoas na frente na fila"


def limits_summary() -> str:
    """Resumo dos limites vigentes — exibido no rodapé da barra lateral."""
    return (
        f"{MAX_HEAVY_JOBS} análise(s) e {MAX_RENDER_JOBS} PDF(s) em paralelo "
        f"(ajuste com ESMART_MAX_HEAVY_JOBS / ESMART_MAX_RENDER_JOBS)"
    )
