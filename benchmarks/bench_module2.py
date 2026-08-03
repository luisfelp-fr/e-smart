"""Mede tempo e pico de memória de uma análise do Módulo 2.

Por que isto existe no repositório: dimensionar a máquina para uso por várias
pessoas exige saber **quanto custa uma análise**, e esse custo depende do
tamanho da grade (linhas × indicadores) e do hardware. Rodar este script na
máquina candidata dá o número da casa, em vez de estimativa.

    python benchmarks/bench_module2.py                 # grades padrão
    python benchmarks/bench_module2.py 10000 40        # uma grade específica

Cada grade roda num **subprocesso isolado**, porque o pico de memória é medido
por ``ru_maxrss``, que é marca d'água do processo inteiro e nunca desce.

Como ler o resultado, para dimensionar:

    núcleos ≈ (runs_por_dia × segundos_por_run) / (segundos_de_expediente × 0,6)
    RAM     ≈ base + (sessões_ativas × ~resultado) + (runs_simultâneos × pico)
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# (linhas, indicadores) — o custo cresce MUITO mais que linearmente com o
# número de indicadores, então as grades começam pequenas de propósito. Para
# medir uma grade maior, passe-a por argumento.
GRIDS = [(2_000, 10), (5_000, 20)]


def make_frame(n_rows: int, n_params: int, seed: int = 42) -> pd.DataFrame:
    """Série sintética com estrutura temporal real (não ruído branco).

    Ruído branco puro deixaria o Granger e o Ljung-Box saírem cedo demais e
    subestimaria o custo. Aqui cada indicador é um AR(1) e o alvo depende de
    alguns deles com defasagem — o caminho caro do pipeline é exercitado.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    cols = {}
    for i in range(n_params):
        noise = rng.normal(0, 1, n_rows)
        serie = np.empty(n_rows)
        serie[0] = noise[0]
        phi = 0.6 + 0.3 * rng.random()  # AR(1) com persistência variável
        for t in range(1, n_rows):
            serie[t] = phi * serie[t - 1] + noise[t]
        cols[f"indicador_{i:02d}"] = serie

    frame = pd.DataFrame(cols, index=idx)
    # alvo: efeito com lag de três indicadores + não-linearidade + ruído
    alvo = (
        1.2 * frame.iloc[:, 0].shift(3)
        - 0.8 * frame.iloc[:, 1].shift(1)
        + 0.5 * frame.iloc[:, 2].rolling(7, min_periods=2).mean()
        + 0.4 * frame.iloc[:, 3] ** 2
        + rng.normal(0, 0.5, n_rows)
    )
    frame["alvo"] = alvo
    return frame.dropna()


def run_one(n_rows: int, n_params: int) -> None:
    """Roda uma grade e imprime a linha de resultado (usado no subprocesso)."""
    from causal_analysis.pipeline import analyze_dataframe

    df = make_frame(n_rows, n_params)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    t0 = time.perf_counter()
    result = analyze_dataframe(df, "alvo", verbose=False)
    elapsed = time.perf_counter() - t0

    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    n_feat = result.ml.n_features if result.ml else 0
    print(
        f"{n_rows:>7} {n_params:>7} {n_feat:>9} {elapsed:>9.1f} "
        f"{rss_peak:>9.0f} {rss_peak - rss_before:>9.0f}"
    )


def main() -> None:
    if len(sys.argv) == 3:  # modo subprocesso: uma grade só
        run_one(int(sys.argv[1]), int(sys.argv[2]))
        return

    grids = GRIDS
    if len(sys.argv) > 1:
        print("uso: bench_module2.py [linhas indicadores]", file=sys.stderr)
        raise SystemExit(2)

    print(f"CPUs: {os.cpu_count()}  |  ESMART_RF_JOBS="
          f"{os.environ.get('ESMART_RF_JOBS', '(padrão)')}")
    print(f"{'linhas':>7} {'indics':>7} {'features':>9} {'seg':>9} "
          f"{'RSS_MB':>9} {'delta_MB':>9}")
    for n_rows, n_params in grids:
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), str(n_rows), str(n_params)],
            check=True,
        )


if __name__ == "__main__":
    main()
