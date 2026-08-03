"""Quanto cada etapa do Módulo 2 custa, em segundos e em % do total.

Saber o tempo total de uma análise diz se a máquina aguenta. Saber **qual
etapa** consome esse tempo diz qual botão girar: se a varredura de lags
domina, baixar o "lag máximo" na interface resolve; se é o Random Forest,
o que pesa é o número de indicadores.

    python benchmarks/profile_steps.py            # grade padrão
    python benchmarks/profile_steps.py 5000 20    # grade específica
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from bench_module2 import make_frame  # noqa: E402


def main() -> None:
    n_rows = int(sys.argv[1]) if len(sys.argv) > 2 else 2_000
    n_params = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    from causal_analysis.modeling import ml_importance
    from causal_analysis.stats_tests import (
        correlations,
        distance_correlation,
        granger_causality,
        ljung_box,
        mutual_information,
        percentile_effect,
        scan_transforms,
    )

    df = make_frame(n_rows, n_params)
    y = df["alvo"]
    params = [c for c in df.columns if c != "alvo"]
    max_lag, windows = 14, [3, 7, 14]

    def timed(nome, fn):
        t0 = time.perf_counter()
        fn()
        return nome, time.perf_counter() - t0

    etapas = [
        timed("Correlações (Pearson/Spearman/Kendall)",
              lambda: [correlations(df[p], y) for p in params]),
        timed("Correlação de distância",
              lambda: [distance_correlation(df[p], y) for p in params]),
        timed("Informação mútua",
              lambda: [mutual_information(df[p], y) for p in params]),
        timed("Varredura de lags e médias móveis",
              lambda: [scan_transforms(df[p], y, max_lag, windows) for p in params]),
        timed("Ljung-Box",
              lambda: [ljung_box(df[p], lags=max_lag) for p in params]),
        timed("Granger (com ADF)",
              lambda: [granger_causality(df[p], y, max_lag) for p in params]),
        timed("Contraste de percentis",
              lambda: [percentile_effect(df[p], y) for p in params]),
        timed("Random Forest + importância por permutação",
              lambda: ml_importance(df, "alvo", max_lag, windows)),
    ]

    total = sum(s for _, s in etapas)
    print(f"Grade: {n_rows} linhas x {n_params} indicadores  |  "
          f"CPUs: {os.cpu_count()}  |  "
          f"ESMART_RF_JOBS={os.environ.get('ESMART_RF_JOBS', '(padrão)')}")
    print(f"{'etapa':<46}{'seg':>9}{'%':>7}")
    for nome, seg in sorted(etapas, key=lambda e: -e[1]):
        print(f"{nome:<46}{seg:>9.1f}{100 * seg / total:>7.1f}")
    print(f"{'TOTAL':<46}{total:>9.1f}{100.0:>7.1f}")


if __name__ == "__main__":
    main()
