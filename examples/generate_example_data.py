"""Gera um conjunto de dados sintético com estrutura causal conhecida.

Serve para demonstrar e validar o analisador: sabemos de antemão quem são os
"culpados" e por qual mecanismo cada um age.

Mecanismos embutidos (alvo = ``rendimento``):
- ``temp_forno``      efeito NÃO-LINEAR em U invertido, com LAG de 2 dias
                      (ótimo em ~85 °C; desvios derrubam o rendimento);
- ``pressao``         efeito LINEAR positivo imediato;
- ``vazao_agua``      efeito NEGATIVO via MÉDIA MÓVEL de 7 dias (acumulado);
- ``dosagem_quimica`` efeito de LIMIAR: só prejudica acima do percentil 80;
- ``ph``              NÃO causa nada — apenas acompanha a pressão (confusão);
- ``ruido_a/ruido_b`` puro ruído, não devem ser apontados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N = 540


def main(path: str = "examples/dados_exemplo.csv") -> pd.DataFrame:
    datas = pd.date_range("2024-06-01", periods=N, freq="D")
    t = np.arange(N)

    temp_forno = 85 + 6 * np.sin(2 * np.pi * t / 60) + RNG.normal(0, 2.5, N)
    pressao = 5 + 0.8 * np.sin(2 * np.pi * t / 45 + 1) + RNG.normal(0, 0.5, N)
    vazao_agua = 120 + 15 * np.sin(2 * np.pi * t / 90 + 2) + RNG.normal(0, 6, N)
    dosagem_quimica = np.abs(10 + RNG.normal(0, 3, N))
    ph = 7 + 0.35 * (pressao - 5) + RNG.normal(0, 0.25, N)  # segue a pressão
    ruido_a = RNG.normal(50, 10, N)
    ruido_b = RNG.uniform(0, 100, N)

    temp_lag2 = pd.Series(temp_forno).shift(2).bfill().to_numpy()
    vazao_mm7 = pd.Series(vazao_agua).rolling(7, min_periods=1).mean().to_numpy()
    limiar_dosagem = np.quantile(dosagem_quimica, 0.80)

    rendimento = (
        70.0
        - 0.35 * (temp_lag2 - 85.0) ** 2          # U invertido, lag 2
        + 4.0 * (pressao - 5.0)                    # linear imediato
        - 0.45 * (vazao_mm7 - 120.0)               # média móvel 7d, negativo
        - 6.0 * (dosagem_quimica > limiar_dosagem) # limiar (percentil 80)
        + RNG.normal(0, 2.0, N)                    # ruído do processo
    )

    df = pd.DataFrame(
        {
            "data": datas.strftime("%d/%m/%Y"),
            "rendimento": np.round(rendimento, 2),
            "temp_forno": np.round(temp_forno, 2),
            "pressao": np.round(pressao, 3),
            "vazao_agua": np.round(vazao_agua, 1),
            "dosagem_quimica": np.round(dosagem_quimica, 2),
            "ph": np.round(ph, 2),
            "ruido_a": np.round(ruido_a, 2),
            "ruido_b": np.round(ruido_b, 2),
        }
    )
    # lacunas realistas em alguns parâmetros
    for col in ("vazao_agua", "ph"):
        idx = RNG.choice(N, size=int(0.03 * N), replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(path, sep=";", index=False, encoding="utf-8")
    print(f"Gerado: {path} ({len(df)} linhas)")
    return df


if __name__ == "__main__":
    main()
