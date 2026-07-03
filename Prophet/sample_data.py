"""Gerador de planilhas de exemplo para testar o Criador de Modelos Preditivos.

Executar:  python sample_data.py
Gera dois arquivos CSV em sample_data/:
  - exemplo_vendas.csv   (regressão: prever vendas diárias)
  - exemplo_clientes.csv (classificação: prever cancelamento de clientes)
"""

import os

import numpy as np
import pandas as pd

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def gerar_exemplo_vendas(n_dias: int = 365) -> pd.DataFrame:
    """Vendas diárias em função de preço, propaganda, temperatura e sazonalidade."""
    rng = np.random.default_rng(42)
    datas = pd.date_range("2024-01-01", periods=n_dias, freq="D")

    temperatura = 25 + 8 * np.sin(2 * np.pi * np.arange(n_dias) / 365) + rng.normal(0, 2, n_dias)
    preco = rng.uniform(8, 15, n_dias).round(2)
    propaganda = rng.uniform(0, 1000, n_dias).round(0)
    concorrentes = rng.integers(1, 6, n_dias)
    # coluna irrelevante de propósito (para o score mostrar baixa relevância)
    ruido = rng.normal(50, 10, n_dias).round(2)

    dia_semana = pd.Series(datas).dt.dayofweek.to_numpy()
    fim_de_semana = (dia_semana >= 5).astype(float)

    vendas = (
        500
        - 20 * preco
        + 0.15 * propaganda
        + 6 * temperatura
        + 80 * fim_de_semana
        - 15 * concorrentes
        + rng.normal(0, 25, n_dias)
    ).round(1)

    df = pd.DataFrame(
        {
            "data": datas.strftime("%d/%m/%Y"),
            "preco": preco,
            "propaganda": propaganda,
            "temperatura": temperatura.round(1),
            "concorrentes": concorrentes,
            "indice_ruido": ruido,
            "vendas": vendas,
        }
    )
    # alguns valores faltantes e outliers para exercitar o tratamento automático
    df.loc[rng.choice(n_dias, 12, replace=False), "temperatura"] = np.nan
    df.loc[rng.choice(n_dias, 8, replace=False), "propaganda"] = np.nan
    df.loc[rng.choice(n_dias, 3, replace=False), "preco"] = 99.0  # outliers
    return df


def gerar_exemplo_clientes(n: int = 600) -> pd.DataFrame:
    """Cancelamento (churn) em função de uso, mensalidade, reclamações e plano."""
    rng = np.random.default_rng(7)
    datas = pd.date_range("2023-06-01", periods=n, freq="12h")[:n]

    idade = rng.integers(18, 75, n)
    mensalidade = rng.uniform(30, 250, n).round(2)
    meses_contrato = rng.integers(1, 60, n)
    reclamacoes = rng.poisson(1.2, n)
    uso_mensal_horas = rng.gamma(4, 8, n).round(1)
    plano = rng.choice(["Basico", "Intermediario", "Premium"], n, p=[0.5, 0.3, 0.2])

    logit = (
        -1.0
        + 0.9 * reclamacoes
        + 0.012 * mensalidade
        - 0.05 * meses_contrato
        - 0.03 * uso_mensal_horas
        + np.where(plano == "Premium", -0.8, 0.0)
        + rng.normal(0, 0.7, n)
    )
    cancelou = np.where(1 / (1 + np.exp(-logit)) > 0.5, "Sim", "Nao")

    df = pd.DataFrame(
        {
            "data_cadastro": pd.Series(datas).dt.strftime("%d/%m/%Y"),
            "idade": idade,
            "mensalidade": mensalidade,
            "meses_contrato": meses_contrato,
            "reclamacoes": reclamacoes,
            "uso_mensal_horas": uso_mensal_horas,
            "plano": plano,
            "cancelou": cancelou,
        }
    )
    df.loc[rng.choice(n, 20, replace=False), "uso_mensal_horas"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "mensalidade"] = 2000.0  # outliers
    return df


def main() -> None:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    gerar_exemplo_vendas().to_csv(os.path.join(SAMPLE_DIR, "exemplo_vendas.csv"), index=False)
    gerar_exemplo_clientes().to_csv(os.path.join(SAMPLE_DIR, "exemplo_clientes.csv"), index=False)
    print(f"Arquivos de exemplo gerados em {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
