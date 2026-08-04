"""Análise automática de relações causa-efeito em séries temporais tabulares.

O pacote recebe uma tabela (1ª coluna com datas, demais colunas com parâmetros),
o nome da coluna-alvo, e produz um relatório HTML que ranqueia os parâmetros
"culpados" pelo comportamento do alvo, combinando:

- correlações lineares e monotônicas (Pearson, Spearman, Kendall);
- medidas de dependência não-linear (correlação de distância, informação mútua);
- análise de defasagens (lags) e médias móveis (melhor transformação temporal);
- causalidade de Granger com verificação de estacionariedade (ADF);
- análise por percentis (efeito do parâmetro alto vs. baixo sobre o alvo);
- importância por permutação em Random Forest (validação temporal);
- correção de múltiplos testes (FDR Benjamini-Hochberg).

Sobre o ranking, o módulo ``counterfactual`` responde *quanto* (e não só
*quem*): o efeito de cada indicador na unidade do alvo pelo contrafactual
("o que teria acontecido se X estivesse no valor típico"), a repartição da
variação do alvo pelo valor de Shapley e intervalos de confiança por
bootstrap por blocos.
"""

__version__ = "0.2.0"

from .pipeline import run_analysis  # noqa: F401
from .data_loader import load_table  # noqa: F401
from .report import render_report  # noqa: F401
from .counterfactual import (  # noqa: F401
    attribute_day,
    fit_counterfactual_model,
    response_curve,
    scenario_effects,
    variance_attribution,
)
