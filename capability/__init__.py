"""Análise de capabilidade de indicadores de processo (Módulo 1).

Fluxo por indicador, dado um par de limites de especificação (LIE/LSE,
podendo ser unilateral):

1. pré-tratamento opcional (outliers por IQR/Z-score e dados faltantes),
   aplicável a qualquer coluna, inclusive o alvo;
2. carta de controle de Individuais e Amplitude Móvel (I-AM) com detecção
   de causas especiais e opção de removê-las;
3. teste de normalidade (Anderson-Darling). Se normal → Caso 1: índices
   Cp/Cpk/Pp/Ppk clássicos. Se não → tenta transformações normalizadoras
   (log, raiz, Box-Cox, Yeo-Johnson, Johnson) → Caso 2: índices no espaço
   transformado com exibição retro-convertida à escala original. Se nada
   normaliza → Caso 3: análise não-paramétrica por box-plot/percentis com
   sugestão de limites realistas.
"""

__version__ = "0.1.0"

from .pipeline import run_capability  # noqa: F401
from .data_prep import load_indicator_table  # noqa: F401
