"""Carga e pré-tratamento dos dados para a análise de capabilidade.

Diferenças em relação ao loader da análise causal: a coluna de datas é
opcional (sem data, o índice vira a sequência 1..N na ordem do arquivo),
uma única coluna de indicador é aceita, e valores ausentes do alvo NÃO são
descartados automaticamente — o tratamento (outliers e faltantes) é uma
escolha explícita do usuário, aplicável a qualquer coluna, inclusive o alvo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shared.parsing import datelike_columns, parse_dates, read_raw, to_numeric

SEQ_INDEX_NAME = "ordem"


@dataclass
class PrepDiagnostics:
    """Resumo do que foi feito com os dados na carga."""

    n_rows_raw: int = 0
    n_rows_used: int = 0
    date_col: str | None = None
    has_dates: bool = False
    date_start: str = ""
    date_end: str = ""
    missing_pct: dict[str, float] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class OutlierReport:
    """Registro do tratamento de outliers de uma coluna."""

    method: str = "nenhum"  # 'nenhum' | 'iqr' | 'zscore' | 'zscore_mad'
    threshold: float = np.nan
    n_removed: int = 0
    removed_index: list = field(default_factory=list)
    lower_bound: float = np.nan
    upper_bound: float = np.nan


def load_indicator_table(
    path: str,
    date_col: str | None = None,
    sep: str | None = None,
    sheet: int | str = 0,
) -> tuple[pd.DataFrame, PrepDiagnostics]:
    """Carrega a tabela preservando a ordem das linhas do arquivo.

    Tenta interpretar ``date_col`` (ou a primeira coluna) como datas; se
    menos da metade parsear, mantém todas as colunas e usa uma sequência
    numérica 1..N como índice ("ordem"). A ordem original só é alterada
    (ordenação por data) quando existe uma coluna de datas real — a carta
    I-AM depende da ordem de produção.
    """
    diag = PrepDiagnostics()
    df = read_raw(path, sep=sep, sheet=sheet)
    diag.n_rows_raw = len(df)
    if df.shape[1] < 1 or len(df) == 0:
        raise ValueError("A tabela está vazia ou não tem colunas.")

    candidate = date_col or df.columns[0]
    dates = None
    if candidate in df.columns:
        col = df[candidate]
        # números ("10,5", "42") também "parseiam" como datas — uma coluna só
        # é aceita como data se parsear MELHOR como data do que como número
        # (colunas já em dtype datetime são aceitas direto)
        if pd.api.types.is_datetime64_any_dtype(col):
            parsed, dayfirst = col, True
        else:
            parsed, dayfirst = parse_dates(col)
            n_numeric = to_numeric(col).notna().sum()
            if n_numeric >= parsed.notna().sum():
                parsed = pd.Series(pd.NaT, index=col.index)
        if parsed.notna().sum() >= len(df) * 0.5:
            dates = pd.Series(parsed)
            diag.date_col = str(candidate)
            diag.has_dates = True
            if not dayfirst:
                diag.notes.append("Datas interpretadas no formato mês/dia (mm/dd).")
        elif date_col is not None:
            raise ValueError(
                f"Não foi possível interpretar a coluna '{date_col}' como datas."
            )

    if dates is not None:
        bad = int(dates.isna().sum())
        if bad:
            diag.notes.append(f"{bad} linha(s) com data inválida foram removidas.")
        df = df.drop(columns=[candidate])
        df.index = dates
        df = df[df.index.notna()].sort_index()
        dup = int(df.index.duplicated().sum())
        if dup:
            # converte ANTES de agregar: mean sobre strings ("10,5") quebraria
            df = df.apply(to_numeric)
            df = df.groupby(level=0).mean(numeric_only=False)
            diag.notes.append(f"{dup} data(s) duplicada(s) foram agregadas pela média.")
    else:
        # sem datas: sequência crescente 1..N, preservando a ordem do arquivo
        df = df.copy()
        df.index = pd.RangeIndex(1, len(df) + 1, name=SEQ_INDEX_NAME)
        diag.notes.append(
            "Nenhuma coluna de data/tempo identificada; usando sequência "
            "numérica crescente iniciando em 1 como eixo temporal."
        )
        # dica: o detector só olha a 1ª coluna — avisa se há datas em outra
        hints = datelike_columns(df, exclude=(candidate,) if candidate else ())
        if hints:
            diag.notes.append(
                f"A coluna '{hints[0]}' parece conter datas — para usá-la "
                "como eixo temporal, deixe-a como PRIMEIRA coluna da planilha."
            )

    # conversão numérica coluna a coluna (decimal com vírgula suportado)
    for col in list(df.columns):
        df[col] = to_numeric(df[col])
        valid = df[col].notna().mean()
        diag.missing_pct[str(col)] = round(100 * (1 - valid), 1)
        if df[col].notna().sum() == 0:
            df = df.drop(columns=[col])
            diag.dropped_columns.append(str(col))

    if diag.dropped_columns:
        diag.notes.append(
            "Colunas sem nenhum valor numérico foram descartadas: "
            + ", ".join(diag.dropped_columns)
        )
    if df.shape[1] == 0:
        raise ValueError("Nenhuma coluna numérica encontrada na tabela.")

    # colunas constantes não são descartadas (o usuário pode querer vê-las),
    # mas são sinalizadas: sem variação não há Cp/Cpk definido
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            diag.constant_columns.append(str(col))
    if diag.constant_columns:
        diag.notes.append(
            "Colunas com valor constante (capabilidade indefinida): "
            + ", ".join(diag.constant_columns)
        )

    diag.n_rows_used = len(df)
    if diag.has_dates and len(df):
        diag.date_start = str(df.index.min())
        diag.date_end = str(df.index.max())
    return df, diag


# ------------------------------------------------------------- outliers

def treat_outliers(
    s: pd.Series, method: str = "nenhum", k: float | None = None
) -> tuple[pd.Series, OutlierReport]:
    """Remove outliers segundo o método escolhido, devolvendo série e registro.

    - ``iqr``: fora de [Q1 - k·IQR, Q3 + k·IQR] (k padrão 1.5);
    - ``zscore``: |(x - média)/desvio| > k (k padrão 3.0);
    - ``zscore_mad``: Z-score modificado de Iglewicz-Hoaglin,
      |0.6745·(x - mediana)/MAD| > k (k padrão 3.5) — robusto, pois mediana
      e MAD não são corrompidos pelos próprios outliers;
    - ``nenhum``: devolve a série intacta.

    Valores removidos viram NaN (o índice é preservado para os gráficos).
    """
    rep = OutlierReport(method=method)
    x = s.astype(float)
    valid = x.dropna()
    if method == "nenhum" or len(valid) < 4:
        return x, rep

    if method == "iqr":
        rep.threshold = 1.5 if k is None else float(k)
        q1, q3 = valid.quantile([0.25, 0.75])
        iqr = q3 - q1
        rep.lower_bound = float(q1 - rep.threshold * iqr)
        rep.upper_bound = float(q3 + rep.threshold * iqr)
        mask = (x < rep.lower_bound) | (x > rep.upper_bound)
    elif method == "zscore":
        rep.threshold = 3.0 if k is None else float(k)
        mu, sd = valid.mean(), valid.std(ddof=1)
        if not sd or not np.isfinite(sd):
            return x, rep
        z = (x - mu) / sd
        rep.lower_bound = float(mu - rep.threshold * sd)
        rep.upper_bound = float(mu + rep.threshold * sd)
        mask = z.abs() > rep.threshold
    elif method == "zscore_mad":
        rep.threshold = 3.5 if k is None else float(k)
        med = valid.median()
        mad = (valid - med).abs().median()
        if not mad or not np.isfinite(mad):
            return x, rep
        mz = 0.6745 * (x - med) / mad
        half = rep.threshold * mad / 0.6745
        rep.lower_bound = float(med - half)
        rep.upper_bound = float(med + half)
        mask = mz.abs() > rep.threshold
    else:
        raise ValueError(f"Método de outlier desconhecido: {method}")

    mask = mask.fillna(False)
    rep.n_removed = int(mask.sum())
    rep.removed_index = list(x.index[mask])
    out = x.copy()
    out[mask] = np.nan
    return out, rep


# ------------------------------------------------------------ faltantes

def treat_missing(
    s: pd.Series, method: str = "nenhum", limit: int = 3
) -> tuple[pd.Series, list[str]]:
    """Trata valores ausentes; devolve (série tratada, notas).

    - ``nenhum``: mantém NaN (as análises ignoram esses pontos);
    - ``descartar``: idem (NaN são ignorados nos cálculos; explícito na UI);
    - ``interpolar``: interpolação linear de lacunas de até ``limit`` pontos;
    - ``mediana`` / ``media``: preenche com o valor central da coluna.

    Pontos preenchidos artificialmente encolhem a variância — o chamador
    deve exibi-los como marcados e considerar excluí-los dos momentos.
    """
    notes: list[str] = []
    x = s.astype(float)
    n_missing = int(x.isna().sum())
    if n_missing == 0 or method in ("nenhum", "descartar"):
        return x, notes

    before = n_missing
    if method == "interpolar":
        # só preenche lacunas INTEIRAS de até ``limit`` pontos; lacunas mais
        # longas ficam intactas (interpolate(limit=n) preencheria o começo)
        na = x.isna()
        gap_id = (na != na.shift()).cumsum()
        gap_size = na.groupby(gap_id).transform("sum")
        fillable = na & (gap_size <= limit)
        interp = x.interpolate(method="linear", limit_area="inside")
        x = x.where(~fillable, interp)
    elif method == "mediana":
        x = x.fillna(x.median())
    elif method == "media":
        x = x.fillna(x.mean())
    else:
        raise ValueError(f"Método de faltantes desconhecido: {method}")

    filled = before - int(x.isna().sum())
    if filled:
        notes.append(
            f"{filled} valor(es) ausente(s) preenchido(s) por '{method}'. "
            "Preenchimentos artificiais reduzem a variância aparente — "
            "interprete os índices com cautela."
        )
    return x, notes
