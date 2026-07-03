"""Carga e tratamento automático de dados.

Fase A do pré-processamento (global e determinística):
  - parse e ordenação por data, remoção de linhas sem data/alvo válidos
  - criação de variáveis derivadas da data (mês, dia da semana, tendência...)
  - ajuste de outliers por IQR (clipping, sem remover linhas)
  - remoção de colunas inúteis (constantes, quase todas vazias, alta cardinalidade)

A imputação de faltantes e a padronização ficam DENTRO do Pipeline sklearn
(fase B, em training.py), ajustadas apenas no conjunto de treino para evitar
vazamento de dados.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Fatores do método IQR: valores fora de [Q1 - 1.5*IQR, Q3 + 1.5*IQR] são atípicos
IQR_FACTOR = 1.5
MAX_MISSING_RATIO = 0.60
MAX_CATEGORIES = 50

DATE_FEATURES = ["mes", "dia_semana", "dia_do_mes", "trimestre", "indice_tendencia"]

DATE_FEATURE_BOUNDS = {
    "mes": (1, 12),
    "dia_semana": (0, 6),
    "dia_do_mes": (1, 31),
    "trimestre": (1, 4),
}

DATE_FEATURE_LABELS = {
    "mes": "Mês (1-12)",
    "dia_semana": "Dia da semana (0=segunda ... 6=domingo)",
    "dia_do_mes": "Dia do mês (1-31)",
    "trimestre": "Trimestre (1-4)",
    "indice_tendencia": "Índice de tendência (0 = 1ª data da base)",
}


class DadosInvalidosError(ValueError):
    """Erro amigável exibido ao usuário quando os dados não podem ser processados."""


@dataclass
class TreatmentReport:
    """Relatório de tudo que o tratamento automático fez, para exibir ao usuário."""

    linhas_originais: int = 0
    linhas_finais: int = 0
    datas_invalidas_removidas: int = 0
    alvo_ausente_removidas: int = 0
    outliers_ajustados: dict = field(default_factory=dict)  # coluna -> nº de valores ajustados
    outliers_alvo_detectados: int = 0
    ausentes_por_coluna: dict = field(default_factory=dict)  # coluna -> nº de faltantes
    colunas_removidas: list = field(default_factory=list)  # (coluna, motivo)
    features_de_data_criadas: list = field(default_factory=list)

    def to_markdown_pt(self) -> str:
        linhas = [
            f"- **Linhas na planilha original:** {self.linhas_originais}",
            f"- **Linhas após o tratamento:** {self.linhas_finais}",
        ]
        if self.datas_invalidas_removidas:
            linhas.append(
                f"- ⚠️ **{self.datas_invalidas_removidas} linha(s) removida(s)** por data inválida/vazia"
            )
        if self.alvo_ausente_removidas:
            linhas.append(
                f"- ⚠️ **{self.alvo_ausente_removidas} linha(s) removida(s)** por variável alvo vazia "
                "(o alvo nunca é preenchido automaticamente)"
            )
        total_outliers = sum(self.outliers_ajustados.values())
        if total_outliers:
            detalhe = ", ".join(f"`{c}`: {n}" for c, n in self.outliers_ajustados.items() if n)
            linhas.append(
                f"- 🔧 **{total_outliers} valor(es) atípico(s) ajustado(s)** pelo método IQR "
                f"(trazidos para o limite aceitável, sem excluir linhas) — {detalhe}"
            )
        else:
            linhas.append("- ✅ Nenhum outlier detectado nas variáveis numéricas")
        if self.outliers_alvo_detectados:
            linhas.append(
                f"- ℹ️ **{self.outliers_alvo_detectados} valor(es) atípico(s) detectado(s) na variável alvo** "
                "— não foram alterados para não distorcer o que o modelo deve prever"
            )
        faltantes = {c: n for c, n in self.ausentes_por_coluna.items() if n}
        if faltantes:
            detalhe = ", ".join(f"`{c}`: {n}" for c, n in faltantes.items())
            linhas.append(
                f"- 🩹 **Valores ausentes detectados** ({detalhe}) — serão preenchidos automaticamente "
                "durante o treinamento (mediana para números, valor mais frequente para categorias), "
                "usando apenas os dados de treino para evitar vazamento"
            )
        if self.colunas_removidas:
            for col, motivo in self.colunas_removidas:
                linhas.append(f"- 🗑️ Coluna **`{col}` removida**: {motivo}")
        if self.features_de_data_criadas:
            linhas.append(
                "- 📅 **Variáveis criadas a partir da data:** "
                + ", ".join(f"`{c}`" for c in self.features_de_data_criadas)
                + " — entram como candidatas na seleção de variáveis"
            )
        return "\n".join(linhas)


def load_file(file_name: str, content: bytes) -> pd.DataFrame:
    """Lê .csv/.xlsx/.xls a partir dos bytes enviados no upload."""
    nome = file_name.lower()
    try:
        if nome.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            # sep=None detecta ; ou , (padrão brasileiro costuma ser ;)
            try:
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="latin-1")
    except Exception as exc:  # noqa: BLE001 - qualquer falha vira erro amigável
        raise DadosInvalidosError(
            f"Não foi possível ler o arquivo '{file_name}'. "
            f"Verifique se é um CSV ou Excel válido. Detalhe técnico: {exc}"
        ) from exc

    if df.empty or df.shape[1] < 2:
        raise DadosInvalidosError(
            "A planilha precisa ter pelo menos 2 colunas: a primeira com datas "
            "e ao menos uma coluna de parâmetros/alvo."
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_dates(series: pd.Series) -> pd.Series:
    """Converte a coluna de data tentando primeiro o formato brasileiro (dia/mês/ano)."""
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.5:
        alt = pd.to_datetime(series, errors="coerce", dayfirst=False)
        if alt.notna().sum() > parsed.notna().sum():
            parsed = alt
    return parsed


def detect_problem_type(series: pd.Series) -> str:
    """Retorna 'classificacao' ou 'regressao' a partir da variável alvo."""
    s = series.dropna()
    if s.empty:
        return "regressao"
    if pd.api.types.is_bool_dtype(s):
        return "classificacao"
    if pd.api.types.is_numeric_dtype(s):
        valores = s.unique()
        if len(valores) <= 10 and np.allclose(valores, np.round(valores.astype(float))):
            return "classificacao"
        return "regressao"
    # texto/categoria (inclui o dtype 'str' do pandas >= 3)
    return "classificacao"


def treat_data(raw_df: pd.DataFrame, date_col: str, target_col: str, problem_type: str):
    """Executa a fase A do tratamento. Retorna (treated_df, feature_meta, report)."""
    report = TreatmentReport(linhas_originais=len(raw_df))
    df = raw_df.copy()

    # 1-2. datas: parse, remoção de inválidas e ordenação
    datas = parse_dates(df[date_col])
    if datas.notna().sum() == 0:
        raise DadosInvalidosError(
            f"Nenhum valor da primeira coluna ('{date_col}') pôde ser interpretado como data. "
            "A primeira coluna da planilha deve conter datas (ex.: 31/12/2024)."
        )
    report.datas_invalidas_removidas = int(datas.isna().sum())
    df = df.loc[datas.notna()].copy()
    df[date_col] = datas.dropna()
    df = df.sort_values(date_col).reset_index(drop=True)

    # 3. alvo ausente
    antes = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    report.alvo_ausente_removidas = antes - len(df)
    if len(df) == 0:
        raise DadosInvalidosError(f"Todas as linhas têm a variável alvo '{target_col}' vazia.")

    # se o usuário forçou regressão, o alvo precisa ser numérico
    if problem_type == "regressao" and not pd.api.types.is_numeric_dtype(df[target_col]):
        convertido = pd.to_numeric(df[target_col], errors="coerce")
        if convertido.isna().all():
            raise DadosInvalidosError(
                f"A variável alvo '{target_col}' contém texto e não pode ser usada em regressão. "
                "Use o modo classificação ou escolha outra coluna."
            )
        perdidas = int(convertido.isna().sum())
        df[target_col] = convertido
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        report.alvo_ausente_removidas += perdidas

    # 4. variáveis derivadas da data (candidatas; a data original sai do conjunto)
    dt = df[date_col].dt
    df["mes"] = dt.month
    df["dia_semana"] = dt.dayofweek
    df["dia_do_mes"] = dt.day
    df["trimestre"] = dt.quarter
    df["indice_tendencia"] = np.arange(len(df))
    report.features_de_data_criadas = list(DATE_FEATURES)

    feature_cols = [c for c in df.columns if c not in (date_col, target_col)]

    # tenta converter colunas de texto que são números com vírgula decimal (padrão BR)
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            convertido = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
            if convertido.notna().mean() >= 0.8:
                df[col] = convertido

    # 5. outliers via IQR: clipping nas features numéricas; alvo apenas reportado
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]) or col in DATE_FEATURES:
            continue
        ajustados = _clip_outliers_iqr(df, col)
        if ajustados:
            report.outliers_ajustados[col] = ajustados
    if problem_type == "regressao" and pd.api.types.is_numeric_dtype(df[target_col]):
        report.outliers_alvo_detectados = _count_outliers_iqr(df[target_col])

    # 6-7. colunas inúteis: quase vazias, constantes, alta cardinalidade
    for col in list(feature_cols):
        serie = df[col]
        n_faltantes = int(serie.isna().sum())
        if n_faltantes:
            report.ausentes_por_coluna[col] = n_faltantes
        if n_faltantes / len(df) > MAX_MISSING_RATIO:
            report.colunas_removidas.append(
                (col, f"{n_faltantes / len(df):.0%} de valores ausentes (acima de {MAX_MISSING_RATIO:.0%})")
            )
            feature_cols.remove(col)
            report.ausentes_por_coluna.pop(col, None)
        elif serie.dropna().nunique() <= 1:
            report.colunas_removidas.append((col, "valor constante (sem poder preditivo)"))
            feature_cols.remove(col)
            report.ausentes_por_coluna.pop(col, None)
        elif not pd.api.types.is_numeric_dtype(serie) and serie.nunique() > MAX_CATEGORIES:
            report.colunas_removidas.append(
                (col, f"{serie.nunique()} categorias distintas (alta cardinalidade, acima de {MAX_CATEGORIES})")
            )
            feature_cols.remove(col)
            report.ausentes_por_coluna.pop(col, None)

    treated_df = df[[date_col] + feature_cols + [target_col]].copy()
    report.linhas_finais = len(treated_df)

    # 8. metadados por feature (usados no simulador e na seleção de variáveis)
    feature_meta = {}
    for col in feature_cols:
        serie = treated_df[col]
        if pd.api.types.is_numeric_dtype(serie):
            eh_inteira = bool(
                pd.api.types.is_integer_dtype(serie)
                or np.allclose(serie.dropna(), np.round(serie.dropna()), equal_nan=True)
            )
            feature_meta[col] = {
                "tipo": "numerica",
                "min": float(serie.min()),
                "max": float(serie.max()),
                "media": float(serie.mean()),
                "mediana": float(serie.median()),
                "inteira": eh_inteira,
                "derivada_da_data": col in DATE_FEATURES,
            }
        else:
            opcoes = sorted(serie.dropna().astype(str).unique().tolist())
            moda = serie.mode()
            feature_meta[col] = {
                "tipo": "categorica",
                "opcoes": opcoes,
                "moda": str(moda.iloc[0]) if not moda.empty else (opcoes[0] if opcoes else ""),
                "derivada_da_data": False,
            }

    return treated_df, feature_meta, report


def _iqr_bounds(serie: pd.Series):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr


def _clip_outliers_iqr(df: pd.DataFrame, col: str) -> int:
    serie = df[col]
    low, high = _iqr_bounds(serie.dropna())
    mask = (serie < low) | (serie > high)
    n = int(mask.sum())
    if n:
        df[col] = serie.clip(lower=low, upper=high)
    return n


def _count_outliers_iqr(serie: pd.Series) -> int:
    s = serie.dropna()
    if s.empty:
        return 0
    low, high = _iqr_bounds(s)
    return int(((s < low) | (s > high)).sum())
