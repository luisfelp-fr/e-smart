"""Transformações normalizadoras com inversão exata.

Cada transformação é registrada como ``nome + parâmetros`` (nunca funções),
o que mantém os resultados serializáveis e permite converter limites de
especificação para o espaço transformado (forward) e percentis de volta à
escala original (inverse). Todas as transformações são monótonas crescentes,
condição para que limites transformados preservem o significado.

Famílias tentadas, da mais simples à mais flexível:

- ``log``  : ln(x + c)                          (c desloca dados p/ positivos)
- ``sqrt`` : raiz(x + c)
- ``boxcox``: Box-Cox com λ ajustado por MV      (exige x + c > 0)
- ``yeojohnson``: Yeo-Johnson (aceita negativos; inversa analítica própria)
- ``johnsonsu`` / ``johnsonsb``: z = γ + δ·g((x-ξ)/λ) da família Johnson,
  ajustada por MV — equivale à transformação quantil-normal da distribuição
  ajustada e é exatamente inversível.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from .normality import NormalityResult, test_normality

# ordem de preferência: mais simples primeiro (desempate na seleção)
COMPLEXITY = {"log": 0, "sqrt": 1, "boxcox": 2, "yeojohnson": 3,
              "johnsonsu": 4, "johnsonsb": 5}

LABELS = {
    "log": "logaritmo natural",
    "sqrt": "raiz quadrada",
    "boxcox": "Box-Cox",
    "yeojohnson": "Yeo-Johnson",
    "johnsonsu": "Johnson (ilimitada)",
    "johnsonsb": "Johnson (limitada)",
}


@dataclass
class TransformFit:
    """Uma transformação ajustada: nome + parâmetros + normalidade obtida."""

    name: str
    params: dict[str, float] = field(default_factory=dict)
    achieved: NormalityResult | None = None

    @property
    def label(self) -> str:
        extra = ""
        if self.name == "boxcox":
            extra = f" (λ={self.params.get('lmbda', float('nan')):.3f})"
        elif self.name == "yeojohnson":
            extra = f" (λ={self.params.get('lmbda', float('nan')):.3f})"
        shift = self.params.get("shift", 0.0)
        if shift:
            extra += f" [deslocamento +{shift:g}]"
        return LABELS.get(self.name, self.name) + extra


@dataclass
class TransformSearch:
    """Resultado da busca: candidatas testadas e a melhor (se normalizou)."""

    candidates: list[TransformFit] = field(default_factory=list)
    best: TransformFit | None = None
    note: str = ""


# ----------------------------------------------------------- forward/inverse

def forward(name: str, params: dict[str, float], x: np.ndarray | float) -> np.ndarray:
    """Aplica a transformação ``nome`` com ``params`` a valores na escala original."""
    x = np.asarray(x, dtype=float)
    shift = params.get("shift", 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "log":
            return np.log(x + shift)
        if name == "sqrt":
            return np.sqrt(np.maximum(x + shift, 0.0))
        if name == "boxcox":
            return special.boxcox(x + shift, params["lmbda"])
        if name == "yeojohnson":
            return _yeojohnson_forward(x, params["lmbda"])
        if name in ("johnsonsu", "johnsonsb"):
            a, b = params["a"], params["b"]
            loc, scale = params["loc"], params["scale"]
            u = (x - loc) / scale
            if name == "johnsonsu":
                return a + b * np.arcsinh(u)
            u = np.clip(u, 1e-12, 1 - 1e-12)
            return a + b * np.log(u / (1.0 - u))
    raise ValueError(f"Transformação desconhecida: {name}")


def inverse(name: str, params: dict[str, float], y: np.ndarray | float) -> np.ndarray:
    """Converte valores do espaço transformado de volta à escala original."""
    y = np.asarray(y, dtype=float)
    shift = params.get("shift", 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "log":
            return np.exp(y) - shift
        if name == "sqrt":
            return np.square(y) - shift
        if name == "boxcox":
            return special.inv_boxcox(y, params["lmbda"]) - shift
        if name == "yeojohnson":
            return _yeojohnson_inverse(y, params["lmbda"])
        if name in ("johnsonsu", "johnsonsb"):
            a, b = params["a"], params["b"]
            loc, scale = params["loc"], params["scale"]
            z = (y - a) / b
            if name == "johnsonsu":
                return loc + scale * np.sinh(z)
            return loc + scale / (1.0 + np.exp(-z))
    raise ValueError(f"Transformação desconhecida: {name}")


def _yeojohnson_forward(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Yeo-Johnson (idêntica à do scipy, vetorizada aqui por clareza)."""
    out = np.empty_like(x)
    pos = x >= 0
    if abs(lmbda) > 1e-12:
        out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
    else:
        out[pos] = np.log1p(x[pos])
    if abs(lmbda - 2.0) > 1e-12:
        out[~pos] = -(((-x[~pos] + 1.0) ** (2.0 - lmbda)) - 1.0) / (2.0 - lmbda)
    else:
        out[~pos] = -np.log1p(-x[~pos])
    return out


def _yeojohnson_inverse(t: np.ndarray, lmbda: float) -> np.ndarray:
    """Inversa analítica da Yeo-Johnson (o scipy não fornece)."""
    out = np.empty_like(t)
    pos = t >= 0
    if abs(lmbda) > 1e-12:
        out[pos] = (t[pos] * lmbda + 1.0) ** (1.0 / lmbda) - 1.0
    else:
        out[pos] = np.expm1(t[pos])
    if abs(lmbda - 2.0) > 1e-12:
        out[~pos] = 1.0 - (-(2.0 - lmbda) * t[~pos] + 1.0) ** (1.0 / (2.0 - lmbda))
    else:
        out[~pos] = -np.expm1(-t[~pos])
    return out


# ----------------------------------------------------------------- ajuste

def _fit_one(name: str, x: np.ndarray) -> TransformFit | None:
    """Ajusta uma família à amostra; None quando não aplicável/instável."""
    params: dict[str, float] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name in ("log", "sqrt", "boxcox"):
                xmin = float(np.min(x))
                shift = 0.0 if xmin > 0 else 1.0 - xmin  # leva o mínimo a 1
                if shift:
                    params["shift"] = shift
                if name == "boxcox":
                    _, lmbda = stats.boxcox(x + shift)
                    params["lmbda"] = float(lmbda)
            elif name == "yeojohnson":
                _, lmbda = stats.yeojohnson(x)
                params["lmbda"] = float(lmbda)
            elif name == "johnsonsu":
                a, b, loc, scale = stats.johnsonsu.fit(x)
                if b <= 0 or scale <= 0:
                    return None
                params = {"a": float(a), "b": float(b),
                          "loc": float(loc), "scale": float(scale)}
            elif name == "johnsonsb":
                a, b, loc, scale = stats.johnsonsb.fit(x)
                if b <= 0 or scale <= 0:
                    return None
                # dados precisam caber estritamente no suporte (loc, loc+scale)
                if np.min(x) <= loc or np.max(x) >= loc + scale:
                    return None
                params = {"a": float(a), "b": float(b),
                          "loc": float(loc), "scale": float(scale)}
            else:
                return None
        t = forward(name, params, x)
        if not np.all(np.isfinite(t)):
            return None
        return TransformFit(name=name, params=params)
    except Exception:
        return None


def best_normalizing_transform(
    s: pd.Series, alpha: float = 0.05
) -> TransformSearch:
    """Tenta as famílias e devolve a que melhor normaliza (se alguma passar).

    Critério: entre as candidatas cujo Anderson-Darling da série transformada
    passa (p > alfa), vence a de menor estatística AD; empates aproximados
    são resolvidos pela transformação mais simples (melhor de explicar).
    """
    search = TransformSearch()
    x = pd.Series(s).dropna().astype(float).to_numpy()
    if len(x) < 8 or np.std(x) == 0:
        search.note = "Dados insuficientes para tentar transformações."
        return search

    for name in sorted(COMPLEXITY, key=COMPLEXITY.get):
        fit = _fit_one(name, x)
        if fit is None:
            continue
        fit.achieved = test_normality(
            pd.Series(forward(fit.name, fit.params, x)), alpha=alpha
        )
        search.candidates.append(fit)

    passing = [
        f for f in search.candidates
        if f.achieved is not None and f.achieved.is_normal
    ]
    if passing:
        # menor AD vence; empate (2 casas) fica com a mais simples
        passing.sort(
            key=lambda f: (round(f.achieved.ad_stat, 2), COMPLEXITY[f.name])
        )
        search.best = passing[0]
        if search.best.name.startswith("johnson") and len(x) < 30:
            search.note = (
                "A transformação de Johnson tem 4 parâmetros e pode se ajustar "
                "demais em amostras pequenas — confirme com mais dados."
            )
    else:
        search.note = (
            "Nenhuma transformação atingiu normalidade; seguindo para a "
            "análise não-paramétrica (box-plot/percentis)."
        )
    return search
