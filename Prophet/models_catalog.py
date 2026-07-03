"""Catálogo de modelos de machine learning com guia de prós/contras em PT-BR.

Cada entrada define:
  - nome, classe sklearn e parâmetros fixos
  - params_ui: especificação dos hiperparâmetros ajustáveis na interface
  - quando_usar, pros e contras (exibidos como tabelas no app)
"""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

MODEL_CATALOG = {
    "regressao": {
        "linear": {
            "nome": "Regressão Linear",
            "classe": LinearRegression,
            "params_fixos": {},
            "aceita_random_state": False,
            "params_ui": {},
            "quando_usar": "Relações aproximadamente lineares, poucos dados, ou quando a "
            "interpretabilidade (entender o peso de cada variável) é essencial.",
            "pros": [
                "Muito rápida de treinar",
                "Totalmente interpretável (coeficientes)",
                "Funciona bem com poucos dados",
                "Base de comparação para modelos mais complexos",
            ],
            "contras": [
                "Não captura relações não lineares",
                "Sensível a outliers",
                "Sofre com variáveis muito correlacionadas entre si",
            ],
        },
        "ridge": {
            "nome": "Ridge (Linear com regularização L2)",
            "classe": Ridge,
            "params_fixos": {},
            "aceita_random_state": True,
            "params_ui": {
                "alpha": {
                    "tipo": "slider_float",
                    "min": 0.01,
                    "max": 100.0,
                    "default": 1.0,
                    "step": 0.01,
                    "rotulo": "Regularização (alpha)",
                },
            },
            "quando_usar": "Como a Regressão Linear, mas quando há muitas variáveis "
            "correlacionadas entre si ou risco de sobreajuste.",
            "pros": [
                "Estável com variáveis correlacionadas",
                "Reduz sobreajuste em relação à linear pura",
                "Rápida e interpretável",
            ],
            "contras": [
                "Continua limitada a relações lineares",
                "Exige ajuste do parâmetro de regularização",
                "Não zera coeficientes (mantém todas as variáveis)",
            ],
        },
        "lasso": {
            "nome": "Lasso (Linear com regularização L1)",
            "classe": Lasso,
            "params_fixos": {"max_iter": 10000},
            "aceita_random_state": True,
            "params_ui": {
                "alpha": {
                    "tipo": "slider_float",
                    "min": 0.001,
                    "max": 10.0,
                    "default": 0.1,
                    "step": 0.001,
                    "rotulo": "Regularização (alpha)",
                },
            },
            "quando_usar": "Quando há mais variáveis do que o necessário e você quer que o "
            "próprio modelo selecione as importantes (zera coeficientes irrelevantes).",
            "pros": [
                "Faz seleção automática de variáveis",
                "Bom quando há mais colunas que linhas",
                "Modelo final enxuto e interpretável",
            ],
            "contras": [
                "Limitado a relações lineares",
                "Pode descartar variáveis relevantes se alpha for alto",
                "Instável quando variáveis são muito correlacionadas",
            ],
        },
        "random_forest": {
            "nome": "Random Forest (Floresta Aleatória)",
            "classe": RandomForestRegressor,
            "params_fixos": {"n_jobs": -1},
            "aceita_random_state": True,
            "params_ui": {
                "n_estimators": {
                    "tipo": "slider_int",
                    "min": 50,
                    "max": 500,
                    "default": 200,
                    "step": 50,
                    "rotulo": "Número de árvores",
                },
                "max_depth": {
                    "tipo": "slider_int",
                    "min": 2,
                    "max": 30,
                    "default": 12,
                    "step": 1,
                    "rotulo": "Profundidade máxima",
                },
            },
            "quando_usar": "Escolha padrão para a maioria dos problemas com dados tabulares: "
            "robusta, captura não linearidades e exige pouco ajuste.",
            "pros": [
                "Captura relações não lineares e interações",
                "Robusta a outliers e escalas diferentes",
                "Pouco sensível a hiperparâmetros",
                "Fornece importância das variáveis",
            ],
            "contras": [
                "Menos interpretável que modelos lineares",
                "Modelo maior e mais lento para prever",
                "Não extrapola além dos valores vistos no treino",
            ],
        },
        "gradient_boosting": {
            "nome": "Gradient Boosting",
            "classe": GradientBoostingRegressor,
            "params_fixos": {},
            "aceita_random_state": True,
            "params_ui": {
                "n_estimators": {
                    "tipo": "slider_int",
                    "min": 50,
                    "max": 500,
                    "default": 200,
                    "step": 50,
                    "rotulo": "Número de estágios",
                },
                "learning_rate": {
                    "tipo": "slider_float",
                    "min": 0.01,
                    "max": 0.5,
                    "default": 0.1,
                    "step": 0.01,
                    "rotulo": "Taxa de aprendizado",
                },
                "max_depth": {
                    "tipo": "slider_int",
                    "min": 2,
                    "max": 10,
                    "default": 3,
                    "step": 1,
                    "rotulo": "Profundidade máxima",
                },
            },
            "quando_usar": "Quando se busca a melhor precisão possível em dados tabulares e há "
            "tempo para ajustar hiperparâmetros. Costuma vencer competições.",
            "pros": [
                "Geralmente a melhor precisão em dados tabulares",
                "Captura relações complexas",
                "Fornece importância das variáveis",
            ],
            "contras": [
                "Mais sensível a hiperparâmetros",
                "Treino mais lento",
                "Risco de sobreajuste se mal configurado",
            ],
        },
        "svr": {
            "nome": "SVR (Máquina de Vetores de Suporte)",
            "classe": SVR,
            "params_fixos": {},
            "aceita_random_state": False,
            "params_ui": {
                "C": {
                    "tipo": "slider_float",
                    "min": 0.1,
                    "max": 100.0,
                    "default": 1.0,
                    "step": 0.1,
                    "rotulo": "Penalidade (C)",
                },
                "kernel": {
                    "tipo": "selectbox",
                    "opcoes": ["rbf", "linear", "poly"],
                    "default": "rbf",
                    "rotulo": "Kernel",
                },
            },
            "quando_usar": "Conjuntos pequenos/médios com relações não lineares suaves; "
            "bom quando há muitas variáveis e poucas linhas.",
            "pros": [
                "Eficaz em espaços de alta dimensão",
                "Flexível via escolha do kernel",
                "Robusto a sobreajuste em bases pequenas",
            ],
            "contras": [
                "Lento em bases grandes (milhares de linhas)",
                "Difícil de interpretar",
                "Sensível à escala e aos hiperparâmetros",
            ],
        },
        "knn": {
            "nome": "KNN (Vizinhos mais Próximos)",
            "classe": KNeighborsRegressor,
            "params_fixos": {},
            "aceita_random_state": False,
            "params_ui": {
                "n_neighbors": {
                    "tipo": "slider_int",
                    "min": 1,
                    "max": 30,
                    "default": 5,
                    "step": 1,
                    "rotulo": "Número de vizinhos (K)",
                },
            },
            "quando_usar": "Bases pequenas onde casos parecidos tendem a ter resultados "
            "parecidos; útil como comparação rápida.",
            "pros": [
                "Simples de entender e explicar",
                "Sem fase de treino (memoriza os dados)",
                "Captura padrões locais",
            ],
            "contras": [
                "Lento para prever em bases grandes",
                "Sofre com muitas variáveis (maldição da dimensionalidade)",
                "Sensível à escala e a variáveis irrelevantes",
            ],
        },
    },
    "classificacao": {
        "logistica": {
            "nome": "Regressão Logística",
            "classe": LogisticRegression,
            "params_fixos": {"max_iter": 2000},
            "aceita_random_state": True,
            "params_ui": {
                "C": {
                    "tipo": "slider_float",
                    "min": 0.01,
                    "max": 10.0,
                    "default": 1.0,
                    "step": 0.01,
                    "rotulo": "Inverso da regularização (C)",
                },
            },
            "quando_usar": "Primeiro modelo a testar em classificação: rápido, interpretável "
            "e fornece probabilidades bem calibradas.",
            "pros": [
                "Rápida e interpretável (peso de cada variável)",
                "Probabilidades bem calibradas",
                "Funciona bem com poucos dados",
            ],
            "contras": [
                "Fronteira de decisão apenas linear",
                "Sofre com variáveis muito correlacionadas",
                "Exige tratamento de não linearidades manualmente",
            ],
        },
        "random_forest": {
            "nome": "Random Forest (Floresta Aleatória)",
            "classe": RandomForestClassifier,
            "params_fixos": {"n_jobs": -1},
            "aceita_random_state": True,
            "params_ui": {
                "n_estimators": {
                    "tipo": "slider_int",
                    "min": 50,
                    "max": 500,
                    "default": 200,
                    "step": 50,
                    "rotulo": "Número de árvores",
                },
                "max_depth": {
                    "tipo": "slider_int",
                    "min": 2,
                    "max": 30,
                    "default": 12,
                    "step": 1,
                    "rotulo": "Profundidade máxima",
                },
            },
            "quando_usar": "Escolha padrão para classificação em dados tabulares: robusta e "
            "com bom desempenho sem muito ajuste.",
            "pros": [
                "Captura relações não lineares e interações",
                "Robusta a outliers e dados com escalas diferentes",
                "Fornece importância das variáveis",
                "Lida bem com classes desbalanceadas (com ajustes)",
            ],
            "contras": [
                "Menos interpretável que a logística",
                "Modelo maior e mais lento para prever",
                "Probabilidades menos calibradas",
            ],
        },
        "gradient_boosting": {
            "nome": "Gradient Boosting",
            "classe": GradientBoostingClassifier,
            "params_fixos": {},
            "aceita_random_state": True,
            "params_ui": {
                "n_estimators": {
                    "tipo": "slider_int",
                    "min": 50,
                    "max": 500,
                    "default": 200,
                    "step": 50,
                    "rotulo": "Número de estágios",
                },
                "learning_rate": {
                    "tipo": "slider_float",
                    "min": 0.01,
                    "max": 0.5,
                    "default": 0.1,
                    "step": 0.01,
                    "rotulo": "Taxa de aprendizado",
                },
                "max_depth": {
                    "tipo": "slider_int",
                    "min": 2,
                    "max": 10,
                    "default": 3,
                    "step": 1,
                    "rotulo": "Profundidade máxima",
                },
            },
            "quando_usar": "Quando se busca a maior precisão possível e há tempo para ajustar "
            "hiperparâmetros.",
            "pros": [
                "Geralmente a melhor precisão em dados tabulares",
                "Captura relações complexas",
                "Fornece importância das variáveis",
            ],
            "contras": [
                "Sensível a hiperparâmetros",
                "Treino mais lento",
                "Risco de sobreajuste se mal configurado",
            ],
        },
        "svm": {
            "nome": "SVM (Máquina de Vetores de Suporte)",
            "classe": SVC,
            "params_fixos": {"probability": True},
            "aceita_random_state": True,
            "params_ui": {
                "C": {
                    "tipo": "slider_float",
                    "min": 0.1,
                    "max": 100.0,
                    "default": 1.0,
                    "step": 0.1,
                    "rotulo": "Penalidade (C)",
                },
                "kernel": {
                    "tipo": "selectbox",
                    "opcoes": ["rbf", "linear", "poly"],
                    "default": "rbf",
                    "rotulo": "Kernel",
                },
            },
            "quando_usar": "Bases pequenas/médias com fronteiras de decisão complexas; bom "
            "com muitas variáveis e poucas linhas.",
            "pros": [
                "Eficaz em alta dimensão",
                "Fronteiras de decisão flexíveis (kernel)",
                "Robusto em bases pequenas",
            ],
            "contras": [
                "Lento em bases grandes",
                "Difícil de interpretar",
                "Sensível à escala e aos hiperparâmetros",
            ],
        },
        "knn": {
            "nome": "KNN (Vizinhos mais Próximos)",
            "classe": KNeighborsClassifier,
            "params_fixos": {},
            "aceita_random_state": False,
            "params_ui": {
                "n_neighbors": {
                    "tipo": "slider_int",
                    "min": 1,
                    "max": 30,
                    "default": 5,
                    "step": 1,
                    "rotulo": "Número de vizinhos (K)",
                },
            },
            "quando_usar": "Bases pequenas onde exemplos parecidos tendem a pertencer à mesma "
            "classe.",
            "pros": [
                "Simples de entender e explicar",
                "Sem fase de treino",
                "Captura padrões locais",
            ],
            "contras": [
                "Lento para prever em bases grandes",
                "Sofre com muitas variáveis",
                "Sensível à escala e a variáveis irrelevantes",
            ],
        },
        "naive_bayes": {
            "nome": "Naive Bayes (Gaussiano)",
            "classe": GaussianNB,
            "params_fixos": {},
            "aceita_random_state": False,
            "params_ui": {},
            "quando_usar": "Bases muito pequenas ou como comparação rápida; bom quando as "
            "variáveis são razoavelmente independentes entre si.",
            "pros": [
                "Extremamente rápido",
                "Funciona com pouquíssimos dados",
                "Bom ponto de partida (baseline)",
            ],
            "contras": [
                "Assume independência entre variáveis (raramente verdade)",
                "Precisão limitada em problemas complexos",
                "Sensível a variáveis irrelevantes",
            ],
        },
    },
}


def recommend_model(n_rows: int, n_features: int, problem_type: str) -> tuple[str, str]:
    """Sugere um modelo do catálogo. Retorna (chave_do_modelo, justificativa)."""
    if n_features > n_rows:
        if problem_type == "regressao":
            return "lasso", (
                "Há mais variáveis do que linhas — a regularização L1 do Lasso seleciona "
                "automaticamente as variáveis mais relevantes e evita sobreajuste."
            )
        return "logistica", (
            "Há mais variáveis do que linhas — um modelo linear regularizado é essencial "
            "para evitar sobreajuste."
        )
    if n_rows < 100:
        if problem_type == "regressao":
            return "ridge", (
                f"Base pequena ({n_rows} linhas) — modelos simples e regularizados tendem a "
                "generalizar melhor do que modelos complexos."
            )
        return "logistica", (
            f"Base pequena ({n_rows} linhas) — a Regressão Logística é estável e "
            "interpretável com poucos dados."
        )
    if n_rows < 5000:
        return "random_forest", (
            f"Base de tamanho médio ({n_rows} linhas) — a Random Forest oferece bom "
            "equilíbrio entre desempenho, robustez e pouca necessidade de ajuste."
        )
    return "gradient_boosting", (
        f"Base grande ({n_rows} linhas) — o Gradient Boosting costuma atingir a melhor "
        "precisão quando há bastante dado disponível."
    )
