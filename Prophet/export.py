"""Serialização do modelo treinado (.joblib) e geração do script de exemplo de uso."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import joblib
import sklearn


def serialize_artifact(
    pipeline,
    selected_features: list,
    feature_meta: dict,
    target_col: str,
    problem_type: str,
    model_name: str,
    metrics: dict,
) -> bytes:
    """Empacota pipeline + metadados em um único .joblib (bytes para download)."""
    artifact = {
        "pipeline": pipeline,
        "features": list(selected_features),
        "feature_meta": {c: feature_meta[c] for c in selected_features},
        "target": target_col,
        "problem_type": problem_type,
        "model_name": model_name,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def generate_usage_script(selected_features: list, feature_meta: dict, target_col: str,
                          problem_type: str) -> str:
    """Gera exemplo_uso.py pronto para rodar com o .joblib baixado."""
    linhas_exemplo = []
    for col in selected_features:
        meta = feature_meta[col]
        if meta["tipo"] == "numerica":
            valor = meta["mediana"]
            valor = int(round(valor)) if meta.get("inteira") else round(valor, 4)
            linhas_exemplo.append(f'    "{col}": {valor},')
        else:
            linhas_exemplo.append(f'    "{col}": "{meta["moda"]}",')
    exemplo = "\n".join(linhas_exemplo)

    colunas = ", ".join(f'"{c}"' for c in selected_features)
    categoricas = [c for c in selected_features if feature_meta[c]["tipo"] == "categorica"]
    nota_cat = ""
    if categoricas:
        nota_cat = (
            "\n# Observação: para as colunas categóricas ("
            + ", ".join(categoricas)
            + "),\n# valores nunca vistos no treino são ignorados pelo codificador "
            "(previsão continua funcionando)."
        )

    return f'''"""Exemplo de uso do modelo baixado do Criador de Modelos Preditivos.

Requisitos (instale com pip):
    pip install pandas joblib scikit-learn=={sklearn.__version__}

Coloque este arquivo na mesma pasta que 'modelo_preditivo.joblib' e execute:
    python exemplo_uso.py
"""

import joblib
import pandas as pd

# Carrega o artefato: um dicionário com o pipeline treinado e os metadados
artifact = joblib.load("modelo_preditivo.joblib")
pipeline = artifact["pipeline"]

print("Modelo:", artifact["model_name"])
print("Alvo previsto:", artifact["target"])
print("Métricas na avaliação:", artifact["metrics"])
print("Colunas necessárias:", artifact["features"])

# Monte um DataFrame com as colunas exatamente com estes nomes: {colunas}
# (os valores abaixo são um exemplo real tirado da base de treino){nota_cat}
novos_dados = pd.DataFrame([{{
{exemplo}
}}])

previsao = pipeline.predict(novos_dados)
print("\\nPrevisão de '{target_col}':", previsao[0])
'''
