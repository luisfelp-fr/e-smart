"""Testes das defesas contra arquivo hostil (shared/safety.py).

Cada teste tenta o ataque e verifica que ele é barrado. Testar só o caminho
feliz não prova nada aqui: a defesa existe justamente para o caminho infeliz.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.parsing import read_all_sheets, read_raw  # noqa: E402
from shared.safety import (  # noqa: E402
    MAX_UNCOMPRESSED_BYTES,
    neutralize_formulas,
    validate_upload,
)


def _escreve(tmp_path, nome: str, conteudo: bytes) -> str:
    caminho = os.path.join(tmp_path, nome)
    with open(caminho, "wb") as f:
        f.write(conteudo)
    return caminho


def _xlsx_valido(tmp_path, nome="ok.xlsx") -> str:
    caminho = os.path.join(tmp_path, nome)
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(caminho, index=False)
    return caminho


# ------------------------------------------------ assinatura x extensão

def test_xlsx_que_nao_e_zip_e_recusado(tmp_path):
    """Renomear um executável para .xlsx não deve chegar ao openpyxl."""
    caminho = _escreve(tmp_path, "falso.xlsx", b"MZ\x90\x00" + b"\x00" * 200)
    with pytest.raises(ValueError, match="não corresponde à extensão"):
        validate_upload(caminho)


def test_csv_com_conteudo_binario_e_recusado(tmp_path):
    """Planilha binária renomeada para .csv não é CSV."""
    caminho = _escreve(tmp_path, "disfarce.csv", b"PK\x03\x04" + b"\x00" * 100)
    with pytest.raises(ValueError, match="conteúdo é uma planilha"):
        validate_upload(caminho)


def test_arquivo_vazio_e_recusado(tmp_path):
    caminho = _escreve(tmp_path, "vazio.csv", b"")
    with pytest.raises(ValueError, match="vazio"):
        validate_upload(caminho)


def test_xlsx_valido_passa(tmp_path):
    validate_upload(_xlsx_valido(tmp_path))  # não levanta


def test_csv_de_texto_passa(tmp_path):
    caminho = _escreve(tmp_path, "dados.csv", "a;b\n1;2\n".encode())
    validate_upload(caminho)  # não levanta


# ------------------------------------------------------ bomba de zip

def test_zip_que_promete_expandir_demais_e_recusado(tmp_path):
    """Zip declarando gigabytes descomprimidos não chega ao parser.

    Construímos um zip real cujo cabeçalho declara um tamanho enorme — é
    exatamente a forma da bomba de descompressão clássica.
    """
    caminho = os.path.join(tmp_path, "bomba.xlsx")
    with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(12):  # 720 MB declarados, acima do teto de 500 MB
            zf.writestr(f"p{i}.bin", b"\x00" * (60 * 1024 * 1024))

    with pytest.raises(ValueError, match="expande para cerca de"):
        validate_upload(caminho)

    # o ponto da bomba é ser minúscula em disco: passa por qualquer limite
    # de tamanho de upload e só explode quando o parser a expande
    assert os.path.getsize(caminho) < 5 * 1024 * 1024
    assert MAX_UNCOMPRESSED_BYTES < 12 * 60 * 1024 * 1024


def test_zip_com_entradas_demais_e_recusado(tmp_path):
    caminho = os.path.join(tmp_path, "muitos.xlsx")
    with zipfile.ZipFile(caminho, "w") as zf:
        for i in range(5001):
            zf.writestr(f"f{i}.txt", b"x")
    with pytest.raises(ValueError, match="arquivos internos"):
        validate_upload(caminho)


def test_zip_corrompido_da_erro_amigavel(tmp_path):
    caminho = _escreve(tmp_path, "quebrado.xlsx", b"PK\x03\x04" + b"lixo" * 50)
    with pytest.raises(ValueError, match="não é um arquivo válido"):
        validate_upload(caminho)


# ------------------------------------------ integração com a leitura real

def test_read_raw_valida_antes_de_parsear(tmp_path):
    caminho = _escreve(tmp_path, "falso.xlsx", b"MZ\x90\x00" + b"\x00" * 200)
    with pytest.raises(ValueError, match="não corresponde à extensão"):
        read_raw(caminho)


def test_read_all_sheets_valida_antes_de_parsear(tmp_path):
    caminho = _escreve(tmp_path, "falso.xlsx", b"MZ\x90\x00" + b"\x00" * 200)
    with pytest.raises(ValueError, match="não corresponde à extensão"):
        read_all_sheets(caminho)


def test_xls_desligado_por_padrao(tmp_path):
    """O formato legado sai da superfície a menos que seja religado."""
    from shared.parsing import ALLOW_XLS

    if ALLOW_XLS:
        pytest.skip("ESMART_ALLOW_XLS ligado neste ambiente")
    caminho = _escreve(tmp_path, "antigo.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    with pytest.raises(ValueError, match="não são aceitos por segurança"):
        read_raw(caminho)


# ------------------------------------------------- fórmula na exportação

def test_celula_de_formula_e_neutralizada():
    """=cmd|'/c calc'!A1 numa célula executa ao abrir o arquivo no Excel."""
    df = pd.DataFrame({
        "texto": ["=cmd|'/c calc'!A1", "+1+1", "-2+3", "@SUM(A1)", "normal"],
        "numero": [1, 2, 3, 4, 5],
    })
    seguro = neutralize_formulas(df)

    for original, tratado in zip(df["texto"], seguro["texto"]):
        if original == "normal":
            assert tratado == "normal", "texto inofensivo foi alterado"
        else:
            assert tratado.startswith("'"), f"{original} não foi neutralizado"
            assert tratado[1:] == original, "o conteúdo original se perdeu"
    # números não são tocados
    assert list(seguro["numero"]) == [1, 2, 3, 4, 5]


def test_nome_de_coluna_com_formula_e_neutralizado():
    """O cabeçalho vai para a planilha exportada tanto quanto as células."""
    df = pd.DataFrame({"=1+1": [1, 2], "ok": [3, 4]})
    seguro = neutralize_formulas(df)
    assert "'=1+1" in list(seguro.columns)


def test_indice_de_texto_com_formula_e_neutralizado():
    df = pd.DataFrame({"v": [1, 2]}, index=["=A1", "linha"])
    seguro = neutralize_formulas(df)
    assert list(seguro.index) == ["'=A1", "linha"]


def test_original_nao_e_modificado():
    """A neutralização não pode alterar a base que o app segue usando."""
    df = pd.DataFrame({"t": ["=1+1"]})
    neutralize_formulas(df)
    assert df.loc[0, "t"] == "=1+1"


def test_exportacao_do_app_neutraliza(tmp_path):
    """Ponta a ponta: o .xlsx que o usuário baixa não carrega fórmula."""
    from analytics.utils.helpers import df_to_excel_bytes

    df = pd.DataFrame({"texto": ["=1+1"], "n": [1]})
    lido = pd.read_excel(io.BytesIO(df_to_excel_bytes({"aba": df})), index_col=0)
    assert lido.loc[0, "texto"] == "'=1+1"


# ------------------------------------------------------ HTML do relatório

def test_relatorio_html_escapa_texto_do_usuario():
    """Nome de coluna vira conteúdo do HTML — e o HTML é baixado e circula.

    Uma planilha com a coluna ``<img src=x onerror=...>`` injetaria script no
    preview e no arquivo que o usuário manda por e-mail depois.
    """
    import re

    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
    from report_builder import build_html

    ataque = "<img src=x onerror=alert(1)>"
    item = {
        "id": "a", "module": f"Mod {ataque}", "title": f"Análise {ataque}",
        "texts": [f"texto {ataque}"],
        "tables": {f"Tab {ataque}": pd.DataFrame({ataque: [1]}, index=[ataque])},
        "figures": {},
        "meta": {"limit_review": {
            "indicador": ataque, "situacao": ataque, "metodo": ataque,
            "lsl": 1, "usl": 2, "rec_lsl": 1, "rec_usl": 2,
        }},
    }
    html = build_html([item])

    assert not re.findall(r"<img[^>]*>", html), "tag injetada sobreviveu"
    assert not re.findall(r"<script", html, re.I)
    # escapado, mas ainda legível para quem lê o relatório
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
