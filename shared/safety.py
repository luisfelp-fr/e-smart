"""Defesas contra arquivo hostil na entrada e na saída.

O app recebe planilha de qualquer pessoa que alcance a URL e devolve arquivo
para download. São duas superfícies distintas:

**Na entrada** — bibliotecas de parsing (openpyxl, xlrd, odfpy) são código
complexo lendo bytes não confiáveis, e historicamente acumulam CVEs. A
defesa barata é não deixar o arquivo chegar até elas quando ele já é
obviamente inválido: conferir a assinatura contra a extensão e recusar zip
que promete expandir para centenas de MB.

**Na saída** — um CSV/XLSX que o usuário abre no Excel executa fórmula.
Célula de texto começando com ``=``, ``+``, ``-``, ``@``, tab ou CR é
interpretada como fórmula, e ``=cmd|'/c calc'!A1`` vira execução de comando
na máquina de quem abriu. Como o conteúdo vem da planilha enviada, o app
seria o mensageiro. Prefixar com aspa simples neutraliza sem perder o dado.
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd

# Assinaturas de arquivo. xlsx/xlsm/ods são zip; xls é o formato composto OLE2.
_ZIP_MAGIC = b"PK\x03\x04"
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_MAGIC_BY_EXT = {
    ".xlsx": (_ZIP_MAGIC,),
    ".xlsm": (_ZIP_MAGIC,),
    ".ods": (_ZIP_MAGIC,),
    ".xls": (_OLE2_MAGIC, _ZIP_MAGIC),  # .xls "moderno" às vezes é xlsx renomeado
}

# Teto do total DECLARADO como descomprimido no cabeçalho do zip. Um .xlsx
# de 200 KB que promete expandir para 5 GB é bomba de descompressão: o
# parser aloca a memória antes de qualquer validação de conteúdo.
MAX_UNCOMPRESSED_BYTES = int(
    float(os.environ.get("ESMART_MAX_UNCOMPRESSED_MB", "500")) * 1024 * 1024
)

# Um .xlsx normal tem dezenas de entradas. Milhares indicam bomba por
# quantidade de arquivos, que não depende do tamanho de nenhum deles.
MAX_ZIP_ENTRIES = int(os.environ.get("ESMART_MAX_ZIP_ENTRIES", "5000"))

# Caracteres que fazem o Excel/LibreOffice tratar a célula como fórmula.
_FORMULA_STARTERS = ("=", "+", "-", "@", "\t", "\r")


def _head(path: str, n: int = 8) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _check_zip_expansion(path: str) -> None:
    """Recusa zip que promete expandir demais, ANTES de o parser abrir.

    O tamanho aqui é o DECLARADO no cabeçalho — um arquivo malicioso pode
    mentir. Não é defesa completa: é o filtro barato que elimina a bomba de
    descompressão clássica sem custo de leitura. O teto de memória real
    continua sendo a máquina.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            entradas = zf.infolist()
            if len(entradas) > MAX_ZIP_ENTRIES:
                raise ValueError(
                    f"A planilha contém {len(entradas)} arquivos internos, "
                    f"acima do limite de {MAX_ZIP_ENTRIES}. Arquivos assim "
                    "costumam estar corrompidos ou ser maliciosos."
                )
            total = sum(e.file_size for e in entradas)
    except zipfile.BadZipFile as e:
        raise ValueError(
            "O arquivo tem extensão de planilha mas não é um arquivo válido. "
            "Reabra no Excel e salve de novo como .xlsx."
        ) from e
    except OSError as e:
        raise ValueError(f"Não foi possível ler o arquivo: {e}") from e

    if total > MAX_UNCOMPRESSED_BYTES:
        mb = total / (1024 * 1024)
        teto = MAX_UNCOMPRESSED_BYTES / (1024 * 1024)
        raise ValueError(
            f"A planilha expande para cerca de {mb:.0f} MB, acima do limite "
            f"de {teto:.0f} MB. Reduza o período ou o número de colunas, ou "
            "envie em CSV."
        )


def validate_upload(path: str) -> None:
    """Valida assinatura, coerência com a extensão e expansão do zip.

    Levanta ``ValueError`` com mensagem para o usuário final. Chamada uma vez
    antes de qualquer parser tocar no arquivo.
    """
    ext = os.path.splitext(path)[1].lower()
    head = _head(path)
    if not head:
        raise ValueError("O arquivo enviado está vazio ou não pôde ser lido.")

    esperados = _MAGIC_BY_EXT.get(ext)
    if esperados is not None:
        if not any(head.startswith(m) for m in esperados):
            raise ValueError(
                f"O conteúdo do arquivo não corresponde à extensão {ext}. "
                "Ele pode ter sido renomeado ou estar corrompido — reabra no "
                "Excel e salve novamente."
            )
        if head.startswith(_ZIP_MAGIC):
            _check_zip_expansion(path)
        return

    # CSV/TXT: não têm assinatura, mas também não podem ser binário disfarçado
    if head.startswith(_ZIP_MAGIC) or head.startswith(_OLE2_MAGIC):
        raise ValueError(
            f"O arquivo tem extensão {ext} mas o conteúdo é uma planilha "
            "binária. Renomeie para a extensão correta (.xlsx) e envie de "
            "novo."
        )


def _escape_cell(value):
    """Neutraliza fórmula numa célula de texto, preservando o conteúdo.

    Só mexe em texto: número continua número, e data continua data. O prefixo
    ``'`` é a convenção que o Excel entende como "isto é texto literal" e não
    aparece no conteúdo da célula ao abrir.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_STARTERS):
        return "'" + value
    return value


def _tem_texto(dtype) -> bool:
    """A coluna pode conter texto?

    Checar ``dtype == object`` não basta: a partir do pandas 3 uma coluna de
    texto tem dtype ``str``, e o teste antigo pulava justamente as colunas
    que precisam ser tratadas — a defesa passava a não fazer nada, em
    silêncio, só por atualizar a biblioteca.
    """
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


def neutralize_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve cópia do DataFrame sem células que o Excel executaria.

    Cobre valores de texto, nomes de coluna e rótulos do índice — os três
    chegam ao arquivo exportado e os três são interpretados como fórmula.
    """
    out = df.copy()
    for col in out.columns:
        if _tem_texto(out[col].dtype):
            out[col] = out[col].map(_escape_cell)
    out.columns = pd.Index([_escape_cell(c) for c in out.columns],
                           name=out.columns.name)
    if _tem_texto(out.index.dtype):
        out.index = pd.Index([_escape_cell(i) for i in out.index],
                             name=out.index.name)
    return out
