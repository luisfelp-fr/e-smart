"""Carimbo de origem de um resultado: o que foi analisado, com qual código.

Sem isso, um relatório impresso é um número sem passado: não dá para saber
qual planilha o gerou nem qual versão do algoritmo o calculou. Como o ranking
muda quando o método muda, dois relatórios da mesma base podem discordar
legitimamente — e sem carimbo ninguém consegue explicar por quê.

O que fica registrado:

- **versão do algoritmo**: commit do repositório (ou ``ESMART_VERSION``);
- **identidade da base**: nome do arquivo e SHA-256 do conteúdo — dois
  arquivos com o mesmo hash são o mesmo dado, independente do nome;
- **quando** a análise rodou e **com quais parâmetros**.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import os
import subprocess

_VERSION_FALLBACK = "desconhecida"


@functools.lru_cache(maxsize=1)
def app_version() -> str:
    """Versão do algoritmo: ``ESMART_VERSION``, ou o commit do repositório."""
    env = os.environ.get("ESMART_VERSION")
    if env:
        return env.strip()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return _VERSION_FALLBACK


@functools.lru_cache(maxsize=16)
def file_digest(path: str, _mtime: float | None = None) -> str:
    """SHA-256 do conteúdo do arquivo, em blocos (não carrega tudo na RAM).

    ``_mtime`` entra só para invalidar o cache quando o arquivo muda.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for bloco in iter(lambda: f.read(1024 * 1024), b""):
                h.update(bloco)
    except OSError:
        return "indisponível"
    return h.hexdigest()


def digest_of(path: str | None) -> str:
    """SHA-256 do arquivo, revalidado quando o conteúdo muda."""
    if not path:
        return "indisponível"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return "indisponível"
    return file_digest(path, mtime)


def stamp(
    file_path: str | None = None,
    file_name: str | None = None,
    params: dict | None = None,
) -> dict:
    """Carimbo de proveniência para anexar a um relatório."""
    digest = digest_of(file_path)
    return {
        "versao_algoritmo": app_version(),
        "base": file_name or (os.path.basename(file_path) if file_path else "—"),
        "sha256": digest,
        "sha256_curto": digest[:16] if digest != "indisponível" else digest,
        "gerado_em": dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "parametros": params or {},
    }


def stamp_lines(st: dict) -> list[str]:
    """Carimbo em linhas de texto, para cabeçalho de relatório."""
    linhas = [
        f"Base analisada: {st['base']}",
        f"SHA-256 do conteúdo: {st['sha256_curto']}…",
        f"Versão do algoritmo: {st['versao_algoritmo']}",
        f"Gerado em: {st['gerado_em']}",
    ]
    if st.get("parametros"):
        itens = ", ".join(f"{k}={v}" for k, v in st["parametros"].items())
        linhas.append(f"Parâmetros: {itens}")
    return linhas
