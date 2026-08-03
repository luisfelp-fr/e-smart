"""Componentes de interface compartilhados entre as páginas do app."""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import time

import numpy as np
import streamlit as st

from shared.limits import MAX_REPORT_ITEMS

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "e_smart_uploads")

# Uploads não são apagados por ninguém: sem esta varredura, o disco do
# contêiner só cresce (50 pessoas × planilhas de até 200 MB).
UPLOAD_TTL_S = float(os.environ.get("ESMART_UPLOAD_TTL_H", "24")) * 3600


def _sweep_old_uploads() -> None:
    """Remove uploads mais velhos que o TTL. Falhas aqui nunca quebram o app."""
    cutoff = time.time() - UPLOAD_TTL_S
    try:
        entries = os.scandir(UPLOAD_DIR)
    except OSError:
        return
    with entries:
        for entry in entries:
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
            except OSError:
                pass  # outro processo já removeu, ou permissão — irrelevante


def save_upload(uploaded_file) -> str:
    """Grava o upload em disco uma única vez e devolve o caminho.

    Os loaders trabalham com caminhos de arquivo; o conteúdo é identificado
    por hash para não regravar a cada rerun do Streamlit.

    Dois cuidados que só aparecem com mais de um usuário:

    - o Streamlit reexecuta o script a cada clique em qualquer widget, e ler
      + hashear a planilha inteira toda vez custa centenas de MB e frações de
      segundo por interação; o ``file_id`` do upload é estável, então serve de
      atalho guardado na sessão;
    - gravar com ``open(..., "wb")`` depois de checar ``exists()`` trunca o
      arquivo: uma segunda sessão com o mesmo conteúdo podia ler um arquivo
      pela metade. A gravação agora é em arquivo temporário + ``os.replace``,
      que é atômico — ou o caminho tem o conteúdo completo, ou não existe.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # atalho: mesmo upload desta sessão, já gravado e ainda em disco
    file_id = getattr(uploaded_file, "file_id", None)
    cached = st.session_state.get("_upload_cache")
    if file_id and cached and cached[0] == file_id and os.path.exists(cached[1]):
        return cached[1]

    _sweep_old_uploads()

    data = uploaded_file.getvalue()
    digest = hashlib.md5(data).hexdigest()[:16]
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".csv"
    path = os.path.join(UPLOAD_DIR, f"{digest}{ext}")
    if not os.path.exists(path):
        # nome temporário exclusivo por processo+thread: duas sessões gravando
        # o mesmo conteúdo não se atropelam, e o replace final é atômico
        tmp = f"{path}.{os.getpid()}.{id(uploaded_file):x}.part"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except OSError:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise

    if file_id:
        st.session_state["_upload_cache"] = (file_id, path)
    return path


def parse_br_number(text: str) -> float | None:
    """Converte "12,5" / "1.234,5" / "12.5" em float; None se vazio/inválido."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def fmt_br(v, nd: int = 2) -> str:
    """Formata número com vírgula decimal para exibição."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{nd}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def metric_row(pairs: list[tuple[str, str, str | None]], cols_per_row: int = 4) -> None:
    """Linha de st.metric com tooltip: [(rótulo, valor, ajuda), ...]."""
    for i in range(0, len(pairs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (label, value, help_txt) in zip(cols, pairs[i:i + cols_per_row]):
            with col:
                st.metric(label, value, help=help_txt)


def add_to_report_button(item: dict, key: str) -> None:
    """Botão "Adicionar ao relatório" que guarda o item na sessão."""
    items = st.session_state.setdefault("report_items", [])
    already = any(existing["id"] == item["id"] for existing in items)
    if already:
        st.success("✓ Esta análise já está no relatório.", icon="📄")
        return
    if len(items) >= MAX_REPORT_ITEMS:
        st.warning(
            f"O relatório atingiu o limite de {MAX_REPORT_ITEMS} análises. "
            "Baixe o PDF e limpe a lista para continuar adicionando.",
            icon="📄",
        )
        return
    if st.button("📄 Adicionar ao relatório", key=key, type="secondary"):
        # cópia das tabelas: o item fica guardado na sessão e não deve seguir
        # apontando para o DataFrame vivo do resultado da análise
        item = dict(item)
        item["tables"] = {
            k: (v.copy() if hasattr(v, "copy") else v)
            for k, v in (item.get("tables") or {}).items()
        }
        items.append(item)
        st.rerun()
