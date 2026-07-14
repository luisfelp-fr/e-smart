"""Componentes de interface compartilhados entre as páginas do app."""

from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import streamlit as st

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "e_smart_uploads")


def save_upload(uploaded_file) -> str:
    """Grava o upload em disco uma única vez e devolve o caminho.

    Os loaders trabalham com caminhos de arquivo; o conteúdo é identificado
    por hash para não regravar a cada rerun do Streamlit.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    data = uploaded_file.getvalue()
    digest = hashlib.md5(data).hexdigest()[:16]
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".csv"
    path = os.path.join(UPLOAD_DIR, f"{digest}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
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
    if st.button("📄 Adicionar ao relatório", key=key, type="secondary"):
        items.append(item)
        st.rerun()
