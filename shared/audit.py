"""Trilha de auditoria: registro do que rodou, quando, sobre qual base.

Escreve uma linha JSON por evento (JSON Lines), formato que qualquer
ferramenta de log consome sem parser próprio. Serve para responder depois
"quem rodou esta análise e sobre qual dado?" — pergunta que aparece quando um
resultado é contestado.

Fica **desligado por padrão**: sem ``ESMART_AUDIT_LOG`` apontando para um
arquivo, nada é escrito. Ligue-o em ambiente corporativo.

Limite honesto: o campo ``usuario`` só é preenchido quando existe
autenticação na frente do app (via ``ESMART_USER_HEADER``/``st.user``). Sem
login, a trilha registra O QUE aconteceu, não QUEM fez — e uma trilha sem
identidade não sustenta auditoria de responsabilidade. É por isso que
autenticação vem antes de auditoria na ordem de implantação.

Falhas de escrita nunca derrubam a aplicação: log é observabilidade, não
funcionalidade.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading

AUDIT_LOG = os.environ.get("ESMART_AUDIT_LOG", "").strip()

_lock = threading.Lock()


def enabled() -> bool:
    """A trilha está ligada?"""
    return bool(AUDIT_LOG)


def _current_user() -> str | None:
    """Identidade do usuário, quando a hospedagem fornece uma."""
    try:
        import streamlit as st
    except ImportError:
        return None
    # Streamlit >= 1.42 expõe o usuário autenticado via OIDC em st.user
    try:
        user = getattr(st, "user", None)
        for campo in ("email", "name"):
            valor = user.get(campo) if hasattr(user, "get") else None
            if valor:
                return str(valor)
    except Exception:  # noqa: BLE001 - identidade é opcional, nunca bloqueia
        pass
    return None


def record(evento: str, **campos) -> None:
    """Registra um evento. Silencioso quando desligado; nunca levanta erro."""
    if not AUDIT_LOG:
        return
    linha = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "evento": evento,
        "usuario": _current_user(),
        **campos,
    }
    try:
        texto = json.dumps(linha, ensure_ascii=False, default=str)
        with _lock:  # uma linha por vez: o app é multithread
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(texto + "\n")
    except (OSError, TypeError, ValueError):
        pass  # observabilidade não pode quebrar a análise
