# -*- coding: utf-8 -*-
"""Bot keep-alive do app no Streamlit Community Cloud.

Visita o app com navegador headless (a hibernação do Streamlit é decidida por
sessões reais/websocket — ping HTTP simples não conta), clica no botão de
acordar se a tela "Zzzz" aparecer e só sai com sucesso quando o app renderiza.

Executado pelo workflow agendado .github/workflows/keep-alive.yml.
Uso local: APP_URL=http://localhost:8501 python .github/scripts/keep_alive.py
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://e-smart-analytics.streamlit.app/")
BOOT_TIMEOUT_MS = 240_000   # até 4 min para o app acordar e subir
RENDER_MARKER = '[data-testid="stApp"]'
WAKE_SELECTORS = (
    '[data-testid="wakeup-button-viewer"]',
    'button:has-text("Yes, get this app back up")',
    'button:has-text("app back up")',
)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(60_000)
        print(f"visitando {APP_URL} ...", flush=True)
        page.goto(APP_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(5_000)

        # tela de hibernação? clica para acordar
        for sel in WAKE_SELECTORS:
            btn = page.locator(sel)
            if btn.count():
                print("app hibernado — clicando para acordar...", flush=True)
                btn.first.click()
                break
        else:
            body = page.content().lower()
            if "gone to sleep" in body or "zzzz" in body:
                print("tela de hibernação sem botão reconhecido; aguardando...",
                      flush=True)

        # espera o Streamlit renderizar de fato
        try:
            page.wait_for_selector(RENDER_MARKER, timeout=BOOT_TIMEOUT_MS)
        except Exception:
            print("ERRO: o app não renderizou dentro do tempo limite.",
                  flush=True)
            page.screenshot(path="keep_alive_fail.png", full_page=True)
            browser.close()
            return 1

        # mantém a sessão aberta alguns segundos para registrar atividade
        page.wait_for_timeout(20_000)
        title = page.title()
        print(f"app ativo (título: {title!r}).", flush=True)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
