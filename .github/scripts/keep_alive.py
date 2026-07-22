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

        # tela de hibernação? clica para acordar (procura em qualquer elemento
        # clicável, inclusive dentro de iframes)
        def try_wake(target) -> bool:
            for sel in WAKE_SELECTORS:
                btn = target.locator(sel)
                if btn.count():
                    print("app hibernado — clicando para acordar...", flush=True)
                    btn.first.click()
                    return True
            txt = target.get_by_text("get this app back up")
            if txt.count():
                print("app hibernado — clicando no texto de acordar...",
                      flush=True)
                txt.first.click()
                return True
            return False

        woke = try_wake(page)
        if not woke:
            for frame in page.frames[1:]:
                try:
                    if try_wake(frame):
                        woke = True
                        break
                except Exception:
                    continue

        # espera o Streamlit renderizar de fato
        try:
            page.wait_for_selector(RENDER_MARKER, timeout=BOOT_TIMEOUT_MS)
        except Exception:
            print("ERRO: o app não renderizou dentro do tempo limite.",
                  flush=True)
            # diagnóstico para o log do workflow
            try:
                print(f"URL final: {page.url}", flush=True)
                print(f"título: {page.title()!r}", flush=True)
                botoes = page.locator("button").all_inner_texts()
                print(f"botões na página: {botoes[:10]}", flush=True)
                corpo = " ".join(page.inner_text("body").split())
                print(f"corpo (600c): {corpo[:600]!r}", flush=True)
                print(f"iframes: {[f.url for f in page.frames]}", flush=True)
            except Exception as diag_err:  # diagnóstico é melhor-esforço
                print(f"(diagnóstico indisponível: {diag_err})", flush=True)
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
