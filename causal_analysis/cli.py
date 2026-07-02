"""Interface de linha de comando.

Exemplo:
    python -m causal_analysis dados.csv --alvo rendimento --max-lag 14 \\
        --janelas 3 7 14 --saida relatorio.html
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import DEFAULT_WINDOWS, run_analysis
from .report import render_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="causal_analysis",
        description=(
            "Analisa relações causa-efeito entre parâmetros e uma coluna-alvo "
            "em uma tabela temporal (1ª coluna com datas) e gera um relatório "
            "HTML automático apontando os prováveis 'culpados'."
        ),
    )
    p.add_argument("arquivo", help="Tabela de entrada (.csv, .xlsx, .xls)")
    p.add_argument(
        "--alvo", "--target", required=True, dest="alvo",
        help="Nome da coluna-alvo (resultado a explicar)",
    )
    p.add_argument(
        "--coluna-data", "--date-col", default=None, dest="coluna_data",
        help="Nome da coluna de datas (padrão: primeira coluna)",
    )
    p.add_argument(
        "--max-lag", type=int, default=14,
        help="Defasagem máxima testada, em períodos (padrão: 14)",
    )
    p.add_argument(
        "--janelas", "--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
        dest="janelas",
        help=f"Janelas das médias móveis (padrão: {' '.join(map(str, DEFAULT_WINDOWS))})",
    )
    p.add_argument(
        "--alfa", "--alpha", type=float, default=0.05, dest="alfa",
        help="Nível de significância com correção FDR (padrão: 0.05)",
    )
    p.add_argument(
        "--saida", "--output", default="relatorio_causal.html", dest="saida",
        help="Arquivo HTML de saída (padrão: relatorio_causal.html)",
    )
    p.add_argument(
        "--sep", default=None, help="Separador do CSV (padrão: autodetecta)"
    )
    p.add_argument(
        "--planilha", "--sheet", default=0, dest="planilha",
        help="Nome ou índice da planilha, para arquivos Excel (padrão: 0)",
    )
    p.add_argument(
        "--top-detalhe", type=int, default=5,
        help="Máximo de parâmetros com seção detalhada no relatório (padrão: 5)",
    )
    p.add_argument("--silencioso", action="store_true", help="Suprime o progresso")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sheet = int(args.planilha) if str(args.planilha).isdigit() else args.planilha
    if not args.silencioso:
        print(f"Analisando '{args.arquivo}' (alvo: {args.alvo})...")
    try:
        result = run_analysis(
            args.arquivo,
            target=args.alvo,
            date_col=args.coluna_data,
            max_lag=args.max_lag,
            windows=args.janelas,
            alpha=args.alfa,
            sep=args.sep,
            sheet=sheet,
            verbose=not args.silencioso,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    out = render_report(result, args.saida, top_detail=args.top_detalhe)
    if not args.silencioso:
        scores = result.scores
        culprits = scores[scores["veredito"].str.startswith("Culpado")]
        print(f"\nRelatório gravado em: {out}")
        if len(culprits):
            print("Principais suspeitos:")
            for rank, row in culprits.head(5).iterrows():
                print(
                    f"  {rank}. {row['parametro']} — score {row['score']} "
                    f"({row['veredito'].lower()}, relação {row['direcao_label']}, "
                    f"{row['melhor_transformacao']})"
                )
        else:
            print("Nenhum culpado com evidência estatística consistente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
