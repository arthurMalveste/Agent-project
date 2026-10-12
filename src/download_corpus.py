"""Baixa os PDFs listados em data/metadata.csv (coluna url) para data/raw/.

Uso: python src/download_corpus.py
Só aceita respostas que realmente sejam PDF (cabeçalho %PDF). Documentos atrás de bloqueio anti-bot
(ou pagos) não são baixados: o script lista o que falta para você baixar manualmente.
"""
import csv
import sys
import time
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
CABECALHOS = {"User-Agent": "Mozilla/5.0 (compatible; agente-rag-corpus/0.1)"}


def baixar(url: str, destino: Path) -> bool:
    resp = requests.get(url, headers=CABECALHOS, timeout=90, allow_redirects=True)
    if resp.status_code != 200 or not resp.content.startswith(b"%PDF"):
        return False
    destino.write_bytes(resp.content)
    return True


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAIZ / "data" / "metadata.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    faltando = []
    for linha in linhas:
        destino = RAW / linha["arquivo"]
        if destino.exists() and destino.stat().st_size > 0:
            print(f"[ok]      {linha['arquivo']}")
            continue
        try:
            ok = baixar(linha["url"], destino)
        except requests.RequestException as exc:
            print(f"[erro]    {linha['arquivo']}: {exc}")
            ok = False
        if ok:
            print(f"[baixado] {linha['arquivo']}")
            time.sleep(3)  # gentileza com o servidor (arXiv pede ~1 requisição a cada 3 s)
        else:
            faltando.append(linha)
    if faltando:
        print("\nBaixe manualmente e salve em data/raw/ com o nome indicado:")
        for linha in faltando:
            print(f"  {linha['arquivo']}  <-  {linha['url']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
