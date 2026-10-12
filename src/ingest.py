"""Fase 2: PDFs -> texto por página -> limpeza -> trechos -> Chroma + chunks.jsonl.

Uso: python src/ingest.py            (reindexa tudo; os IDs são estáveis, então não duplica)
     python src/ingest.py --stats    (só mostra estatísticas do que já foi indexado)
"""
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pymupdf  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, COLECAO, METADATA_CSV, RAW,
)


def carregar_metadados() -> dict[str, dict]:
    with open(METADATA_CSV, encoding="utf-8") as f:
        return {linha["arquivo"]: linha for linha in csv.DictReader(f)}


def extrair_paginas(pdf: Path) -> list[tuple[int, str]]:
    """Texto bruto por página, guardando o número da página (começa em 1)."""
    with pymupdf.open(pdf) as doc:
        return [(i, p.get_text()) for i, p in enumerate(doc, start=1)]


def _linhas_repetidas(paginas: list[tuple[int, str]]) -> set[str]:
    """Linhas curtas que aparecem em muitas páginas (cabeçalhos e rodapés)."""
    if len(paginas) < 6:
        return set()
    contagem: Counter[str] = Counter()
    for _, texto in paginas:
        linhas = {re.sub(r"\d+", "#", l.strip()) for l in texto.splitlines() if 3 < len(l.strip()) < 90}
        contagem.update(linhas)
    limite = max(4, int(len(paginas) * 0.3))
    return {l for l, n in contagem.items() if n >= limite}


def limpar(texto: str, repetidas: set[str]) -> str:
    texto = texto.replace("\x00", " ").replace("­", "")
    linhas = []
    for linha in texto.splitlines():
        s = linha.strip()
        if re.sub(r"\d+", "#", s) in repetidas:
            continue
        if re.fullmatch(r"(página |page |p\. )?\d{1,4}", s.lower()):  # número de página solto
            continue
        linhas.append(s)
    texto = "\n".join(linhas)
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)       # palavra hifenizada na quebra de linha
    texto = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", texto)  # junta linhas de um mesmo parágrafo
    texto = re.sub(r"[ \t]+", " ", texto)
    return re.sub(r"\n{3,}", "\n\n", texto).strip()


def gerar_trechos(pdf: Path, meta: dict) -> list[dict]:
    """Trechos nunca atravessam páginas, então a página citada é sempre exata."""
    divisor = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    paginas = extrair_paginas(pdf)
    repetidas = _linhas_repetidas(paginas)
    doc_id = pdf.stem
    trechos = []
    for pagina, bruto in paginas:
        texto = limpar(bruto, repetidas)
        if len(texto) < 40:
            continue
        for c, parte in enumerate(divisor.split_text(texto), start=1):
            trechos.append({
                "id": f"{doc_id}:p{pagina}:c{c}",
                "texto": parte,
                "doc_id": doc_id,
                "titulo": meta["titulo"],
                "autores": meta["autores"],
                "ano": int(meta["ano"]),
                "pagina": pagina,
            })
    return trechos


def indexar_chroma(trechos: list[dict]) -> None:
    from langchain_chroma import Chroma

    from src.embeddings import E5Embeddings

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # reconstrução limpa: evita lixo de execuções anteriores
    store = Chroma(collection_name=COLECAO, embedding_function=E5Embeddings(), persist_directory=str(CHROMA_DIR))
    lote = 128
    for i in range(0, len(trechos), lote):
        parte = trechos[i:i + lote]
        store.add_texts(
            texts=[t["texto"] for t in parte],
            metadatas=[{k: v for k, v in t.items() if k != "texto"} for t in parte],
            ids=[t["id"] for t in parte],
        )
        print(f"  indexados {min(i + lote, len(trechos))}/{len(trechos)}")


def estatisticas() -> None:
    trechos = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    por_doc = Counter(t["doc_id"] for t in trechos)
    tamanhos = sorted(len(t["texto"]) for t in trechos)
    print(f"{len(trechos)} trechos em {len(por_doc)} documentos; "
          f"tamanho mediano {tamanhos[len(tamanhos) // 2]} caracteres")
    for doc, n in por_doc.most_common():
        print(f"  {n:5d}  {doc}")


def main() -> None:
    if "--stats" in sys.argv:
        return estatisticas()
    metadados = carregar_metadados()
    todos: list[dict] = []
    for arquivo, meta in metadados.items():
        pdf = RAW / arquivo
        if not pdf.exists():
            print(f"[pulado] {arquivo} não está em data/raw/")
            continue
        trechos = gerar_trechos(pdf, meta)
        if not trechos:
            print(f"[vazio]  {arquivo}: sem texto extraível (PDF escaneado? precisa de OCR)")
            continue
        print(f"[ok]     {arquivo}: {len(trechos)} trechos")
        todos.extend(trechos)
    CHUNKS_PATH.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in todos), encoding="utf-8")
    print(f"\n{len(todos)} trechos gravados em {CHUNKS_PATH.name}. Gerando embeddings (CPU, pode levar alguns minutos)...")
    indexar_chroma(todos)
    print("Pronto. Teste com: python src/ingest.py --stats")


if __name__ == "__main__":
    main()
