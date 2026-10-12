"""Fase 3: ferramentas do agente. Funções pequenas e testáveis isoladamente.

- buscar_trechos:      busca híbrida (vetorial + BM25) combinada por Reciprocal Rank Fusion
- ler_vizinhos:        trecho anterior e seguinte (mesmo documento)
- listar_documentos:   catálogo do corpus
"""
import json
from functools import lru_cache

from rank_bm25 import BM25Okapi

from src.config import CHROMA_DIR, CHUNKS_PATH, COLECAO, METADATA_CSV, tokenizar


@lru_cache(maxsize=1)
def _trechos() -> dict[str, dict]:
    """Todos os trechos por id, na ordem do documento (lidos de chunks.jsonl)."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("data/chunks.jsonl não existe. Rode primeiro: python src/ingest.py")
    linhas = CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
    return {t["id"]: t for t in map(json.loads, linhas)}


@lru_cache(maxsize=1)
def _ordem_por_doc() -> dict[str, list[str]]:
    ordem: dict[str, list[str]] = {}
    for t in _trechos().values():
        ordem.setdefault(t["doc_id"], []).append(t["id"])
    return ordem


@lru_cache(maxsize=1)
def _bm25() -> tuple[BM25Okapi, list[str]]:
    ids = list(_trechos().keys())
    corpus = [tokenizar(_trechos()[i]["texto"]) for i in ids]
    return BM25Okapi(corpus), ids


@lru_cache(maxsize=1)
def _store():
    from langchain_chroma import Chroma

    from src.embeddings import E5Embeddings

    return Chroma(collection_name=COLECAO, embedding_function=E5Embeddings(), persist_directory=str(CHROMA_DIR))


def rrf(listas: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: combina rankings sem precisar comparar scores de escalas diferentes."""
    pontos: dict[str, float] = {}
    for lista in listas:
        for pos, trecho_id in enumerate(lista):
            pontos[trecho_id] = pontos.get(trecho_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(pontos, key=pontos.get, reverse=True)


def _passa_filtro(trecho: dict, doc_id: str | None, ano: int | None) -> bool:
    return (doc_id is None or trecho["doc_id"] == doc_id) and (ano is None or trecho["ano"] == ano)


def _filtro_chroma(doc_id: str | None, ano: int | None) -> dict | None:
    condicoes = []
    if doc_id:
        condicoes.append({"doc_id": doc_id})
    if ano:
        condicoes.append({"ano": ano})
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


def busca_vetorial(consulta: str, n: int = 20, doc_id: str | None = None, ano: int | None = None) -> list[str]:
    docs = _store().similarity_search(consulta, k=n, filter=_filtro_chroma(doc_id, ano))
    return [d.metadata["id"] for d in docs]


def busca_bm25(consulta: str, n: int = 20, doc_id: str | None = None, ano: int | None = None) -> list[str]:
    bm25, ids = _bm25()
    scores = bm25.get_scores(tokenizar(consulta))
    ranking = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    saida = []
    for i in ranking:
        if scores[i] <= 0:
            break
        if _passa_filtro(_trechos()[ids[i]], doc_id, ano):
            saida.append(ids[i])
            if len(saida) == n:
                break
    return saida


def buscar_trechos(
    consulta: str,
    k: int = 6,
    doc_id: str | None = None,
    ano: int | None = None,
    modo: str = "hibrida",
) -> list[dict]:
    """modo: 'hibrida' (padrão), 'vetorial' ou 'bm25' (os dois últimos servem de baseline na avaliação)."""
    if modo == "vetorial":
        ids = busca_vetorial(consulta, 20, doc_id, ano)
    elif modo == "bm25":
        ids = busca_bm25(consulta, 20, doc_id, ano)
    else:
        ids = rrf([busca_vetorial(consulta, 20, doc_id, ano), busca_bm25(consulta, 20, doc_id, ano)])
    return [_trechos()[i] for i in ids[:k]]


def ler_vizinhos(trecho_id: str) -> dict:
    """Devolve {'anterior': trecho|None, 'seguinte': trecho|None} dentro do mesmo documento."""
    t = _trechos()[trecho_id]
    ordem = _ordem_por_doc()[t["doc_id"]]
    pos = ordem.index(trecho_id)
    return {
        "anterior": _trechos()[ordem[pos - 1]] if pos > 0 else None,
        "seguinte": _trechos()[ordem[pos + 1]] if pos < len(ordem) - 1 else None,
    }


def listar_documentos() -> list[dict]:
    """Catálogo: doc_id, título, autores, ano (de metadata.csv, só dos documentos indexados)."""
    import csv

    indexados = set(_ordem_por_doc())
    with open(METADATA_CSV, encoding="utf-8") as f:
        return [
            {"doc_id": l["arquivo"].removesuffix(".pdf"), "titulo": l["titulo"],
             "autores": l["autores"], "ano": int(l["ano"])}
            for l in csv.DictReader(f)
            if l["arquivo"].removesuffix(".pdf") in indexados
        ]
