"""Caminhos e parâmetros compartilhados entre ingestão, ferramentas, grafo e avaliação."""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
RAW = DATA / "raw"
CHROMA_DIR = DATA / "chroma"
CHUNKS_PATH = DATA / "chunks.jsonl"
METADATA_CSV = DATA / "metadata.csv"

COLECAO = "corpus"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def tokenizar(texto: str) -> list[str]:
    """Tokenização simples para o BM25: minúsculas, letras/números (mantém siglas como UCA e 29148)."""
    return re.findall(r"\w+", texto.lower())
