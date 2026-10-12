"""Embeddings locais (sem custo, sem chave). O e5 exige prefixos 'query: ' e 'passage: '."""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class E5Embeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.modelo = SentenceTransformer(model_name)

    def _codificar(self, textos: list[str], prefixo: str) -> list[list[float]]:
        vetores = self.modelo.encode(
            [prefixo + t for t in textos], normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vetores.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._codificar(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._codificar([text], "query: ")[0]
