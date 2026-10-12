"""Teste de conexão com os provedores configurados no .env. Uso: python teste_llm.py"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
from src.llm import get_llm  # noqa: E402

for papel in ("fast", "main", "judge"):
    chave_ok = bool(os.getenv("GEMINI_API_KEY" if papel == "judge" else "GROQ_API_KEY"))
    if not chave_ok:
        print(f"[{papel}] pulado: chave ausente no .env")
        continue
    try:
        llm = get_llm(papel)
        resp = llm.invoke("Responda apenas com uma frase curta: diga olá em português.")
        print(f"[{papel}] {llm.provider}/{llm.nome}: {resp.content.strip()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{papel}] FALHOU: {type(exc).__name__}: {str(exc)[:300]}")
