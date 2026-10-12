"""Fábrica de LLMs com provedor configurável, cache em disco e retry em rate limit.

O provedor é escolhido pelo .env (LLM_PROVIDER), então trocar de Groq para Gemini
ou Ollama não exige mexer no resto do código.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"
MAX_RETRIES = 5


def _modelo_base(provider: str, papel: str, temperature: float) -> BaseChatModel:
    """Cria o modelo bruto. papel: 'main' (sintetizar/comparar), 'fast' (demais nós) ou 'judge'."""
    if provider == "groq":
        from langchain_groq import ChatGroq

        nome = os.getenv("GROQ_MODEL_MAIN" if papel == "main" else "GROQ_MODEL_FAST")
        return ChatGroq(model=nome, temperature=temperature, max_retries=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            max_retries=0,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), temperature=temperature)
    raise ValueError(f"LLM_PROVIDER desconhecido: {provider!r} (use groq, gemini ou ollama)")


def _eh_rate_limit(exc: Exception) -> bool:
    texto = f"{type(exc).__name__} {exc}".lower()
    sinais = ("429", "rate limit", "rate_limit", "resource_exhausted", "too many requests", "503", "unavailable", "overloaded")
    return any(s in texto for s in sinais)


def _espera_sugerida(exc: Exception, padrao: float) -> float:
    """Groq informa o tempo de espera na mensagem ('try again in 7.5s' / '350ms' / '1m2.5s')."""
    m = re.search(r"try again in (?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", str(exc))
    if not m or not (m.group(1) or m.group(2)):
        return padrao
    seg = int(m.group(1) or 0) * 60
    if m.group(2):
        seg += float(m.group(2)) / (1000 if m.group(3) == "ms" else 1)
    return min(seg + 1.0, 90.0)


def _chave(provider: str, modelo: str, mensagens: list[BaseMessage]) -> str:
    conteudo = json.dumps([[m.type, m.content] for m in mensagens], ensure_ascii=False)
    return hashlib.sha256(f"{provider}|{modelo}|{conteudo}".encode()).hexdigest()


class LLMComCache:
    """Envolve um chat model com cache em disco (por prompt) e retry com backoff.

    Expõe `invoke(...)` e `with_structured_output(...)` como o modelo original. O cache só vale para
    chamadas determinísticas (temperature=0): repetir um teste não gasta a cota gratuita.
    """

    def __init__(self, modelo: BaseChatModel, provider: str, usar_cache: bool = True):
        self.modelo = modelo
        self.provider = provider
        self.usar_cache = usar_cache
        self.nome = getattr(modelo, "model_name", None) or getattr(modelo, "model", "desconhecido")

    def _com_retry(self, fn):
        for tentativa in range(MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - só repetimos erros de rate limit
                if not _eh_rate_limit(exc) or tentativa == MAX_RETRIES - 1:
                    raise
                espera = _espera_sugerida(exc, min(2 ** tentativa * 3, 60))
                print(f"[llm] limite/indisponibilidade ({self.provider}); aguardando {espera:.0f}s...", flush=True)
                time.sleep(espera)

    def invoke(self, entrada, **kwargs) -> BaseMessage:
        mensagens = self.modelo._convert_input(entrada).to_messages()
        arquivo = CACHE_DIR / f"{_chave(self.provider, str(self.nome), mensagens)}.json"
        if self.usar_cache and arquivo.exists():
            return AIMessage(content=json.loads(arquivo.read_text(encoding="utf-8"))["content"])
        resposta = self._com_retry(lambda: self.modelo.invoke(mensagens, **kwargs))
        if self.usar_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            arquivo.write_text(json.dumps({"content": resposta.content}, ensure_ascii=False), encoding="utf-8")
        return resposta

    def with_structured_output(self, schema):
        """Saída estruturada (Pydantic). Sem cache em disco, mas com retry em rate limit."""
        # No Groq, o tool calling dos modelos gpt-oss é instável (chama ferramenta inexistente); json_schema é estável.
        if self.provider == "groq":
            estruturado = self.modelo.with_structured_output(schema, method="json_schema")
        else:
            estruturado = self.modelo.with_structured_output(schema)

        class _Estruturado:
            def invoke(inner, entrada, **kwargs):  # noqa: N805
                return self._com_retry(lambda: estruturado.invoke(entrada, **kwargs))

        return _Estruturado()


def get_llm(papel: str = "main", temperature: float = 0.0, provider: str | None = None) -> LLMComCache:
    """papel: 'main' | 'fast' | 'judge'. 'judge' usa Gemini quando houver chave; senão cai no provedor padrão."""
    provider = provider or os.getenv("LLM_PROVIDER", "groq")
    if papel == "judge" and os.getenv("GEMINI_API_KEY"):
        provider = "gemini"
    return LLMComCache(_modelo_base(provider, papel, temperature), provider)
