"""Fase 4: o grafo do agente em LangGraph.

classificar -> (direto) responder_direto
            -> (pesquisa) planejar -> buscar -> avaliar -> (insuficiente) reescrever -> buscar ...
                                                        -> (suficiente ou limite) [comparar] -> sintetizar -> verificar

Parâmetros em tempo de execução (UI e avaliação): config={"configurable": {"k": 6, "modo": "hibrida",
"max_tentativas": 3, "verificar_citacoes": True}}.
"""
import sys
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import citations  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.tools import buscar_trechos, listar_documentos  # noqa: E402

MAX_TRECHOS_RESPOSTA = 10
MIN_RELEVANTES_POR_CONSULTA = 2


class Estado(TypedDict, total=False):
    pergunta: str
    rota: str                      # "pesquisa" | "direto"
    tipo: str                      # "factual" | "comparacao"
    doc_id: str | None             # filtro quando a pergunta cita um documento específico
    consultas: list[str]
    hits: dict[str, list[str]]     # consulta -> ids de trechos encontrados
    trechos: dict[str, dict]       # id -> trecho (acumula entre tentativas)
    notas: dict[str, int]          # id -> nota de relevância 0..2
    relevantes: list[str]          # ids aprovados, ordenados
    rotulos: dict[str, str]        # "S1" -> id do trecho
    suficiente: bool
    tentativas: int
    comparacao: str
    resposta: dict


# ---------- schemas de saída estruturada ----------
class Rota(BaseModel):
    rota: Literal["pesquisa", "direto"]
    motivo: str


class Plano(BaseModel):
    consultas: list[str] = Field(description="1 a 4 subconsultas de busca, em inglês")
    doc_id: str | None = Field(description="doc_id do catálogo se a pergunta cita UM documento específico; senão null")
    tipo: Literal["factual", "comparacao"]


class NotaTrecho(BaseModel):
    rotulo: str
    nota: int = Field(description="0 = irrelevante, 1 = parcialmente útil, 2 = responde diretamente")


class Avaliacao(BaseModel):
    notas: list[NotaTrecho]


class Reescrita(BaseModel):
    consultas: list[str] = Field(description="Uma consulta nova para cada consulta fraca, na mesma ordem")


def _cfg(config: RunnableConfig) -> dict:
    c = (config or {}).get("configurable", {})
    return {
        "k": c.get("k", 6),
        "modo": c.get("modo", "hibrida"),
        "max_tentativas": c.get("max_tentativas", 3),
        "verificar_citacoes": c.get("verificar_citacoes", True),
    }


def _estruturado(papel: str, schema, prompt: str):
    """Chama o LLM com saída estruturada; tenta de novo uma vez se o formato vier inválido."""
    llm = get_llm(papel).with_structured_output(schema)
    try:
        return llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        if "429" in str(exc) or "rate" in str(exc).lower() or "503" in str(exc):
            raise
        return llm.invoke(prompt + "\n\nResponda estritamente no formato pedido.")


def _catalogo() -> str:
    return "\n".join(f"- {d['doc_id']} | {d['titulo']} | {d['autores']} | {d['ano']}" for d in listar_documentos())


# ---------- nós ----------
def classificar(estado: Estado) -> Estado:
    prompt = (
        "Decida se a pergunta do usuário precisa de pesquisa num corpus de documentos sobre STPA "
        "(análise de segurança de sistemas), engenharia de requisitos e segurança de sistemas críticos, "
        "ou se é uma saudação, conversa casual ou assunto totalmente fora desse escopo.\n"
        "Use 'pesquisa' sempre que a pergunta possa ser respondida (mesmo parcialmente) com esses documentos.\n\n"
        f"Pergunta: {estado['pergunta']}"
    )
    r = _estruturado("fast", Rota, prompt)
    return {"rota": r.rota, "tentativas": 0}


def planejar(estado: Estado) -> Estado:
    prompt = (
        "Você planeja a busca num corpus de documentos (a maioria em inglês). Quebre a pergunta em 1 a 4 "
        "subconsultas curtas e independentes, escritas em inglês técnico, mantendo siglas (STPA, UCA, "
        "hazard, control structure...). Pergunta simples = 1 subconsulta. Uma subconsulta por documento "
        "ou aspecto quando a pergunta compara coisas.\n"
        "tipo='comparacao' só se a pergunta pede para comparar/contrastar documentos, abordagens ou autores.\n"
        "doc_id: preencha só se a pergunta cita explicitamente UM documento do catálogo (ex.: 'o Handbook'); "
        "caso contrário, null.\n\n"
        f"Catálogo (doc_id | título | autores | ano):\n{_catalogo()}\n\n"
        f"Pergunta: {estado['pergunta']}"
    )
    p = _estruturado("fast", Plano, prompt)
    validos = {d["doc_id"] for d in listar_documentos()}
    consultas = [c.strip() for c in p.consultas if c.strip()][:4] or [estado["pergunta"]]
    return {
        "consultas": consultas,
        "tipo": p.tipo,
        "doc_id": p.doc_id if p.doc_id in validos else None,
    }


def buscar(estado: Estado, config: RunnableConfig) -> Estado:
    c = _cfg(config)
    trechos = dict(estado.get("trechos", {}))
    hits = {}
    for consulta in estado["consultas"]:
        achados = buscar_trechos(consulta, k=c["k"], doc_id=estado.get("doc_id"), modo=c["modo"])
        hits[consulta] = [t["id"] for t in achados]
        trechos.update({t["id"]: t for t in achados})
    return {"hits": hits, "trechos": trechos}


def avaliar(estado: Estado, config: RunnableConfig) -> Estado:
    """Nota de relevância (em lote, uma chamada) só para trechos ainda não avaliados."""
    c = _cfg(config)
    notas = dict(estado.get("notas", {}))
    novos = [i for i in dict.fromkeys(i for ids in estado["hits"].values() for i in ids) if i not in notas]
    if novos:
        rotulo_para_id = {f"T{n}": i for n, i in enumerate(novos, start=1)}
        blocos = "\n\n".join(
            f"[{r}] {estado['trechos'][i]['texto'][:450]}" for r, i in rotulo_para_id.items()
        )
        prompt = (
            "Avalie a relevância de cada trecho para responder à pergunta. nota 0 = irrelevante; "
            "1 = parcialmente útil; 2 = responde diretamente. Seja criterioso: listas de referências "
            "bibliográficas, índices e trechos só tangenciais recebem 0.\n\n"
            f"Pergunta: {estado['pergunta']}\nSubconsultas: {estado['consultas']}\n\nTrechos:\n{blocos}"
        )
        av = _estruturado("fast", Avaliacao, prompt)
        dadas = {n.rotulo.strip("[]").upper(): max(0, min(2, n.nota)) for n in av.notas}
        for r, i in rotulo_para_id.items():
            notas[i] = dadas.get(r, 0)

    por_consulta_ok = {
        q: sum(1 for i in ids if notas.get(i, 0) >= 1) >= MIN_RELEVANTES_POR_CONSULTA
        for q, ids in estado["hits"].items()
    }
    relevantes = sorted((i for i, n in notas.items() if n >= 1), key=lambda i: (-notas[i], i))
    relevantes = relevantes[:MAX_TRECHOS_RESPOSTA]
    rotulos = {f"S{n}": i for n, i in enumerate(relevantes, start=1)}
    return {
        "notas": notas,
        "suficiente": all(por_consulta_ok.values()),
        "relevantes": relevantes,
        "rotulos": rotulos,
    }


def reescrever(estado: Estado) -> Estado:
    notas = estado["notas"]
    fracas = [q for q, ids in estado["hits"].items()
              if sum(1 for i in ids if notas.get(i, 0) >= 1) < MIN_RELEVANTES_POR_CONSULTA]
    prompt = (
        "As buscas abaixo trouxeram poucos trechos relevantes. Reescreva cada uma para achar o que faltou: "
        "use sinônimos e termos técnicos alternativos em inglês, expanda siglas (ou use a sigla, se estava "
        "expandida) e mude o ângulo. Devolva exatamente uma nova consulta para cada consulta fraca, na mesma ordem.\n\n"
        f"Pergunta original: {estado['pergunta']}\nConsultas fracas: {fracas}"
    )
    r = _estruturado("fast", Reescrita, prompt)
    novas = dict(zip(fracas, (c.strip() for c in r.consultas if c.strip())))
    consultas = [novas.get(q, q) for q in estado["consultas"]]
    return {"consultas": consultas, "tentativas": estado.get("tentativas", 0) + 1}


def _texto_rotulado(estado: Estado) -> str:
    return "\n\n".join(
        f"[{r}] ({citations.formatar_referencia(estado['trechos'][i])})\n{estado['trechos'][i]['texto']}"
        for r, i in estado["rotulos"].items()
    )


def prompt_sintese(pergunta: str, trechos_rotulados: str, comparacao: str = "") -> str:
    """Prompt de síntese único: o agente e os baselines da avaliação usam exatamente o mesmo."""
    previa = f"\nAnálise comparativa prévia (use para estruturar a resposta):\n{comparacao}\n" if comparacao else ""
    return (
        "Responda à pergunta usando SOMENTE os trechos fornecidos. Regras:\n"
        "- Divida a resposta em afirmações curtas; cada uma com pelo menos um rótulo de fonte em 'fontes' "
        "(ex.: [\"S1\"]). Nunca cite um rótulo que não exista.\n"
        "- Não complete com conhecimento próprio. O que os trechos não cobrem vai em 'lacunas'.\n"
        "- Se dois documentos discordam, diga isso explicitamente.\n"
        "- Escreva na mesma língua da pergunta.\n\n"
        f"Pergunta: {pergunta}\n{previa}\nTrechos:\n{trechos_rotulados}"
    )


def comparar(estado: Estado) -> Estado:
    prompt = (
        "Com base APENAS nos trechos rotulados, agrupe o que cada documento diz sobre a pergunta e aponte "
        "concordâncias e divergências entre eles. Seja objetivo, cite os rótulos [S1], [S2]... Se um "
        "documento não trata do assunto, diga isso. Responda na língua da pergunta.\n\n"
        f"Pergunta: {estado['pergunta']}\n\nTrechos:\n{_texto_rotulado(estado)}"
    )
    return {"comparacao": get_llm("main").invoke(prompt).content}


def sintetizar(estado: Estado) -> Estado:
    if not estado.get("rotulos"):
        vazio = citations.Resposta(afirmacoes=[], lacunas=[f"O corpus não trouxe trechos relevantes para: {estado['pergunta']}"])
        return {"resposta": {"bruta": vazio.model_dump()}}
    prompt = prompt_sintese(estado["pergunta"], _texto_rotulado(estado), estado.get("comparacao", ""))
    r = _estruturado("main", citations.Resposta, prompt)
    return {"resposta": {"bruta": r.model_dump()}}


def verificar(estado: Estado, config: RunnableConfig) -> Estado:
    c = _cfg(config)
    bruta = citations.Resposta(**estado["resposta"]["bruta"])
    r = citations.verificar(bruta, estado.get("rotulos", {}), estado.get("trechos", {}), usar_juiz=c["verificar_citacoes"])
    r["texto"] = citations.renderizar(r)
    return {"resposta": r}


def responder_direto(estado: Estado) -> Estado:
    prompt = (
        "Responda brevemente e em tom cordial. Você é um assistente de pesquisa sobre STPA e segurança de sistemas; "
        "esta pergunta não precisa do corpus.\n\n" + estado["pergunta"]
    )
    texto = get_llm("fast").invoke(prompt).content
    return {"resposta": {"texto": texto + "\n\n*(Resposta direta, sem consulta ao corpus: não há fontes.)*",
                         "afirmacoes": [], "referencias": [], "removidas": [], "lacunas": [], "taxa_rejeitadas": 0.0}}


# ---------- arestas condicionais ----------
def _depois_de_avaliar(estado: Estado, config: RunnableConfig) -> str:
    if estado["suficiente"] or estado.get("tentativas", 0) >= _cfg(config)["max_tentativas"]:
        return "comparar" if estado.get("tipo") == "comparacao" and estado.get("rotulos") else "sintetizar"
    return "reescrever"


def construir_grafo():
    g = StateGraph(Estado)
    for nome, fn in [("classificar", classificar), ("planejar", planejar), ("buscar", buscar),
                     ("avaliar", avaliar), ("reescrever", reescrever), ("comparar", comparar),
                     ("sintetizar", sintetizar), ("verificar", verificar),
                     ("responder_direto", responder_direto)]:
        g.add_node(nome, fn)
    g.add_edge(START, "classificar")
    g.add_conditional_edges("classificar", lambda s: s["rota"],
                            {"pesquisa": "planejar", "direto": "responder_direto"})
    g.add_edge("planejar", "buscar")
    g.add_edge("buscar", "avaliar")
    g.add_conditional_edges("avaliar", _depois_de_avaliar,
                            {"comparar": "comparar", "sintetizar": "sintetizar", "reescrever": "reescrever"})
    g.add_edge("reescrever", "buscar")
    g.add_edge("comparar", "sintetizar")
    g.add_edge("sintetizar", "verificar")
    g.add_edge("verificar", END)
    g.add_edge("responder_direto", END)
    return g.compile()


app_grafo = construir_grafo()


def perguntar(pergunta: str, verbose: bool = True, **configurable) -> Estado:
    """Executa o agente e devolve o estado final. verbose imprime cada passo (útil no terminal)."""
    estado: Estado = {}
    for evento in app_grafo.stream({"pergunta": pergunta}, config={"configurable": configurable},
                                   stream_mode="updates"):
        for no, saida in evento.items():
            estado.update(saida)
            if no == "sintetizar":
                estado["bruta"] = saida["resposta"]["bruta"]  # antes da verificação (usado na avaliação)
            if verbose:
                extra = ""
                if no == "planejar":
                    extra = f" consultas={saida['consultas']} tipo={saida['tipo']} doc={saida['doc_id']}"
                elif no == "avaliar":
                    extra = f" suficiente={saida['suficiente']} relevantes={len(saida['relevantes'])}"
                elif no == "reescrever":
                    extra = f" novas={saida['consultas']}"
                elif no == "buscar":
                    extra = f" trechos={len(saida['trechos'])}"
                print(f"[{no}]{extra}", flush=True)
    return estado


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pergunta = " ".join(sys.argv[1:]) or "O que é uma unsafe control action no STPA?"
    final = perguntar(pergunta)
    print("\n" + final["resposta"]["texto"])
    print(f"\n(citações rejeitadas: {final['resposta']['taxa_rejeitadas']:.0%})")
