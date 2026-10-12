"""Fase 5: citações verificáveis.

O LLM só enxerga rótulos curtos ([S1], [S2]...). Aqui, em código:
  1. rótulos inexistentes são descartados (citação inventada);
  2. um juiz barato confere se o trecho citado sustenta a frase;
  3. frases sem sustentação saem da resposta;
  4. as referências finais são montadas a partir dos metadados, nunca do texto do LLM.
"""
from pydantic import BaseModel, Field

from src.llm import get_llm


class Afirmacao(BaseModel):
    texto: str
    fontes: list[str] = Field(description='Rótulos dos trechos que sustentam a frase, ex.: ["S1", "S3"]')


class Resposta(BaseModel):
    afirmacoes: list[Afirmacao]
    lacunas: list[str] = Field(description="O que o corpus não permitiu responder")


class Veredicto(BaseModel):
    indice: int
    sustentada: bool


class Julgamento(BaseModel):
    veredictos: list[Veredicto]


def normalizar_rotulo(r: str) -> str:
    return r.strip().strip("[]").upper()


def _sobrenome(autor: str) -> str:
    partes = autor.replace("et al.", "").split()
    return partes[-1] if partes else ""


def formatar_autores(autores: str) -> str:
    lista = [a.strip() for a in autores.split(";") if a.strip()]
    if len(lista) == 1 and "et al." not in lista[0]:
        return _sobrenome(lista[0])
    if len(lista) == 2 and "et al." not in autores:
        return f"{_sobrenome(lista[0])} e {_sobrenome(lista[1])}"
    return f"{_sobrenome(lista[0])} et al."


def formatar_referencia(trecho: dict) -> str:
    return f"{formatar_autores(trecho['autores'])} ({trecho['ano']}), {trecho['titulo']}, p. {trecho['pagina']}"


def julgar_sustentacao(afirmacoes: list[dict], rotulos: dict[str, str], trechos: dict[str, dict]) -> list[bool]:
    """Uma única chamada em lote: para cada frase, os trechos citados sustentam o que ela diz?"""
    blocos = []
    for i, a in enumerate(afirmacoes):
        fontes = "\n".join(f"- {trechos[rotulos[r]]['texto'][:700]}" for r in a["fontes"])
        blocos.append(f"### Frase {i}\n{a['texto']}\nTrechos citados:\n{fontes}")
    prompt = (
        "Você é um verificador rigoroso de citações. Para cada frase, diga se os trechos citados "
        "sustentam o que ela afirma (os trechos podem estar em inglês; a frase pode estar em português). "
        "Marque sustentada=false se a frase diz algo que não está nos trechos ou os contradiz. "
        "Responda com um veredicto por frase, usando o número da frase como índice.\n\n" + "\n\n".join(blocos)
    )
    resultado = get_llm("fast").with_structured_output(Julgamento).invoke(prompt)
    por_indice = {v.indice: v.sustentada for v in resultado.veredictos}
    return [por_indice.get(i, False) for i in range(len(afirmacoes))]


def verificar(
    resposta: Resposta,
    rotulos: dict[str, str],
    trechos: dict[str, dict],
    usar_juiz: bool = True,
) -> dict:
    """Devolve a resposta verificada, as referências numeradas e as métricas de rejeição.

    rotulos: {"S1": id_do_trecho}. trechos: {id: trecho}.
    """
    candidatas, removidas = [], []
    for a in resposta.afirmacoes:
        validos = list(dict.fromkeys(normalizar_rotulo(r) for r in a.fontes if normalizar_rotulo(r) in rotulos))
        if validos:
            candidatas.append({"texto": a.texto.strip(), "fontes": validos})
        else:
            removidas.append({"texto": a.texto.strip(), "motivo": "citação inexistente ou ausente"})

    aviso_juiz = None
    if usar_juiz and candidatas:
        try:
            veredictos = julgar_sustentacao(candidatas, rotulos, trechos)
        except Exception as exc:  # noqa: BLE001 - sem juiz, mantemos as frases marcadas como não verificadas
            veredictos = [None] * len(candidatas)
            aviso_juiz = f"juiz indisponível ({type(exc).__name__}); frases marcadas como não verificadas"
    else:
        veredictos = [None] * len(candidatas)

    mantidas = []
    for a, v in zip(candidatas, veredictos):
        if v is False:
            removidas.append({"texto": a["texto"], "motivo": "trecho citado não sustenta a frase"})
        else:
            a["status"] = "verificada" if v else "não verificada"
            mantidas.append(a)

    # Numeração final [1], [2]... na ordem de aparição; referências vêm dos metadados.
    numero: dict[str, int] = {}
    referencias = []
    for a in mantidas:
        a["numeros"] = []
        for r in a["fontes"]:
            if r not in numero:
                numero[r] = len(numero) + 1
                t = trechos[rotulos[r]]
                referencias.append({
                    "numero": numero[r], "rotulo": r, "trecho_id": t["id"], "doc_id": t["doc_id"],
                    "pagina": t["pagina"], "referencia": formatar_referencia(t), "texto": t["texto"],
                })
            a["numeros"].append(numero[r])

    total = len(candidatas) + sum(1 for x in removidas if x["motivo"].startswith("citação inexistente"))
    return {
        "afirmacoes": mantidas,
        "removidas": removidas,
        "lacunas": resposta.lacunas,
        "referencias": referencias,
        "taxa_rejeitadas": (len(removidas) / total) if total else 0.0,
        "aviso": aviso_juiz,
    }


def renderizar(r: dict) -> str:
    """Markdown final: frases com marcadores [n], lacunas e lista de referências."""
    linhas = []
    for a in r["afirmacoes"]:
        marcas = "".join(f"[{n}]" for n in a["numeros"])
        sufixo = "" if a["status"] == "verificada" else " *(não verificada)*"
        linhas.append(f"{a['texto']} {marcas}{sufixo}")
    texto = "\n\n".join(linhas) if linhas else "Não encontrei no corpus informação suficiente para responder."
    if r["lacunas"]:
        texto += "\n\n**Lacunas (o corpus não cobre):**\n" + "\n".join(f"- {l}" for l in r["lacunas"])
    if r["referencias"]:
        texto += "\n\n**Referências:**\n" + "\n".join(f"[{x['numero']}] {x['referencia']}" for x in r["referencias"])
    return texto
