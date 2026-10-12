"""Fase 7: avaliação comparativa. Três etapas, todas retomáveis (pulam o que já está em evals/results/):

  python evals/run_evals.py rodar  --config A B C D     gera as respostas de cada configuração
  python evals/run_evals.py julgar --config A B C D     juiz independente (Gemini) avalia as respostas
  python evals/run_evals.py resumo                      tabelas e CSVs finais

Configurações:
  A  palavras-chave: só BM25 sobre a pergunta crua, sem decisões
  B  RAG simples: uma busca vetorial fixa
  C  RAG híbrido: vetorial + BM25 + RRF, sem decisões
  D  agente completo (grafo da Fase 4 com verificação de citações)
  D_sem_reescrita / D_sem_verificacao   ablações do agente

Se a cota diária gratuita acabar, o script para sozinho; rode o mesmo comando no dia seguinte.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("LLM_CACHE", "0")  # sem cache: latência e custo medidos são reais

from pydantic import BaseModel, Field  # noqa: E402

from src import citations, llm as llm_mod  # noqa: E402
from src.graph import _estruturado, perguntar, prompt_sintese  # noqa: E402
from src.llm import CotaDiariaEsgotada, get_llm  # noqa: E402
from src.tools import buscar_trechos  # noqa: E402

RESULTADOS = RAIZ / "evals" / "results"
RESPOSTAS = RESULTADOS / "respostas.jsonl"
JULGAMENTOS = RESULTADOS / "julgamentos.jsonl"
K = 6

CONFIGS = {
    "A": {"descricao": "Palavras-chave (BM25)", "tipo": "rag", "modo": "bm25"},
    "B": {"descricao": "RAG simples (vetorial)", "tipo": "rag", "modo": "vetorial"},
    "C": {"descricao": "RAG híbrido (RRF)", "tipo": "rag", "modo": "hibrida"},
    "D": {"descricao": "Agente completo", "tipo": "agente", "cfg": {}},
    "D_sem_reescrita": {"descricao": "Agente sem reescrita", "tipo": "agente", "cfg": {"max_tentativas": 0}},
    "D_sem_verificacao": {"descricao": "Agente sem juiz de citações", "tipo": "agente", "cfg": {"verificar_citacoes": False}},
}


# ---------------------------------------------------------------- utilidades
def ler_jsonl(caminho: Path) -> list[dict]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def anexar_jsonl(caminho: Path, registro: dict) -> None:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def carregar_perguntas(ids: list[str] | None) -> list[dict]:
    qs = ler_jsonl(RAIZ / "evals" / "questions.jsonl")
    return [q for q in qs if not ids or q["id"] in ids]


def trechos_por_id() -> dict[str, dict]:
    return {t["id"]: t for t in ler_jsonl(RAIZ / "data" / "chunks.jsonl")}


# ---------------------------------------------------------------- etapa 1: rodar
def responder_rag(pergunta: str, modo: str) -> dict:
    """Baselines A, B e C: uma busca fixa e uma síntese, com o mesmo prompt do agente e SEM verificação."""
    trechos = buscar_trechos(pergunta, k=K, modo=modo)
    rotulos = {f"S{n}": t["id"] for n, t in enumerate(trechos, start=1)}
    texto = "\n\n".join(
        f"[{r}] ({citations.formatar_referencia(t)})\n{t['texto']}" for r, t in zip(rotulos, trechos)
    )
    bruta = _estruturado("main", citations.Resposta, prompt_sintese(pergunta, texto))
    exibidas = [{"texto": a.texto, "fontes": [citations.normalizar_rotulo(f) for f in a.fontes]} for a in bruta.afirmacoes]
    return {
        "contexto_ids": [t["id"] for t in trechos],
        "rotulos": rotulos,
        "bruta": [{"texto": a["texto"], "fontes": a["fontes"]} for a in exibidas],
        "exibidas": exibidas,
        "lacunas": bruta.lacunas,
        "extras": {},
    }


def responder_agente(pergunta: str, cfg: dict) -> dict:
    estado = perguntar(pergunta, verbose=False, **cfg)
    if estado.get("rota") == "direto":  # o agente decidiu não consultar o corpus
        return {"contexto_ids": [], "rotulos": {}, "bruta": [], "exibidas": [], "lacunas": [],
                "extras": {"rota": "direto", "texto": estado["resposta"]["texto"]}}
    final = estado["resposta"]
    bruta = estado.get("bruta", {"afirmacoes": [], "lacunas": []})
    return {
        "contexto_ids": list(estado.get("rotulos", {}).values()),
        "rotulos": estado.get("rotulos", {}),
        "bruta": [{"texto": a["texto"], "fontes": [citations.normalizar_rotulo(f) for f in a["fontes"]]}
                  for a in bruta["afirmacoes"]],
        "exibidas": [{"texto": a["texto"], "fontes": a["fontes"]} for a in final["afirmacoes"]],
        "lacunas": final["lacunas"],
        "extras": {"rota": estado.get("rota"), "tentativas": estado.get("tentativas", 0),
                   "tipo_plano": estado.get("tipo"), "consultas": estado.get("consultas"),
                   "taxa_rejeitadas": final["taxa_rejeitadas"], "texto": final["texto"]},
    }


def cmd_rodar(args) -> int:
    feitos = {(r["config"], r["id"]) for r in ler_jsonl(RESPOSTAS) if not r.get("erro")}
    perguntas = carregar_perguntas(args.ids)
    for nome in args.config:
        conf = CONFIGS[nome]
        print(f"\n=== Configuração {nome}: {conf['descricao']} ===", flush=True)
        for q in perguntas:
            if (nome, q["id"]) in feitos:
                continue
            antes = llm_mod.ESTATISTICAS["chamadas"]
            t0 = time.time()
            registro = {"config": nome, "id": q["id"], "tipo": q["tipo"], "pergunta": q["pergunta"]}
            try:
                if conf["tipo"] == "rag":
                    registro.update(responder_rag(q["pergunta"], conf["modo"]))
                else:
                    registro.update(responder_agente(q["pergunta"], conf["cfg"]))
            except CotaDiariaEsgotada as exc:
                print(f"\nCota diária esgotada ({exc}).\nRode o mesmo comando amanhã: o progresso foi salvo.")
                return 2
            except Exception as exc:  # noqa: BLE001 - um erro numa pergunta não derruba a rodada
                registro["erro"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            registro["tempo_s"] = round(time.time() - t0, 1)
            registro["chamadas_llm"] = llm_mod.ESTATISTICAS["chamadas"] - antes
            anexar_jsonl(RESPOSTAS, registro)
            estado = "ERRO " + registro["erro"][:80] if registro.get("erro") else f"{len(registro['exibidas'])} afirmações"
            print(f"  {q['id']} [{q['tipo']}] {registro['tempo_s']}s, {registro['chamadas_llm']} chamadas: {estado}", flush=True)
    return 0


# ---------------------------------------------------------------- etapa 2: julgar
class Veredicto(BaseModel):
    indice: int
    sustentada: bool


class AvaliacaoSistema(BaseModel):
    sistema: int = Field(description="Número do sistema avaliado (1, 2, ...)")
    veredictos: list[Veredicto] = Field(description="Um por afirmação numerada desse sistema, na ordem")
    correcao: int = Field(description="0 = errada ou vazia, 1 = parcialmente correta, 2 = correta e completa em relação à resposta esperada")
    admite_lacuna: bool = Field(description="True se a resposta diz claramente que o corpus não traz a informação, sem inventar")


class JulgamentoLote(BaseModel):
    avaliacoes: list[AvaliacaoSistema]


def _bloco_sistema(n: int, reg: dict, chunks: dict[str, dict]) -> tuple[str, list[int]]:
    """Texto de um sistema para o juiz e os índices (em reg['exibidas']) das afirmações com citação válida."""
    validas = [i for i, a in enumerate(reg["exibidas"]) if a["fontes"] and all(f in reg["rotulos"] for f in a["fontes"])]
    resposta = "\n".join(a["texto"] for a in reg["exibidas"]) or "(nenhuma afirmação)"
    if reg["lacunas"]:
        resposta += "\nLacunas declaradas: " + "; ".join(reg["lacunas"])
    afirmacoes = []
    for k, i in enumerate(validas):
        a = reg["exibidas"][i]
        fontes = "\n".join(f"  - {chunks[reg['rotulos'][f]]['texto'][:500]}" for f in a["fontes"])
        afirmacoes.append(f"Afirmação {k}: {a['texto']}\n  Trechos citados:\n{fontes}")
    corpo = "\n".join(afirmacoes) if afirmacoes else "(nenhuma afirmação com citação válida)"
    return f"## Sistema {n}\nResposta:\n{resposta}\n\nAfirmações a verificar:\n{corpo}", validas


def julgar_questao(regs: list[dict], pergunta: dict, chunks: dict[str, dict]) -> list[dict]:
    """Uma chamada por pergunta: o juiz vê as respostas de todas as configurações, anonimizadas e embaralhadas."""
    ordem = sorted(regs, key=lambda r: hashlib.sha256(f"{pergunta['id']}{r['config']}".encode()).hexdigest())
    blocos, validas_por_sistema = [], {}
    for n, reg in enumerate(ordem, start=1):
        bloco, validas = _bloco_sistema(n, reg, chunks)
        blocos.append(bloco)
        validas_por_sistema[n] = validas
    esperada = (f"Resposta esperada: {pergunta['resposta_esperada']}" if pergunta["responde"]
                else "Esta pergunta NÃO tem resposta no corpus; o correto é admitir a lacuna.")
    prompt = (
        "Você é um avaliador rigoroso e imparcial de sistemas de perguntas e respostas com citações. "
        "Abaixo estão as respostas de vários sistemas anônimos à mesma pergunta. Para CADA sistema:\n"
        "1) Para cada afirmação, diga se os trechos citados sustentam o que ela afirma (trechos em inglês, afirmação "
        "possivelmente em português). Use o número da afirmação como índice.\n"
        "2) correcao: compare a resposta com a resposta esperada (0, 1 ou 2). Para pergunta sem resposta no corpus, "
        "use 2 se admitiu a lacuna sem inventar e 0 se respondeu como se soubesse.\n"
        "3) admite_lacuna: a resposta reconhece que o corpus não traz a informação?\n"
        "Avalie cada sistema de forma independente, sem comparar um com o outro.\n\n"
        f"Pergunta: {pergunta['pergunta']}\n{esperada}\n\n" + "\n\n".join(blocos)
    )
    j = get_llm("judge").with_structured_output(JulgamentoLote).invoke(prompt)
    por_sistema = {a.sistema: a for a in j.avaliacoes}
    saida = []
    for n, reg in enumerate(ordem, start=1):
        a = por_sistema.get(n)
        if a is None:
            raise ValueError(f"o juiz não avaliou o sistema {n}")
        dados = {v.indice: v.sustentada for v in a.veredictos}
        sustentadas = [False] * len(reg["exibidas"])  # sem citação válida = não sustentada
        for k, i in enumerate(validas_por_sistema[n]):
            sustentadas[i] = bool(dados.get(k, False))
        saida.append({"config": reg["config"], "id": reg["id"], "sustentadas": sustentadas,
                      "correcao": max(0, min(2, a.correcao)), "admite_lacuna": a.admite_lacuna})
    return saida


def cmd_julgar(args) -> int:
    chunks = trechos_por_id()
    perguntas = {q["id"]: q for q in carregar_perguntas(None)}
    feitos = {(j["config"], j["id"]) for j in ler_jsonl(JULGAMENTOS)}
    pendentes: dict[str, list[dict]] = defaultdict(list)
    for reg in ler_jsonl(RESPOSTAS):
        if reg.get("erro") or reg["config"] not in args.config or (reg["config"], reg["id"]) in feitos:
            continue
        if args.ids and reg["id"] not in args.ids:
            continue
        pendentes[reg["id"]].append(reg)
    # só a última resposta de cada (config, pergunta) entra no lote
    erros_seguidos = 0
    for qid, regs in sorted(pendentes.items()):
        unicos = list({r["config"]: r for r in regs}.values())
        try:
            for j in julgar_questao(unicos, perguntas[qid], chunks):
                anexar_jsonl(JULGAMENTOS, j)
            erros_seguidos = 0
            print(f"  julgada {qid} ({len(unicos)} sistemas)", flush=True)
        except CotaDiariaEsgotada as exc:
            print(f"\nCota diária do juiz esgotada ({exc}). Rode de novo amanhã.")
            return 2
        except Exception as exc:  # noqa: BLE001
            erros_seguidos += 1
            print(f"  FALHOU {qid}: {type(exc).__name__}: {str(exc)[:160]}")
            if erros_seguidos >= 2:
                print("2 falhas seguidas do juiz (provável cota do dia); parando. Rode de novo amanhã.")
                return 2
    return 0


# ---------------------------------------------------------------- etapa 3: resumo
def recall(reg: dict, pergunta: dict) -> float | None:
    if not pergunta["responde"]:
        return None
    chunks = trechos_por_id_cache()
    contexto = [chunks[i] for i in reg["contexto_ids"] if i in chunks]
    achados = [any(t["doc_id"] == e["doc_id"] and t["pagina"] in e["paginas"] for t in contexto) for e in pergunta["evidencias"]]
    return sum(achados) / len(achados)


_CACHE_CHUNKS: dict = {}


def trechos_por_id_cache() -> dict[str, dict]:
    if not _CACHE_CHUNKS:
        _CACHE_CHUNKS.update(trechos_por_id())
    return _CACHE_CHUNKS


def media(valores: list[float]) -> float | None:
    valores = [v for v in valores if v is not None]
    return round(mean(valores), 3) if valores else None


def cmd_resumo(_args) -> int:
    perguntas = {q["id"]: q for q in carregar_perguntas(None)}
    respostas = {(r["config"], r["id"]): r for r in ler_jsonl(RESPOSTAS) if not r.get("erro")}
    julgados = {(j["config"], j["id"]): j for j in ler_jsonl(JULGAMENTOS)}
    linhas = []
    for (cfg, qid), r in sorted(respostas.items()):
        q, j = perguntas[qid], julgados.get((cfg, qid))
        n_exib, n_bruta = len(r["exibidas"]), len(r["bruta"])
        validas_brutas = sum(1 for a in r["bruta"] if a["fontes"] and all(f in r["rotulos"] for f in a["fontes"]))
        linhas.append({
            "config": cfg, "id": qid, "tipo": q["tipo"],
            "recall_contexto": recall(r, q),
            "citacoes_validas": (validas_brutas / n_bruta) if n_bruta else None,
            "fidelidade": (sum(j["sustentadas"]) / len(j["sustentadas"])) if j and j["sustentadas"] else None,
            "correcao": (j["correcao"] / 2) if j else None,
            "admite_lacuna": (1.0 if j["admite_lacuna"] else 0.0) if j and not q["responde"] else None,
            "afirmacoes": n_exib, "taxa_rejeitadas": r["extras"].get("taxa_rejeitadas"),
            "chamadas_llm": r["chamadas_llm"], "tempo_s": r["tempo_s"],
        })
    if not linhas:
        print("Nada para resumir: rode primeiro 'rodar'.")
        return 1
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    campos = list(linhas[0])
    with open(RESULTADOS / "detalhe.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    metricas = ["recall_contexto", "citacoes_validas", "fidelidade", "correcao", "admite_lacuna", "chamadas_llm", "tempo_s"]
    grupos: dict[tuple, list[dict]] = defaultdict(list)
    for l in linhas:
        grupos[(l["config"], "todas")].append(l)
        grupos[(l["config"], l["tipo"])].append(l)
    resumo = [{"config": c, "grupo": g, "n": len(ls), **{m: media([l[m] for l in ls]) for m in metricas}}
              for (c, g), ls in sorted(grupos.items())]
    with open(RESULTADOS / "resumo.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "grupo", "n"] + metricas)
        w.writeheader()
        w.writerows(resumo)

    def fmt(v):
        return "-" if v is None else f"{v:.2f}"
    print("\n| Config | Grupo | n | recall | cit. válidas | fidelidade | correção | admite lacuna | chamadas | tempo (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in resumo:
        if r["grupo"] in ("todas", "sem_resposta", "comparacao"):
            print(f"| {r['config']} | {r['grupo']} | {r['n']} | " + " | ".join(fmt(r[m]) for m in metricas) + " |")
    print(f"\nArquivos: {RESULTADOS / 'resumo.csv'} e {RESULTADOS / 'detalhe.csv'}")
    sem_juiz = sum(1 for l in linhas if l["correcao"] is None)
    if sem_juiz:
        print(f"Atenção: {sem_juiz} respostas ainda sem julgamento (fidelidade/correção vazias). Rode 'julgar'.")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for nome in ("rodar", "julgar"):
        s = sub.add_parser(nome)
        s.add_argument("--config", nargs="+", default=["A", "B", "C", "D"], choices=list(CONFIGS))
        s.add_argument("--ids", nargs="+", help="só estas perguntas (ex.: Q01 Q02)")
    sub.add_parser("resumo")
    args = p.parse_args()
    return {"rodar": cmd_rodar, "julgar": cmd_julgar, "resumo": cmd_resumo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
