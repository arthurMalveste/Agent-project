"""Fase 6: interface Streamlit com o raciocínio do agente visível. Uso: streamlit run app.py"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.graph import app_grafo  # noqa: E402
from src.llm import CotaDiariaEsgotada  # noqa: E402

st.set_page_config(page_title="Agente de pesquisa STPA", page_icon="🔎", layout="wide")

NOMES_NOS = {
    "classificar": "Classificar a pergunta", "planejar": "Planejar subconsultas", "buscar": "Buscar no corpus",
    "avaliar": "Avaliar relevância", "reescrever": "Reescrever consulta", "comparar": "Comparar documentos",
    "sintetizar": "Escrever a resposta", "verificar": "Verificar citações", "responder_direto": "Responder direto",
}
EXEMPLOS = [
    "O que é uma unsafe control action e quais são os seus quatro tipos?",
    "Compare como o Handbook e o artigo de Rismani et al. tratam cenários causais.",
    "Que técnica El Badaoui et al. usam para reduzir a subjetividade na priorização de UCAs?",
    "Quais são os critérios da norma ISO/IEC/IEEE 29148?",
]


def resumo_passo(no: str, s: dict) -> str:
    """Uma linha legível para cada decisão do agente."""
    if no == "classificar":
        return "precisa pesquisar o corpus" if s["rota"] == "pesquisa" else "conversa/fora do escopo: resposta direta, sem fontes"
    if no == "planejar":
        doc = f" · filtro: `{s['doc_id']}`" if s.get("doc_id") else ""
        subs = "".join(f"\n  - {c}" for c in s["consultas"])
        return f"tipo **{s['tipo']}**{doc}; subconsultas:{subs}"
    if no == "buscar":
        por_consulta = ", ".join(f"{len(ids)}" for ids in s["hits"].values())
        return f"{len(s['trechos'])} trechos únicos (por subconsulta: {por_consulta})"
    if no == "avaliar":
        notas = s["notas"]
        bons = sum(1 for n in notas.values() if n >= 1)
        return f"{bons} de {len(notas)} trechos relevantes → {'**suficiente**' if s['suficiente'] else '**insuficiente**'}"
    if no == "reescrever":
        subs = "".join(f"\n  - {c}" for c in s["consultas"])
        return f"tentativa {s['tentativas']}: novas consultas:{subs}"
    if no == "comparar":
        return "análise comparativa entre documentos gerada"
    if no == "sintetizar":
        return f"{len(s['resposta']['bruta']['afirmacoes'])} afirmações propostas"
    if no == "verificar":
        r = s["resposta"]
        return f"{len(r['afirmacoes'])} mantidas, {len(r['removidas'])} removidas ({r['taxa_rejeitadas']:.0%} rejeitadas)"
    return "ok"


def mostrar_resposta(r: dict) -> None:
    st.markdown(r["texto"])
    if r.get("referencias"):
        st.markdown("**Fontes**")
        for ref in r["referencias"]:
            with st.expander(f"[{ref['numero']}] {ref['referencia']}"):
                st.caption(f"trecho `{ref['trecho_id']}`")
                st.write(ref["texto"])
    if r.get("removidas"):
        with st.expander(f"{len(r['removidas'])} afirmações removidas pela verificação"):
            for x in r["removidas"]:
                st.markdown(f"- ~~{x['texto']}~~ — {x['motivo']}")


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("Parâmetros")
    provedor = st.selectbox("Provedor de LLM", ["groq", "gemini", "ollama"], index=0)
    modo = st.selectbox("Busca", ["hibrida", "vetorial", "bm25"], help="híbrida = vetorial + BM25 combinadas por RRF")
    k = st.slider("Trechos por subconsulta (k)", 3, 10, 6)
    max_tentativas = st.slider("Reescritas máximas", 0, 3, 3, help="0 desliga o loop de reescrita")
    verificar_citacoes = st.checkbox("Juiz de citações", value=True)
    st.divider()
    st.caption("Exemplos")
    for ex in EXEMPLOS:
        if st.button(ex, use_container_width=True):
            st.session_state["pergunta_exemplo"] = ex
    if st.button("Limpar conversa"):
        st.session_state["historico"] = []

os.environ["LLM_PROVIDER"] = provedor
st.title("🔎 Agente de pesquisa sobre STPA")
st.caption("Responde com documento e página, e verifica cada citação antes de mostrar.")

historico = st.session_state.setdefault("historico", [])
for msg in historico:
    with st.chat_message(msg["papel"]):
        if msg["papel"] == "user":
            st.write(msg["conteudo"])
        else:
            col_resp, col_passos = st.columns([3, 2])
            with col_resp:
                mostrar_resposta(msg["resposta"])
            with col_passos, st.expander("Passos do agente", expanded=False):
                for linha in msg["passos"]:
                    st.markdown(linha)

pergunta = st.chat_input("Pergunte sobre o corpus de STPA") or st.session_state.pop("pergunta_exemplo", None)
if pergunta:
    historico.append({"papel": "user", "conteudo": pergunta})
    with st.chat_message("user"):
        st.write(pergunta)
    with st.chat_message("assistant"):
        col_resp, col_passos = st.columns([3, 2])
        with col_passos:
            status = st.status("Pesquisando...", expanded=True)
        passos, estado = [], {}
        try:
            config = {"configurable": {"k": k, "modo": modo, "max_tentativas": max_tentativas,
                                       "verificar_citacoes": verificar_citacoes}}
            for evento in app_grafo.stream({"pergunta": pergunta}, config=config, stream_mode="updates"):
                for no, saida in evento.items():
                    estado.update(saida)
                    linha = f"**{NOMES_NOS.get(no, no)}** — {resumo_passo(no, saida)}"
                    passos.append(linha)
                    status.markdown(linha)
            status.update(label="Concluído", state="complete", expanded=False)
            with col_resp:
                mostrar_resposta(estado["resposta"])
            historico.append({"papel": "assistant", "resposta": estado["resposta"], "passos": passos})
        except CotaDiariaEsgotada as exc:
            status.update(label="Cota diária esgotada", state="error")
            col_resp.error(f"A cota gratuita do provedor acabou por hoje ({exc}). Troque o provedor na barra lateral.")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Erro", state="error")
            col_resp.error(f"{type(exc).__name__}: {exc}")
