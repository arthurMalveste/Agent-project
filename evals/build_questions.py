"""Gera evals/questions.jsonl e VALIDA cada evidência contra data/chunks.jsonl.

Cada pergunta traz evidências: (doc_id, páginas, regex literal que deve aparecer num trecho dessas páginas).
Se uma evidência não for encontrada no corpus, o script falha: o gabarito nunca fica sem lastro.

Uso: python evals/build_questions.py
Tipos: factual (Handbook), documento (um artigo específico), comparacao (2+ documentos), sem_resposta.
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
H, MY, QI, NO, BA, CH = "leveson2018_handbook", "mylius2025_2506_01782", "qi2023_2302_10588", "nouri2024_2403_09509", "badaoui2025_2508_10446", "chen2025_2508_16708"
DA, YU, SA, AB17, MI, AB16T = "dardik2026_2605_30697", "yu2020_2006_02930", "sahay2022_2212_10830", "abdulkhaleq2017_1703_03657", "mindermann2017_1710_11571", "abdulkhaleq2016_1612_03103"
RI, RA, BAR = "rismani2024_2410_22526", "raeisdanaei2025_2503_12043", "barrett2025_2512_17600"


def ev(doc, paginas, contem):
    return {"doc_id": doc, "paginas": paginas, "contem": contem}


PERGUNTAS = [
    # ---- factual: Handbook ----
    ("factual", "O que é uma perda (loss) no STPA?",
     "Algo de valor para as partes interessadas: vida humana ou lesão, dano material, poluição ambiental, perda de missão, de reputação ou de informação sensível.",
     [ev(H, [16], r"A loss involves something of value to stakeholders")]),
    ("factual", "O que é um hazard no STPA e como ele se relaciona com as perdas?",
     "Um estado do sistema ou conjunto de condições que, somado às piores condições ambientais, leva a uma perda.",
     [ev(H, [17], r"A hazard is a system state or set of conditions")]),
    ("factual", "Quais são as quatro formas de uma ação de controle ser insegura?",
     "Não fornecer a ação causa hazard; fornecê-la causa hazard; fornecer cedo demais, tarde demais ou fora de ordem; e durar tempo demais ou ser interrompida cedo demais.",
     [ev(H, [36], r"four ways a control action can be unsafe")]),
    ("factual", "Quais são as cinco partes de uma ação de controle insegura (UCA) bem formada?",
     "Fonte (controlador), tipo, ação de controle, contexto e vínculo com os hazards.",
     [ev(H, [37], r"A UCA contains five parts")]),
    ("factual", "Quais são as partes da etapa de definir o propósito da análise no STPA?",
     "Identificar perdas, identificar hazards de nível de sistema, identificar restrições de nível de sistema e refinar os hazards (opcional).",
     [ev(H, [15], r"Defining the purpose of the analysis has four parts")]),
    ("factual", "Quais são os dois tipos de cenários de perda que devem ser considerados?",
     "(a) por que ocorreriam ações de controle inseguras; (b) por que ações de controle seriam executadas incorretamente ou não executadas, levando a hazards.",
     [ev(H, [43], r"Two types of loss scenarios")]),
    ("factual", "Como se define uma restrição de nível de sistema a partir de um hazard?",
     "Inverte-se a condição do hazard: a restrição especifica condições ou comportamentos que precisam ser satisfeitos para evitar o hazard.",
     [ev(H, [20], r"simply invert the condition")]),
    # ---- documento específico ----
    ("documento", "Quantos estágios o STPA tem segundo Mylius (2025), no artigo sobre IA de fronteira?",
     "Quatro estágios, aplicados em sequência.", [ev(MY, [4], r"four stages")]),
    ("documento", "Quantos artigos Qi et al. incluíram na revisão sistemática sobre STPA para sistemas habilitados por aprendizado?",
     "31 artigos.", [ev(QI, [1, 3], r"31 papers")]),
    ("documento", "Quantos participantes da indústria foram entrevistados por Nouri et al. no estudo sobre STPA no desenvolvimento distribuído de direção autônoma?",
     "14 participantes.", [ev(NO, [4], r"14 participants")]),
    ("documento", "Que técnica de simulação El Badaoui et al. usam para reduzir a subjetividade na priorização de UCAs?",
     "Simulação de Monte Carlo.", [ev(BA, [1], r"Monte Carlo simulation")]),
    ("documento", "Quais fatores Chen et al. consideram para priorizar os requisitos derivados do STPA?",
     "Tempo de implementação, custo, tipo de requisito e cobertura regulatória.",
     [ev(CH, [1], r"implementation time, cost, requirement type")]),
    ("documento", "O que é a ferramenta FASR e o que ela faz?",
     "Ferramenta (Formalizing and Automating STPA with Robustness) que usa engenharia baseada em modelos e métodos formais para identificar UCAs de forma automática e completa.",
     [ev(DA, [1], r"FASR")]),
    ("documento", "Que extensão do STPA-Sec Yu et al. propõem e em qual caso de uso a aplicam?",
     "STPA-DFSec, baseada em fluxo de dados com palavras-guia; aplicada a um sistema de chave digital de veículo.",
     [ev(YU, [3], r"STPA-DFSec")]),
    ("documento", "Quais foram as ameaças mais críticas para o controlador do motor no estudo do CyberShip?",
     "Spoofing, tampering e negação de serviço.", [ev(SA, [12], r"tampering and denial of service")]),
    ("documento", "Qual é a contribuição de Abdulkhaleq et al. (2017) sobre STPA e a norma ISO 26262?",
     "Orientação de como usar o STPA em conformidade com a ISO 26262, aplicada à arquitetura de um veículo totalmente automatizado.",
     [ev(AB17, [2], r"guidance on how to use STPA in compliance with ISO 26262")]),
    ("documento", "Que ferramenta apoia o STPA-Priv no estudo de Mindermann et al.?",
     "XSTAMPP.", [ev(MI, [1], r"XSTAMPP")]),
    ("documento", "Que ferramenta de código aberto gera casos de teste a partir do STPA e em qual protótipo ela foi ilustrada?",
     "STPA TCGenerator; protótipo de controle de cruzeiro adaptativo (ACC) com um Lego Mindstorms EV3.",
     [ev(AB16T, [1], r"TCGenerator")]),
    # ---- comparação (2+ documentos) ----
    ("comparacao", "Compare como o Handbook e o artigo de Rismani et al. tratam a identificação de cenários causais.",
     "Ambos identificam cenários após as UCAs; o Handbook define dois tipos de cenário, Rismani et al. aplica a etapa a sistemas de IA.",
     [ev(H, [43], r"Two types of loss scenarios"), ev(RI, [27], r"loss scenarios")]),
    ("comparacao", "Compare as abordagens de El Badaoui et al. e Chen et al. para priorizar resultados do STPA.",
     "El Badaoui prioriza UCAs (julgamento de especialistas, Monte Carlo, matriz); Chen prioriza requisitos por quatro fatores (tempo, custo, tipo, cobertura regulatória), também com Monte Carlo.",
     [ev(BA, [1], r"Monte Carlo simulation"), ev(CH, [1], r"implementation time, cost, requirement type")]),
    ("comparacao", "Compare a extensão STPA-DFSec de Yu et al. com o STPA-Priv de Mindermann et al.",
     "STPA-DFSec estende o STPA-Sec para segurança (security) via fluxos de dados; STPA-Priv estende o STPA para riscos de privacidade.",
     [ev(YU, [3], r"STPA-DFSec"), ev(MI, [1], r"privacy")]),
    ("comparacao", "Como as abordagens de Raeisdanaei et al. e de Dardik et al. automatizam o STPA? Compare.",
     "Raeisdanaei: framework Python de código aberto com fluxos baseados em LLM; Dardik: FASR, com métodos formais e engenharia baseada em modelos.",
     [ev(RA, [1], r"large language models"), ev(DA, [1], r"formal methods")]),
    ("comparacao", "Compare como Mylius et al. e Barrett et al. usam STAMP/STPA na segurança de IA.",
     "Mylius: análise sistemática de hazards para IA de fronteira; Barrett: caracterização dos fatores de perda de controle de IA.",
     [ev(MY, [1], r"STPA"), ev(BAR, [1], r"STAMP|STPA")]),
    ("comparacao", "Compare STPA-Sec, STRIDE e CORAS na análise do CyberShip feita por Sahay et al.",
     "Três metodologias de avaliação de risco aplicadas ao CyberShip para identificar ameaças, vulnerabilidades e requisitos de segurança.",
     [ev(SA, [1], r"STPA-Sec, STRIDE and CORAS")]),
    # ---- sem resposta no corpus ----
    ("sem_resposta", "Quais são os critérios de qualidade de requisitos definidos pela norma ISO/IEC/IEEE 29148?",
     "Não consta no corpus (a norma não está nos documentos).", []),
    ("sem_resposta", "Qual é a taxa de falha do motor Rolls-Royce Trent 1000?",
     "Fora do escopo e do corpus.", []),
    ("sem_resposta", "Quais requisitos a norma ISO/SAE 21434 impõe para o gerenciamento de riscos cibernéticos em veículos?",
     "Não consta no corpus (a norma não aparece nos documentos).", []),
    ("sem_resposta", "Como o STPA é combinado com histórias de usuário (user stories) em equipes Scrum?",
     "Não consta no corpus.", []),
]


def normalizar(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def main() -> int:
    chunks = [json.loads(l) for l in (RAIZ / "data" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    por_pagina: dict[tuple[str, int], str] = {}
    for c in chunks:
        k = (c["doc_id"], c["pagina"])
        por_pagina[k] = por_pagina.get(k, "") + " " + normalizar(c["texto"])
    linhas, erros = [], []
    for n, (tipo, pergunta, esperada, evidencias) in enumerate(PERGUNTAS, start=1):
        qid = f"Q{n:02d}"
        for e in evidencias:
            achou = any(re.search(e["contem"], por_pagina.get((e["doc_id"], p), ""), re.I) for p in e["paginas"])
            if not achou:
                erros.append(f"{qid}: '{e['contem']}' não encontrado em {e['doc_id']} p.{e['paginas']}")
        linhas.append({"id": qid, "tipo": tipo, "pergunta": pergunta, "resposta_esperada": esperada,
                       "responde": tipo != "sem_resposta", "evidencias": evidencias})
    # perguntas sem resposta: os termos-chave não podem aparecer no corpus
    texto_total = " ".join(por_pagina.values())
    for termo in ("29148", "Trent 1000", "21434", "Scrum"):
        if re.search(termo, texto_total, re.I):
            erros.append(f"sem_resposta: '{termo}' aparece no corpus")
    if erros:
        print("\n".join(erros))
        return 1
    (RAIZ / "evals" / "questions.jsonl").write_text(
        "\n".join(json.dumps(l, ensure_ascii=False) for l in linhas) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"{len(linhas)} perguntas validadas e gravadas:", dict(Counter(l["tipo"] for l in linhas)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
