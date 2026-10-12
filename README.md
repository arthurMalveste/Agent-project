# Agente de Pesquisa com RAG e Citações Verificáveis

Um agente que responde perguntas sobre um conjunto de documentos e **cita o documento e a página de cada afirmação**. Antes de mostrar a resposta, ele confere se cada citação realmente existe e se o trecho citado sustenta a frase.

O corpus de demonstração reúne 24 documentos de acesso aberto sobre **STPA** (Systems-Theoretic Process Analysis), uma técnica de análise de segurança de sistemas: o STPA Handbook, o STPA Primer e artigos científicos.

## O que o diferencia de um RAG comum

Num RAG simples, toda pergunta faz uma busca fixa e o modelo responde. Aqui o fluxo é um **grafo com decisões** (LangGraph):

```
classificar ─┬─ direto ──► responder_direto ──► FIM
             └─ pesquisa ─► planejar ─► buscar ─► avaliar ─┬─ suficiente ─► [comparar] ─► sintetizar ─► verificar ─► FIM
                                           ▲               │
                                           └── reescrever ◄┘ (no máximo 3 tentativas)
```

| Etapa | O que faz |
|---|---|
| classificar | Decide se a pergunta precisa do corpus ou é conversa/fora do escopo |
| planejar | Quebra a pergunta em subconsultas em inglês e detecta se ela cita um documento específico |
| buscar | Busca híbrida: vetorial + BM25 combinadas por Reciprocal Rank Fusion |
| avaliar | Dá nota de relevância aos trechos e decide se bastam para responder |
| reescrever | Se a busca foi fraca, reformula a consulta (sinônimos, siglas, termos técnicos) |
| comparar | Em perguntas de comparação, agrupa o que cada documento diz e aponta divergências |
| sintetizar | Escreve a resposta em afirmações, cada uma com rótulos de fonte (`[S1]`, `[S2]`...) |
| verificar | Descarta citações inexistentes e frases que o trecho citado não sustenta |

### Citações que não são inventadas

- O modelo só enxerga rótulos curtos (`[S1]`, `[S2]`), nunca títulos ou páginas.
- Rótulos que não existem no contexto são descartados.
- Um juiz confere se o trecho citado sustenta cada frase; as que não sustentam saem da resposta.
- A lista de referências final (autores, ano, título, página) é montada a partir dos **metadados**, não do texto gerado pelo modelo.

## Custo zero

- **LLM:** planos gratuitos do Groq (`gpt-oss-120b` para sintetizar e comparar, `gpt-oss-20b` para os demais nós) e do Gemini (juiz da avaliação). O provedor é trocável pelo `.env`, e há suporte a Ollama para uso offline.
- **Embeddings:** `intfloat/multilingual-e5-small`, local, em CPU, sem chave.
- **Banco vetorial:** Chroma, em arquivo local.
- Respostas do LLM são guardadas em cache de disco, e chamadas que batem no limite são repetidas com espera automática.

## Estrutura

```
app.py                 interface Streamlit (resposta, fontes e passos do agente)
src/
  config.py            caminhos e parâmetros
  llm.py               fábrica de LLMs (Groq / Gemini / Ollama), cache e retry
  embeddings.py        embeddings locais
  ingest.py            PDFs -> texto por página -> trechos -> Chroma
  tools.py             buscar_trechos, ler_vizinhos, listar_documentos
  graph.py             o grafo do agente
  citations.py         verificação de citações e referências
  download_corpus.py   baixa os PDFs listados em data/metadata.csv
evals/
  build_questions.py   gera e valida as perguntas de avaliação
  questions.jsonl      28 perguntas com resposta esperada e evidências (documento e página)
  run_evals.py         compara as configurações e calcula as métricas
data/
  metadata.csv         título, autores, ano e URL de cada documento
```

## Como usar

Requer Python 3.11 ou superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env          # preencha GROQ_API_KEY e GEMINI_API_KEY
python teste_llm.py             # confere a conexão com os provedores

python src/download_corpus.py   # baixa os PDFs para data/raw/
python src/ingest.py            # extrai, fatia e indexa o corpus
python teste_busca.py           # confere a qualidade da busca

streamlit run app.py            # interface
python -m src.graph "O que é uma unsafe control action?"   # ou pelo terminal
```

Chaves gratuitas: [console.groq.com](https://console.groq.com) e [aistudio.google.com](https://aistudio.google.com).

Os PDFs **não** ficam no repositório (direitos autorais). Alguns sites bloqueiam download automático; nesse caso o `download_corpus.py` lista o que baixar manualmente e onde salvar.

## Avaliação

`evals/questions.jsonl` tem 28 perguntas em quatro tipos: factuais, de um documento específico, de comparação entre documentos e sem resposta no corpus (para testar se o agente admite a lacuna em vez de inventar). Cada pergunta traz a resposta esperada e evidências validadas contra o corpus.

O `run_evals.py` compara quatro configurações:

| Configuração | Descrição |
|---|---|
| A | Só palavras-chave (BM25) |
| B | RAG simples (uma busca vetorial) |
| C | RAG híbrido (vetorial + BM25), sem decisões |
| D | Agente completo, com verificação de citações |

Métricas: recall dos trechos esperados no contexto, citações bem formadas, fidelidade das afirmações aos trechos, correção da resposta, reconhecimento de lacunas, número de chamadas ao LLM e latência. O juiz é um modelo de outra família (Gemini), que avalia as respostas anonimizadas.

```powershell
python evals/run_evals.py rodar --config A B C D
python evals/run_evals.py julgar --config A B C D
python evals/run_evals.py resumo
```

As etapas são retomáveis: se a cota gratuita do dia acabar, o script para e continua de onde parou na próxima execução.
