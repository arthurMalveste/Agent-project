# Agente de Pesquisa com RAG e Citações Verificáveis

Agente que responde perguntas sobre um corpus de documentos (STPA e engenharia de requisitos) citando **documento e página** em cada afirmação. Em vez de uma busca fixa, o fluxo é um grafo em LangGraph que classifica a pergunta, planeja subconsultas, avalia os trechos achados, reescreve a consulta quando a busca é fraca, compara documentos e verifica as citações antes de responder.

> Status: em desenvolvimento.

## Arquitetura (resumo)

```
classificar ─┬─ direto ──► responder_direto ──► FIM
             └─ pesquisa ─► planejar ─► buscar ─► avaliar ─┬─ suficiente ─► comparar ─► sintetizar ─► verificar ─► FIM
                                           ▲               │
                                           └── reescrever ◄┘ (máx. 3 tentativas)
```

- **Busca híbrida:** vetorial (Chroma) + BM25, combinadas por Reciprocal Rank Fusion.
- **Citações verificáveis:** o LLM só vê rótulos `[S1]`, `[S2]`; as referências finais saem dos metadados, nunca do texto do modelo.
- **Custo zero:** LLM via planos gratuitos (Groq / Gemini) e embeddings locais.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # preencha as chaves
python teste_llm.py      # confere a conexão com os LLMs
```

## Corpus

Os PDFs **não** estão no repositório (direitos autorais). A lista de documentos está em `data/metadata.csv`, com a URL de download de cada um; o script `python src/download_corpus.py` baixa os que têm acesso aberto para `data/raw/`.
