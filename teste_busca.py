"""Checkpoint da Fase 2/3: roda consultas nos três modos de busca e mostra os trechos para leitura humana.

Uso: python teste_busca.py                      (consultas padrão)
     python teste_busca.py "sua consulta" 3     (consulta própria, k=3)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from src.tools import buscar_trechos, ler_vizinhos, listar_documentos  # noqa: E402

PADRAO = [
    "o que é uma unsafe control action",
    "UCA",
    "how are loss scenarios identified",
    "quantos passos tem o STPA",  # português contra corpus em inglês: testa o embedding multilíngue
]


def mostrar(consulta: str, k: int) -> None:
    print("=" * 100)
    print(f"CONSULTA: {consulta}")
    for modo in ("vetorial", "bm25", "hibrida"):
        print(f"\n  --- {modo} ---")
        for t in buscar_trechos(consulta, k=k, modo=modo):
            trecho = t["texto"].replace("\n", " ")[:170]
            print(f"  [{t['id']}] ({t['autores'].split(';')[0]}, {t['ano']}, p.{t['pagina']}) {trecho}...")


if __name__ == "__main__":
    docs = listar_documentos()
    print(f"{len(docs)} documentos indexados. Exemplo de vizinhos de um trecho:")
    primeiro = buscar_trechos("unsafe control action", k=1)[0]["id"]
    viz = ler_vizinhos(primeiro)
    print(f"  {primeiro} -> anterior: {viz['anterior'] and viz['anterior']['id']}, seguinte: {viz['seguinte'] and viz['seguinte']['id']}\n")
    if len(sys.argv) > 1:
        mostrar(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        for q in PADRAO:
            mostrar(q, 3)
