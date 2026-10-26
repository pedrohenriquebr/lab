# Prorrogação da votação no 2º turno 2026 (25/10/2026) — investigação técnica

**Status: INTERRUPTIDO / PARCIAL.** Só código e evidência de partida; nenhuma conclusão final.

## O que foi verificado
- ID 6258 = "Eleição Ordinária Federal – 2026 2º Turno" (Presidente), pleito 3221 (`ele-c.json`).
- Total oficial TSE (`data/tse_oficial_6258_totais.json`): Lula 64.256.232 × Flávio 59.826.041; margem 4.430.191; comparecimento 128.236.074.
- Logs de votação por seção: `https://resultados.tse.jus.br/oficial/ele2026/arquivo-urna/3221/dados/<uf>/<mun>/<zona>/<secao>/` (aux.json → hash → `*-log.jez`, `*-bu.dat`). Lista de seções: `.../arquivo-urna/3221/config/<uf>/<uf>-p003221-cs.json`.
- Datasets em lote de 2º turno (BU/logs de urna) ainda não publicados no portal de dados abertos; os "logs Gedai" são de preparação, não de votação.
- Relógio da urna = hora local; votação regular 08–17h de Brasília (06–15h no AC).

## Scripts (`src/`)
`01_sections.py` lista seções · `02_crawl.py` baixa e analisa log/BU (retomável por chunks) · `03_fetch_bu1.py` BU do 1º turno ·
`04_build.py` consolida e infere fuso · `05_tables.py` tabelas · `06_ecological.py` análise ecológica (não validada).
Uso: `export RAW_DIR=<dir fora do repo>`; rodar na ordem 01→06 (com `python3 -I`, `sys.path` em `src/`).

## Pendente
Crawl completo, tabela por UF, gráficos, análise ecológica, anomalias, réplica do painel, README final.
