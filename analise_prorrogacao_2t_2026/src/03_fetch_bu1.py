"""Baixa o Boletim de Urna do 1º turno (dataset oficial 'resultados-2026-boletim-de-urna', CSV por UF),
filtra Presidente (CD_ELEICAO 6257) e grava resultados por seção em RAW_DIR/b1/<uf>.parquet.
URLs reais: https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2026/buweb/bweb_1t_<UF>_051020261403.zip
"""
import ssl, sys, urllib.request, zipfile, os
import pandas as pd
from common import *

URL = "https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2026/buweb/bweb_1t_{uf}_051020261403.zip"
COLS = ["CD_ELEICAO", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_VOTAVEL", "QT_VOTOS", "QT_APTOS",
        "QT_COMPARECIMENTO", "DT_ABERTURA", "DT_ENCERRAMENTO", "DS_SECOES_AGREGADAS", "QT_ELEI_BIOM_SEM_HABILITACAO",
        "NR_LOCAL_VOTACAO", "DS_TIPO_VOTAVEL"]
ctx = ssl.create_default_context(cafile=CA_BUNDLE)
out = RAW_DIR / "b1"; out.mkdir(exist_ok=True)

for uf in [u.upper() for u in sys.argv[1:]]:
    dst = out / f"{uf.lower()}.parquet"
    if dst.exists():
        continue
    tmp = out / f"{uf}.zip"
    req = urllib.request.Request(URL.format(uf=uf), headers={"User-Agent": "pesquisa-prorrogacao-2t-2026/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    parts = []
    with zipfile.ZipFile(tmp) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as fh:
            for ch in pd.read_csv(fh, sep=";", encoding="latin1", chunksize=300000, dtype=str, usecols=COLS):
                p = ch[ch.CD_ELEICAO == "6257"]
                if len(p):
                    parts.append(p.drop(columns=["CD_ELEICAO"]))
    df = pd.concat(parts, ignore_index=True)
    for c in ["CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "NR_VOTAVEL", "QT_VOTOS", "QT_APTOS", "QT_COMPARECIMENTO"]:
        df[c] = pd.to_numeric(df[c])
    df.to_parquet(dst, index=False)
    os.remove(tmp)
    print(uf, len(df), flush=True)
