"""Consolida os chunks do crawl em tabelas por seção (RAW_DIR/secoes.parquet) e eventos tardios (RAW_DIR/late.parquet).

Fuso (decisão metodológica, ver README): o relógio da urna é LOCAL. A votação regular vai de 08:00 a 17:00 horário de
Brasília (confirmado pelos dados: 'Urna pronta para receber votos' às 06:00 no AC, 07:00 em AM/MT/MS/RO/RR etc.).
O deslocamento do fuso de cada município é INFERIDO dos dados: moda de round(abertura_local/3600) − 8 entre as seções
do município que abriram dentro de ±10 min de uma hora cheia. Seções cuja abertura difere dessa moda são sinalizadas.
Cortes:
  A = 17:00 no relógio da urna (literal, "17h local")
  B = 17:00 de Brasília convertido ao relógio da urna (17 + off)
  X = 19:00 de Brasília (limite da prorrogação) convertido (19 + off)
"""
import glob, sys
import numpy as np
import pandas as pd
from common import *


def chunks():
    import os
    for uf in UFS:
        for f in sorted(glob.glob(str(RAW_DIR / "out" / uf / "chunk_*.summary.parquet"))):
            if os.path.exists(f.replace(".summary.parquet", ".done")):
                yield f


def load():
    return pd.concat([pd.read_parquet(f) for f in chunks()], ignore_index=True)


def load_late(tzmap, from_brt=16):
    """Eventos de voto a partir de `from_brt` h de Brasília (relógio da urna = from_brt + off)."""
    out = []
    for f in chunks():
        l = pd.read_parquet(f.replace("summary", "late"))
        if len(l) == 0:
            continue
        l = l.merge(tzmap, left_on=["uf", "mun"], right_index=True, how="left")
        l["tz_off"] = l.tz_off.fillna(0)
        l = l[l.t_vote >= (from_brt + l.tz_off) * 3600]
        l["secid"] = l.uf + "-" + l.mun + "-" + l.zona + "-" + l.secao
        out.append(l.drop(columns=["uf", "mun", "zona", "secao"]).astype({"turno": "int8", "tz_off": "int8"}))
    return pd.concat(out, ignore_index=True)


def infer_tz(s):
    """off em horas relativas a Brasília, por município (UF+mun)."""
    o = s.loc[s.t2_open_s.notna(), ["uf", "mun", "t2_open_s"]].copy()
    ok = (o.t2_open_s % 3600 <= 600) | (o.t2_open_s % 3600 >= 3000)   # perto de hora cheia
    o = o[ok]
    o["h"] = (o.t2_open_s / 3600).round().astype(int)
    mode = o.groupby(["uf", "mun"]).h.agg(lambda x: x.value_counts().idxmax()).rename("open_h_mode")
    return (mode - 8).rename("tz_off")


def main():
    s = load()
    s["secid"] = s.uf + "-" + s.mun + "-" + s.zona + "-" + s.secao
    tz = infer_tz(s)
    late = load_late(tz)
    s = s.merge(tz, left_on=["uf", "mun"], right_index=True, how="left")
    s["tz_off"] = s.tz_off.fillna(s.groupby("uf").tz_off.transform(lambda x: x.mode().iloc[0] if x.notna().any() else np.nan))
    s["open_h_sec"] = (s.t2_open_s / 3600).round()
    s["tz_mismatch"] = (s.t2_open_s.notna()) & ((s.t2_open_s % 3600 <= 600) | (s.t2_open_s % 3600 >= 3000)) & (s.open_h_sec - 8 != s.tz_off)
    for t in (1, 2):
        # corte B por hora cheia: horas locais >= 17+off
        offs = s.tz_off.fillna(0).astype(int).values
        for kind in ("vote", "hab"):
            H = s[[f"t{t}_{'vote' if kind=='vote' else 'hab'}_h{h:02d}" for h in range(24)]].values
            idx = np.arange(24)[None, :]
            cutB = (17 + offs)[:, None]; cutA = np.full_like(cutB, 17); cutX = (19 + offs)[:, None]
            s[f"t{t}_{kind}_A"] = (H * (idx >= cutA)).sum(1)
            s[f"t{t}_{kind}_B"] = (H * (idx >= cutB)).sum(1)
            s[f"t{t}_{kind}_X"] = (H * (idx >= cutX)).sum(1)
    s.to_parquet(RAW_DIR / "secoes.parquet", index=False)
    late.to_parquet(RAW_DIR / "late.parquet", index=False)
    print(s.shape, late.shape)
    print(s.groupby("uf").tz_off.agg(lambda x: x.value_counts().to_dict()))


if __name__ == "__main__":
    main()
