"""Tabelas descritivas: votos pós-corte por UF/hora, limite superior, biometria, anomalias, réplica do painel.
Entrada: RAW_DIR/secoes.parquet e late.parquet (04_build.py). Saída: data/*.csv"""
import glob, json
from collections import Counter
import numpy as np
import pandas as pd
from common import *

M = 0  # margem oficial preenchida abaixo
s = pd.read_parquet(RAW_DIR / "secoes.parquet")
late = pd.read_parquet(RAW_DIR / "late.parquet")
late["tz_off"] = late.tz_off.astype("int32")
OF = json.load(open(DATA / "tse_oficial_6258_totais.json"))
V_LULA = 64256232; V_FLAVIO = 59826041
MARGEM = V_LULA - V_FLAVIO


def pct(a, b):
    return np.where(b > 0, 100 * a / b, np.nan)


# ---------------------------------------------------------------- 0. catálogo de eventos
cat = Counter()
for uf in UFS:
    for f in glob.glob(str(RAW_DIR / "out" / uf / "chunk_*.catalog.json")):
        for lv, msg, n in json.load(open(f)):
            cat[(lv, msg)] += n
pd.DataFrame([(k[0], k[1], v) for k, v in cat.most_common()], columns=["nivel", "mensagem_normalizada", "n"]).to_csv(
    DATA / "tipos_evento_log_vota_2t.csv", index=False)

# ---------------------------------------------------------------- 1. totais e conferência BU x log x oficial
tot = {
    "secoes_listadas": len(s), "secoes_com_log": int(s.log_ok.sum()), "secoes_com_bu": int(s.bu_ok.sum()),
    "votos_log_t2": int(s.t2_n_vote.sum()), "comparecimento_bu": int(s.bu_comp.sum()),
    "v13_bu": int(s.v13.sum()), "v22_bu": int(s.v22.sum()), "v13_oficial": V_LULA, "v22_oficial": V_FLAVIO,
    "comparecimento_oficial": int(OF["e"]["c"]),
}
tot["margem_bu"] = tot["v13_bu"] - tot["v22_bu"]; tot["margem_oficial"] = MARGEM
pd.Series(tot).to_csv(DATA / "conferencia_totais.csv", header=["valor"])
s["log_ok"] = s.log_ok.fillna(False).astype(bool); s["bu_ok"] = s.bu_ok.fillna(False).astype(bool)
ok = s.log_ok & s.bu_ok
conc = pd.DataFrame({
    "campo": ["votos computados (log) == comparecimento (BU)", "biométricos (log) == BU", "manuais (log) == BU"],
    "secoes_iguais": [(s.t2_n_vote == s.bu_comp)[ok].sum(), (s.t2_n_bio == s.bu_bio)[ok].sum(), (s.t2_n_man == s.bu_man)[ok].sum()],
    "secoes_comparadas": ok.sum()})
conc["pct"] = 100 * conc.secoes_iguais / conc.secoes_comparadas
conc.to_csv(DATA / "conferencia_log_vs_bu.csv", index=False)

# ---------------------------------------------------------------- 2. tabela por UF (T2 e controle T1)
def uf_table(t):
    g = s.groupby("uf")
    d = pd.DataFrame({"secoes": g.size(), "tz_off_mediana": g.tz_off.median(),
                      "votos": g[f"t{t}_n_vote"].sum()})
    for k in ("A", "B", "X"):
        d[f"voto_pos_{k}"] = g[f"t{t}_vote_{k}"].sum()
        d[f"pct_pos_{k}"] = pct(d[f"voto_pos_{k}"], d.votos)
        d[f"hab_pos_{k}"] = g[f"t{t}_hab_{k}"].sum()
    d["secoes_com_voto_pos_B"] = g[f"t{t}_vote_B"].apply(lambda x: (x > 0).sum())
    d["pct_secoes_com_voto_pos_B"] = 100 * d.secoes_com_voto_pos_B / d.secoes
    d["secoes_com_ge10_pos_B"] = g[f"t{t}_vote_B"].apply(lambda x: (x >= 10).sum())
    tot_ = d.sum(numeric_only=True); tot_["tz_off_mediana"] = np.nan
    for k in ("A", "B", "X"):
        tot_[f"pct_pos_{k}"] = 100 * tot_[f"voto_pos_{k}"] / tot_.votos
    tot_["pct_secoes_com_voto_pos_B"] = 100 * tot_.secoes_com_voto_pos_B / tot_.secoes
    d.loc["BRASIL"] = tot_
    return d

for t in (1, 2):
    uf_table(t).round(4).to_csv(DATA / f"tabela_uf_t{t}.csv")

# distribuição por hora de Brasília (hora local - off)
rows = []
for t in (1, 2):
    for h in range(24):
        v = s[f"t{t}_vote_h{h:02d}"]
        for uf, off in s.groupby("uf").tz_off.median().items():
            pass
    for uf, g in s.groupby("uf"):
        off = int(g.tz_off.median()) if g.tz_off.notna().any() else 0
        for h in range(24):
            rows.append((t, uf, h, (h - off) % 24, int(g[f"t{t}_vote_h{h:02d}"].sum()), int(g[f"t{t}_hab_h{h:02d}"].sum())))
hh = pd.DataFrame(rows, columns=["turno", "uf", "hora_local", "hora_brasilia", "votos", "habilitacoes"])
hh.to_csv(DATA / "votos_por_hora_uf.csv", index=False)

# ---------------------------------------------------------------- 3. limite superior
r = {"margem_oficial_Lula_menos_Flavio": MARGEM, "limiar_remocao (n > margem)": MARGEM, "limiar_inversao (n > margem/2)": MARGEM / 2}
for k, nome in (("A", "17h_relogio_urna"), ("B", "17h_brasilia"), ("X", "19h_brasilia")):
    for basis, bn in (("vote", "voto_computado"), ("hab", "habilitacao")):
        n = int(s[f"t2_{basis}_{k}"].sum())
        r[f"n_pos_{nome}_{bn}"] = n
        r[f"n_pos_{nome}_{bn}_pct_da_margem"] = 100 * n / MARGEM
        # pior caso 1: todos os n votos eram de Lula e nem existiriam (retirados): margem cai n
        r[f"margem_se_todos_n_de_Lula_retirados_{nome}_{bn}"] = MARGEM - n
        # pior caso 2: todos os n eram de Lula e, sem prorrogação, teriam ido a Flávio: margem cai 2n
        r[f"margem_se_todos_n_de_Lula_viram_Flavio_{nome}_{bn}"] = MARGEM - 2 * n
        r[f"muda_vencedor_{nome}_{bn}"] = "SIM" if (MARGEM - 2 * n) < 0 else "NAO"
pd.Series(r).to_csv(DATA / "limite_superior.csv", header=["valor"])

# ---------------------------------------------------------------- 4. biometria: antes x depois (estratos = seção)
def bio_tables(t, kind="B"):
    L = late[late.turno == t].copy()
    off = L.tz_off if kind == "B" else 0
    L = L[L.t_vote >= (17 + off) * 3600]
    g = L.groupby(["secid", "tipo"]).size().unstack(fill_value=0).reindex(columns=["B", "M", "U"], fill_value=0)
    g.columns = ["late_bio", "late_man", "late_unk"]
    d = s.set_index("secid")[["uf", "mun", "zona", f"t{t}_n_bio", f"t{t}_n_man", f"t{t}_n_unk"]].join(g, how="left").fillna(0)
    d["early_bio"] = d[f"t{t}_n_bio"] - d.late_bio
    d["early_nonbio"] = (d[f"t{t}_n_man"] + d[f"t{t}_n_unk"]) - (d.late_man + d.late_unk)
    d["late_nonbio"] = d.late_man + d.late_unk
    d["late_n"] = d.late_bio + d.late_nonbio
    d["early_n"] = d.early_bio + d.early_nonbio
    return d

def mh_or(d):
    """Mantel-Haenszel OR (não biométrico, tardio vs antes) com estratos = seção; IC de Robins-Breslow-Greenland."""
    a = d.late_nonbio.values; b = d.late_bio.values; c = d.early_nonbio.values; dd = d.early_bio.values
    n = a + b + c + dd
    m = (d.late_n > 0) & (d.early_n > 0)
    a, b, c, dd, n = a[m], b[m], c[m], dd[m], n[m]
    R = (a * dd / n); S = (b * c / n)
    OR = R.sum() / S.sum()
    P = (a + dd) / n; Q = (b + c) / n
    var = (P * R).sum() / (2 * R.sum() ** 2) + ((P * S + Q * R).sum()) / (2 * R.sum() * S.sum()) + (Q * S).sum() / (2 * S.sum() ** 2)
    se = np.sqrt(var)
    return OR, np.exp(np.log(OR) - 1.96 * se), np.exp(np.log(OR) + 1.96 * se), int(m.sum())

bio_rows = []
for t in (1, 2):
    for kind in ("B", "A"):
        d = bio_tables(t, kind)
        for uf, g in list(d.groupby("uf")) + [("BRASIL", d)]:
            ln, en = g.late_n.sum(), g.early_n.sum()
            if ln == 0:
                continue
            OR, lo, hi, ns = mh_or(g)
            bio_rows.append((t, kind, uf, int(ln), int(g.late_nonbio.sum()), 100 * g.late_nonbio.sum() / ln, int(en), int(g.early_nonbio.sum()),
                             100 * g.early_nonbio.sum() / en, OR, lo, hi, ns))
pd.DataFrame(bio_rows, columns=["turno", "corte", "uf", "n_tardio", "nao_bio_tardio", "pct_nao_bio_tardio", "n_antes", "nao_bio_antes",
                                "pct_nao_bio_antes", "OR_MH_secao", "OR_ic_inf", "OR_ic_sup", "secoes_estratos"]).round(4).to_csv(
    DATA / "biometria_antes_depois.csv", index=False)

# ---------------------------------------------------------------- 5. anomalias
a = pd.DataFrame({"secid": s.secid, "uf": s.uf})
an = {}
an["secoes_votos_fora_data_t2"] = int((s.t2_n_vote_off_date > 0).sum())
an["votos_fora_data_t2"] = int(s.t2_n_vote_off_date.sum())
an["secoes_tz_discordante_municipio"] = int(s.tz_mismatch.sum())
an["secoes_sem_evento_abertura"] = int(s.t2_open_s.isna().sum())
an["secoes_timestamp_nao_monotonico"] = int((s.t2_n_nonmono > 0).sum())
# voto computado em hora anterior à abertura nominal (08:00 BRT): hora local < 8+off
pre = np.zeros(len(s), dtype=float)
for h in range(24):
    pre += np.where(h < 8 + s.tz_off.fillna(0).values, s[f"t2_vote_h{h:02d}"].fillna(0).values, 0)
s["votos_antes_8h_brt"] = pre.astype(int)
an["secoes_votos_antes_8h_brt"] = int((pre > 0).sum()); an["votos_antes_8h_brt"] = int(pre.sum())
an["secoes_votos_apos_19h_brt"] = int((s.t2_vote_X > 0).sum()); an["votos_apos_19h_brt"] = int(s.t2_vote_X.sum())
an["secoes_log_bu_votos_divergentes"] = int(((s.t2_n_vote != s.bu_comp) & ok).sum())
an["secoes_habilitado_sem_voto"] = int((s.t2_n_hab_no_vote > 0).sum())
an["secoes_voto_sem_habilitacao"] = int((s.t2_n_vote_no_hab > 0).sum())
an["secoes_sem_log"] = int((~s.log_ok).sum())
pd.Series(an).to_csv(DATA / "anomalias_resumo.csv", header=["valor"])
top = s.assign(pct_pos_B=pct(s.t2_vote_B, s.t2_n_vote)).sort_values(["t2_vote_B"], ascending=False).head(200)
top[["secid", "uf", "tz_off", "t2_n_vote", "t2_vote_A", "t2_vote_B", "t2_vote_X", "pct_pos_B", "t2_hab_B", "t2_open_s", "t2_close_s",
     "t2_last_ev_s", "t2_n_hab_no_vote", "bu_comp", "tz_mismatch"]].to_csv(DATA / "secoes_mais_votos_pos_corte_B.csv", index=False)

# ---------------------------------------------------------------- 6. réplica do painel de terceiros (3.079 / 11.995.839 = 0,03% após 17h; sem biometria 16,3% vs 12,2%)
panel = []
for kind, col in (("A_17h_relogio_urna", "A"), ("B_17h_brasilia", "B")):
    for basis in ("vote", "hab"):
        n_late = s[f"t2_{basis}_{col}"].sum(); n_tot = s.t2_n_vote.sum()
        panel.append((kind, "voto computado" if basis == "vote" else "habilitacao", int(n_late), int(n_tot), 100 * n_late / n_tot))
pd.DataFrame(panel, columns=["corte", "base_temporal", "apos_corte", "total", "pct"]).round(5).to_csv(DATA / "replica_painel_global.csv", index=False)
print("ok", tot)
