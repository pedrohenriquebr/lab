"""Análise ecológica (por seção): mudança de voto 1º→2º turno vs. fração de votos tardios (após o corte B).
ATENÇÃO: o log não registra o voto. Isto é associação entre seções; NÃO é inferência sobre indivíduos nem causal.

Desfecho: dL = parcela de Lula nos votos válidos (13 vs 22) no 2º turno − parcela de Lula nos votos válidos do 1º turno
          (válidos 1º turno = todos os candidatos nominais; muda o conjunto de candidatos, por isso a unidade é "parcela de Lula").
Exposição: ell = votos computados após o corte B (17h Brasília) / votos computados da seção (2º turno).
Modelos: WLS (peso = votos válidos 2T) com efeito fixo de zona eleitoral (município+zona), SE robusto agrupado por zona.
"""
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import *

s = pd.read_parquet(RAW_DIR / "secoes.parquet")
s = s[(s.uf != "zz") & s.bu_ok.fillna(False) & s.log_ok.fillna(False)].copy()
b = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RAW_DIR / "b1" / "*.parquet")))], ignore_index=True)
b["uf"] = b.SG_UF.str.lower()
key = ["uf", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
n = b.NR_VOTAVEL
b["is_valid"] = ~n.isin([95, 96])
agg = b.groupby(key).apply(lambda g: pd.Series({
    "v13_1": g.loc[g.NR_VOTAVEL == 13, "QT_VOTOS"].sum(), "v22_1": g.loc[g.NR_VOTAVEL == 22, "QT_VOTOS"].sum(),
    "valid1": g.loc[g.is_valid, "QT_VOTOS"].sum(), "aptos1": g.QT_APTOS.iloc[0], "comp1": g.QT_COMPARECIMENTO.iloc[0]}), include_groups=False).reset_index()
s["CD_MUNICIPIO"] = s.mun.astype(int); s["NR_ZONA"] = s.zona.astype(int); s["NR_SECAO"] = s.secao.astype(int)
d = s.merge(agg, on=key, how="inner")
print("seções 2T com log+BU:", len(s), " casadas com 1T:", len(d))
d["valid2"] = d.v13 + d.v22
d = d[(d.valid1 >= 30) & (d.valid2 >= 30) & (d.aptos1 > 0)].copy()
d["L1"] = d.v13_1 / d.valid1; d["F1"] = d.v22_1 / d.valid1; d["L2"] = d.v13 / d.valid2
d["dL"] = d.L2 - d.L1
d["turn1"] = d.comp1 / d.aptos1; d["turn2"] = d.bu_comp / d.aptos1; d["laptos"] = np.log(d.aptos1)
d["ell"] = d.t2_vote_B / d.t2_n_vote
d["ell1"] = d.t1_vote_B / d.t1_n_vote.replace(0, np.nan)
d["late_any"] = (d.t2_vote_B > 0).astype(float)
d["zid"] = d.uf + "-" + d.mun + "-" + d.zona
w = d.valid2.astype(float)


def within(x, g, w):
    m = (x * w).groupby(g).transform("sum") / w.groupby(g).transform("sum")
    return x - m


def fit(yv, xs, label, data=d):
    dd = data.dropna(subset=[yv] + xs)
    ww = dd.valid2.astype(float)
    Y = within(dd[yv], dd.zid, ww); X = pd.concat([within(dd[c], dd.zid, ww) for c in xs], axis=1)
    r = sm.WLS(Y, X, weights=ww).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.zid)[0]})
    out = []
    for c in xs:
        out.append((label, c, r.params[c], r.bse[c], r.params[c] - 1.96 * r.bse[c], r.params[c] + 1.96 * r.bse[c], r.pvalues[c], int(r.nobs),
                    dd.zid.nunique()))
    return out


res = []
res += fit("dL", ["ell"], "FE zona")
res += fit("dL", ["ell", "L1", "F1", "turn1", "turn2", "laptos"], "FE zona + controles (1T, comparecimento, tamanho)")
res += fit("dL", ["ell", "ell1", "L1", "F1", "turn1", "turn2", "laptos"], "idem + fração tardia no 1º turno (placebo/controle)")
res += fit("dL", ["late_any", "L1", "F1", "turn1", "turn2", "laptos"], "FE zona + controles; indicador de ≥1 voto tardio")
sub = d[d.t2_vote_B >= 5]
zs = set(sub.zid)
res += fit("dL", ["ell", "L1", "F1", "turn1", "turn2", "laptos"], "somente zonas com ≥1 seção com ≥5 votos tardios", d[d.zid.isin(zs)])
R = pd.DataFrame(res, columns=["modelo", "variavel", "coef", "ep", "ic95_inf", "ic95_sup", "p", "n_secoes", "n_zonas"])
R.round(6).to_csv(DATA / "ecologica_regressao.csv", index=False)

# conversão de β em votos: β = diferença (pontos de parcela de Lula) entre quem votou tarde e os demais, na mesma zona e nível de controles
nl = d.t2_vote_B.sum()
cv = R[(R.modelo.str.startswith("FE zona + controles (")) & (R.variavel == "ell")].iloc[0]
imp = {"votos_tardios_(amostra_ecologica)": int(nl)}
for nm, v in (("coef", cv.coef), ("ic_inf", cv.ic95_inf), ("ic_sup", cv.ic95_sup)):
    imp[f"efeito_liquido_na_margem_{nm}"] = 2 * v * nl   # 2·β·n: desvio de β na parcela de Lula entre n votos → margem Lula−Flávio muda 2βn
pd.Series(imp).to_csv(DATA / "ecologica_implicacao_votos.csv", header=["valor"])

# pareamento: seção com ≥5 votos tardios vs. seções da MESMA zona sem voto tardio, vizinho mais próximo em (L1, F1, turn1, laptos)
def match(treated_min=5):
    rows = []
    X = ["L1", "F1", "turn1", "laptos"]
    sd = d[X].std()
    for z, g in d.groupby("zid"):
        t = g[g.t2_vote_B >= treated_min]; c = g[g.t2_vote_B == 0]
        if len(t) == 0 or len(c) == 0:
            continue
        for i, r_ in t.iterrows():
            dist = (((c[X] - r_[X]) / sd) ** 2).sum(1)
            j = dist.idxmin()
            if dist[j] < 1.0:     # caliper
                rows.append((z, r_.dL, c.loc[j, "dL"], r_.valid2, r_.t2_vote_B / r_.t2_n_vote))
    return pd.DataFrame(rows, columns=["zid", "dL_t", "dL_c", "w", "ell"])

P = match()
if len(P):
    P["diff"] = P.dL_t - P.dL_c
    est = np.average(P["diff"], weights=P.w)
    rng = np.random.default_rng(1)
    zs_ = P.zid.unique(); grp = {z: g for z, g in P.groupby("zid")}
    bs = []
    for _ in range(1000):
        pick = rng.choice(zs_, len(zs_), replace=True)
        gg = pd.concat([grp[z] for z in pick])
        bs.append(np.average(gg["diff"], weights=gg.w))
    pd.Series({"pares": len(P), "zonas": len(zs_), "dif_media_dL (pontos de parcela de Lula)": est,
               "ic95_inf": np.percentile(bs, 2.5), "ic95_sup": np.percentile(bs, 97.5)}).to_csv(DATA / "ecologica_pareamento.csv", header=["valor"])
    print(P["diff"].describe(), est, np.percentile(bs, [2.5, 97.5]))
d[["uf", "mun", "zona", "secao", "L1", "F1", "L2", "dL", "ell", "ell1", "t2_vote_B", "t2_n_vote", "valid2"]].to_parquet(RAW_DIR / "eco_base.parquet", index=False)
print(R.round(5).to_string())
