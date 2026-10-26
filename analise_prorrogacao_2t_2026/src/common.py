"""Constantes e caminhos compartilhados. Dados brutos ficam fora do repositório (RAW_DIR)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "raw"))  # não versionado
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://resultados.tse.jus.br/oficial/ele2026"
PLEITO = {1: 3220, 2: 3221}          # cd_pleito 1º e 2º turno (ele-c.json)
ELEICAO_PRES = {1: 6257, 2: 6258}    # cd_eleicao Presidente
ELECTION_DATE = {1: "04/10/2026", 2: "25/10/2026"}
CA_BUNDLE = os.environ.get("CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

UFS = ["ac", "al", "am", "ap", "ba", "ce", "df", "es", "go", "ma", "mg", "ms", "mt", "pa",
       "pb", "pe", "pi", "pr", "rj", "rn", "ro", "rr", "rs", "sc", "se", "sp", "to", "zz"]


def cs_url(uf, turno=2):
    p = PLEITO[turno]
    return f"{BASE}/arquivo-urna/{p}/config/{uf}/{uf}-p{p:06d}-cs.json"


def sec_base(uf, mun, zona, sec, turno=2):
    p = PLEITO[turno]
    return f"{BASE}/arquivo-urna/{p}/dados/{uf}/{mun}/{zona}/{sec}"


def aux_url(uf, mun, zona, sec, turno=2):
    p = PLEITO[turno]
    return f"{sec_base(uf, mun, zona, sec, turno)}/p{p:06d}-{uf}-m{mun}-z{zona}-s{sec}-aux.json"
