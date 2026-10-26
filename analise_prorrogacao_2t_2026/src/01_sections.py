"""Lista as seções do 2º turno (e 1º) a partir de arquivo-urna/<pleito>/config/<uf>/<uf>-p<pleito>-cs.json."""
import csv, json, ssl, sys, urllib.request
from common import *

ctx = ssl.create_default_context(cafile=CA_BUNDLE)


def get(url):
    with urllib.request.urlopen(url, context=ctx, timeout=60) as r:
        return r.read()


def main(turno=2):
    out = DATA / f"sections_t{turno}.csv"
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uf", "mun", "zona", "secao", "cs_da", "cs_ha"])
        for uf in UFS:
            raw = get(cs_url(uf, turno))
            (RAW_DIR / f"cs_t{turno}_{uf}.json").write_bytes(raw)
            d = json.loads(raw)
            for a in d["abr"]:
                for mu in a["mu"]:
                    for z in mu["zon"]:
                        for s in z["sec"]:
                            w.writerow([uf, mu["cd"], z["cd"], s["ns"], s.get("da"), s.get("ha")])
                            n += 1
            print(uf, n, flush=True)
    print("total seções:", n, "->", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
