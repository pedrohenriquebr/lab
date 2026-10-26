"""Baixa aux.json + log.jez + bu.dat de cada seção do 2º turno, faz o parse em memória e grava
resumos por seção (parquet) em chunks reprocessáveis. Não guarda o log bruto (só bu.dat, ~1,5 KB).

Uso: python 02_crawl.py UF [UF ...] [--conc 24] [--chunk 4000] [--limit N]
Saída: RAW_DIR/out/<uf>/chunk_XXXX.{summary,late}.parquet, .catalog.json, .bu.zip
Sem User-Agent falso: identifica-se como cliente de pesquisa. Respeita o servidor (concorrência limitada, retry com backoff).
"""
import argparse, asyncio, csv, json, ssl, sys, time, zipfile, io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import pandas as pd
from common import *
import logparse, buparse

UA = "pesquisa-prorrogacao-2t-2026/1.0 (analise de dados publicos TSE)"


def work(uf, mun, zona, sec, log_blob, bu_blob):
    """Roda em processo separado: parse de log e BU de uma seção."""
    row = {"uf": uf, "mun": mun, "zona": zona, "secao": sec}
    late = []
    cat = None
    if log_blob is not None:
        try:
            acc, cat, n_lines, urna = logparse.parse_logjez(log_blob)
            row["urna"] = urna; row["log_lines"] = n_lines
            for t in (1, 2):
                a = acc[t]
                for k, v in a.items():
                    if k in ("late", "_prev_s"):
                        continue
                    if k in ("vote_h", "vote_bio_h", "hab_h"):
                        for h, x in enumerate(v):
                            row[f"t{t}_{k}{h:02d}"] = x
                    else:
                        row[f"t{t}_{k}"] = v
                for th, ts, typ in a["late"]:
                    late.append((t, th, ts, typ))
            row["log_ok"] = True
        except Exception as e:
            row["log_ok"] = False; row["log_err"] = repr(e)[:200]
    if bu_blob is not None:
        try:
            b = buparse.parse_bu(bu_blob)
            v = b["votos"]
            row.update({"bu_found": b["found"], "bu_abertura": b["abertura"], "bu_encerramento": b["encerramento"],
                        "bu_comp": b["bu_a"], "bu_bio": b["bu_bio"], "bu_man": b["bu_man"],
                        "v13": v["nominal"].get(13, 0), "v22": v["nominal"].get(22, 0),
                        "v_outros": sum(q for c, q in v["nominal"].items() if c not in (13, 22)),
                        "v_branco": v["branco"], "v_nulo": v["nulo"], "v_legenda": v["legenda"]})
            row["bu_ok"] = True
        except Exception as e:
            row["bu_ok"] = False; row["bu_err"] = repr(e)[:200]
    return row, late, cat


async def fetch(session, url, tries=4):
    last = None
    for k in range(tries):
        try:
            async with session.get(url) as r:
                if r.status == 200:
                    return 200, await r.read()
                if r.status in (404, 403):
                    return r.status, None
                last = r.status
        except Exception as e:
            last = repr(e)[:80]
        await asyncio.sleep(1.5 * (2 ** k))
    return last, None


async def one(session, sem, pool, loop, uf, mun, zona, sec):
    async with sem:
        base = sec_base(uf, mun, zona, sec)
        st, aux = await fetch(session, aux_url(uf, mun, zona, sec))
        meta = {"aux_status": st}
        if st != 200:
            return {"uf": uf, "mun": mun, "zona": zona, "secao": sec, **meta}, [], None, None
        d = json.loads(aux)
        hs = d.get("hashes", [])
        meta["n_hashes"] = len(hs); meta["aux_st"] = d.get("st")
        # escolhe o hash mais recente (dr/hr) que contenha log e bu
        def key(h):
            return (h.get("dr", "")[6:10] + h.get("dr", "")[3:5] + h.get("dr", "")[0:2], h.get("hr", ""))
        cands = [h for h in hs if any(a["tp"] == "log" for a in h["arq"])]
        if not cands:
            return {"uf": uf, "mun": mun, "zona": zona, "secao": sec, **meta, "log_status": "sem_log"}, [], None, None
        h = sorted(cands, key=key)[-1]
        meta["hash_dr"] = h.get("dr"); meta["hash_hr"] = h.get("hr")
        files = {a["tp"]: a["nm"] for a in h["arq"]}
        (sl, log_blob), (sb, bu_blob) = await asyncio.gather(
            fetch(session, f"{base}/{h['hash']}/{files['log']}"),
            fetch(session, f"{base}/{h['hash']}/{files['bu']}") if "bu" in files else asyncio.sleep(0, (None, None)))
        meta["log_status"] = sl; meta["bu_status"] = sb
        row, late, cat = await loop.run_in_executor(pool, work, uf, mun, zona, sec, log_blob, bu_blob)
        row.update(meta)
        return row, late, cat, bu_blob


async def run_chunk(session, sem, pool, loop, uf, rows, outdir, ci):
    res = await asyncio.gather(*[one(session, sem, pool, loop, uf, *r) for r in rows])
    summ, late_rows, cat, zbuf = [], [], Counter(), io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as z:
        for idx, (row, late, c, bu) in enumerate(res):
            summ.append(row)
            sid = f"{row['mun']}-{row['zona']}-{row['secao']}"
            for t, th, ts, typ in late:
                late_rows.append((uf, row["mun"], row["zona"], row["secao"], t, th, ts, typ))
            if c:
                cat.update(c)
            if bu:
                z.writestr(sid, bu)
    pd.DataFrame(summ).to_parquet(outdir / f"chunk_{ci:04d}.summary.parquet", index=False)
    pd.DataFrame(late_rows, columns=["uf", "mun", "zona", "secao", "turno", "t_hab", "t_vote", "tipo"]).to_parquet(
        outdir / f"chunk_{ci:04d}.late.parquet", index=False)
    (outdir / f"chunk_{ci:04d}.catalog.json").write_text(json.dumps([[k[0], k[1], v] for k, v in cat.items()]))
    (outdir / f"chunk_{ci:04d}.bu.zip").write_bytes(zbuf.getvalue())
    # marcador de conclusão do chunk (escrito por último)
    (outdir / f"chunk_{ci:04d}.done").write_text(str(len(summ)))


async def main(args):
    ctx = ssl.create_default_context(cafile=CA_BUNDLE)
    conn = aiohttp.TCPConnector(limit=args.conc, ssl=ctx)
    to = aiohttp.ClientTimeout(total=90)
    secs = list(csv.DictReader(open(DATA / "sections_t2.csv")))
    loop = asyncio.get_running_loop()
    pool = ProcessPoolExecutor(max_workers=args.procs)
    sem = asyncio.Semaphore(args.conc)
    async with aiohttp.ClientSession(connector=conn, timeout=to, trust_env=True, headers={"User-Agent": UA}) as session:
        for uf in args.ufs:
            rows = [(s["mun"], s["zona"], s["secao"]) for s in secs if s["uf"] == uf]
            if args.limit:
                rows = rows[: args.limit]
            outdir = RAW_DIR / "out" / uf
            outdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            nb = 0
            for ci, i in enumerate(range(0, len(rows), args.chunk)):
                if (outdir / f"chunk_{ci:04d}.done").exists():
                    continue
                await run_chunk(session, sem, pool, loop, uf, rows[i:i + args.chunk], outdir, ci)
                nb += len(rows[i:i + args.chunk])
                el = time.time() - t0
                print(f"{uf} chunk {ci} ok  {nb}/{len(rows)}  {nb/el:.1f} seç/s", flush=True)
    pool.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ufs", nargs="+")
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
