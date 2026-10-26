"""Parser do log da urna (log.jez -> logd.dat).

Formato (descoberto por inspeção, ver README): arquivo ZIP com 'logd.dat', texto ISO-8859-1,
linhas TSV:  dd/mm/aaaa hh:mm:ss <TAB> NÍVEL <TAB> nº urna <TAB> MÓDULO <TAB> mensagem <TAB> hash
O log é cumulativo (preparação, 1º turno em 04/10, 2º turno em 25/10). Horários = relógio LOCAL da urna.

Eventos usados (módulo VOTA), significado validado contra o BU (ver README/validação):
  'Urna pronta para receber votos'                  -> abertura
  'Inicio do Encerramento'                          -> fechamento (encerramento iniciado no mesário)
  'Eleitor foi habilitado'                          -> habilitação do eleitor (t_hab)
  'Tipo de habilitação do eleitor [biométrica]'     -> habilitação biométrica (precede 'Eleitor foi habilitado')
  'Solicitação de dado pessoal ... habilitação manual' -> habilitação não biométrica (manual)
  'O voto do eleitor foi computado'                 -> voto computado (t_vote)
O log NÃO registra em quem o eleitor votou.
"""
import io
import re
import zipfile
from collections import Counter
from datetime import datetime

ELECTION_DATE = {1: "04/10/2026", 2: "25/10/2026"}
_REF = {t: datetime.strptime(d, "%d/%m/%Y") for t, d in ELECTION_DATE.items()}
_TURNO_RE = re.compile(r"(\d)º turno")
_DIG = re.compile(r"\d+")
LATE_FROM = 14 * 3600  # guarda detalhe de cada voto a partir das 14h locais


def _secs(dt, turno):
    """'dd/mm/aaaa hh:mm:ss' -> (segundos desde 00:00 do dia da eleição do turno, mesma_data?)."""
    date = dt[:10]
    s = int(dt[11:13]) * 3600 + int(dt[14:16]) * 60 + int(dt[17:19])
    if date == ELECTION_DATE[turno]:
        return s, True
    try:
        off = (datetime.strptime(date, "%d/%m/%Y") - _REF[turno]).days
    except ValueError:
        return None, False
    return s + 86400 * off, False


def _new_acc():
    return {
        "open_s": None, "close_s": None, "first_ev_s": None, "last_ev_s": None,
        "n_hab": 0, "n_vote": 0, "n_bio": 0, "n_man": 0, "n_unk": 0,
        "n_vote_no_hab": 0, "n_hab_no_vote": 0, "n_off_date": 0, "n_vote_off_date": 0, "n_nonmono": 0,
        "n_erro": 0, "n_alerta": 0, "n_open_events": 0, "n_close_events": 0,
        "vote_h": [0] * 24, "vote_bio_h": [0] * 24, "hab_h": [0] * 24,
        "late": [], "_prev_s": None,
    }


def parse_logd(text, want_catalog=True):
    """text: conteúdo decodificado de logd.dat. Retorna (resumo por turno, Counter de tipos de evento)."""
    acc = {1: _new_acc(), 2: _new_acc()}
    catalog = Counter()
    turno = None          # turno da sessão VOTA corrente
    n_lines = 0
    urna = None
    bio_flag = man_flag = False
    hab_s = hab_type = None
    for line in text.split("\n"):
        f = line.rstrip("\r").split("\t")
        if len(f) < 5:
            continue
        n_lines += 1
        mod, msg = f[3], f[4]
        if mod == "LOGD" and msg.startswith("Urna ligada em"):
            turno = None
            bio_flag = man_flag = False
            hab_s = hab_type = None
            urna = f[2]
            continue
        if mod != "VOTA":
            continue
        if msg.startswith("Iniciando aplicação"):
            m = _TURNO_RE.search(msg)
            turno = int(m.group(1)) if m else None
            bio_flag = man_flag = False
            hab_s = hab_type = None
            continue
        if turno not in (1, 2):
            continue
        a = acc[turno]
        s, same = _secs(f[0], turno)
        if want_catalog and turno == 2:
            catalog[(f[1], _DIG.sub("#", msg)[:140])] += 1
        if f[1] == "ERRO":
            a["n_erro"] += 1
        elif f[1] == "ALERTA":
            a["n_alerta"] += 1
        if s is None:
            continue
        if not same:
            a["n_off_date"] += 1
        else:
            if a["first_ev_s"] is None or s < a["first_ev_s"]:
                a["first_ev_s"] = s
            if a["last_ev_s"] is None or s > a["last_ev_s"]:
                a["last_ev_s"] = s
            if a["_prev_s"] is not None and s < a["_prev_s"]:
                a["n_nonmono"] += 1
            a["_prev_s"] = s

        if msg == "Urna pronta para receber votos":
            a["n_open_events"] += 1
            if a["open_s"] is None and same:
                a["open_s"] = s
        elif msg == "Inicio do Encerramento":
            a["n_close_events"] += 1
            if same:
                a["close_s"] = s
        elif msg == "Aguardando digitação do identificador do eleitor":
            if hab_s is not None:        # habilitado e voltou ao início sem voto computado
                a["n_hab_no_vote"] += 1
                hab_s = hab_type = None
            bio_flag = man_flag = False
        elif msg.startswith("Tipo de habilitação do eleitor [biom"):
            bio_flag = True
        elif msg.startswith("Solicitação de dado pessoal do eleitor para habilitação manual"):
            man_flag = True
        elif msg == "Eleitor foi habilitado":
            a["n_hab"] += 1
            hab_s = s
            hab_type = "B" if bio_flag else ("M" if man_flag else "U")
            if 0 <= s < 86400:
                a["hab_h"][s // 3600] += 1
        elif msg == "O voto do eleitor foi computado":
            a["n_vote"] += 1
            if hab_s is None:
                a["n_vote_no_hab"] += 1
                typ = "U"
            else:
                typ = hab_type
            a["n_" + {"B": "bio", "M": "man", "U": "unk"}[typ]] += 1
            if not same:
                a["n_vote_off_date"] += 1
            if 0 <= s < 86400:
                a["vote_h"][s // 3600] += 1
                if typ == "B":
                    a["vote_bio_h"][s // 3600] += 1
            if s >= LATE_FROM:
                a["late"].append((-1 if hab_s is None else hab_s, s, typ))
            hab_s = hab_type = None
            bio_flag = man_flag = False
    return acc, catalog, n_lines, urna


def read_logjez(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    return z.read("logd.dat" if "logd.dat" in names else names[0]).decode("latin-1")


def parse_logjez(blob, want_catalog=True):
    return parse_logd(read_logjez(blob), want_catalog)
