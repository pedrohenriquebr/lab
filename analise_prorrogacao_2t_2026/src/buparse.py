"""Parser mínimo de BER para bu.dat (Boletim de Urna), sem schema ASN.1.

Estrutura descoberta por inspeção (ver README):
  envelope SEQUENCE -> OCTET STRING (aninhado) -> EntidadeBoletimUrna
  [2:0] SEQUENCE{'aaaammddThhmmss' abertura, 'aaaammddThhmmss' encerramento}  (relógio local da urna)
  bloco de resultado da eleição: SEQUENCE iniciada pelo INTEGER cd_eleicao (6258 = Presidente 2º turno)
     votos: SEQUENCE com [2:1]=tipo (1 nominal, 2 branco, 3 nulo), [2:2]=quantidade, [2:3]={..., código}
"""
import re

_DT = re.compile(rb"^\d{8}T\d{6}$")


def _dec(b, i=0, end=None):
    end = len(b) if end is None else end
    out = []
    while i < end:
        t = b[i]; i += 1
        cls, cons, tag = t >> 6, (t >> 5) & 1, t & 0x1F
        if tag == 0x1F:
            tag = 0
            while True:
                x = b[i]; i += 1
                tag = (tag << 7) | (x & 0x7F)
                if not x & 0x80:
                    break
        l = b[i]; i += 1
        if l & 0x80:
            n = l & 0x7F
            l = int.from_bytes(b[i:i + n], "big"); i += n
        if cons:
            out.append((cls, tag, _dec(b, i, i + l)))
        else:
            out.append((cls, tag, b[i:i + l]))
        i += l
    return out


def _int(v):
    return int.from_bytes(v, "big", signed=True)


def _walk(nodes):
    for n in nodes:
        yield n
        if isinstance(n[2], list):
            yield from _walk(n[2])


def parse_bu(blob, cd_eleicao=6258):
    top = _dec(blob)
    root = top[0][2]
    inner = None
    for cls, tag, v in root:
        if cls == 0 and tag == 4 and isinstance(v, (bytes, bytearray)) and v[:1] == b"\x30":
            inner = _dec(v)[0][2]
    if inner is None:
        raise ValueError("BU sem envelope aninhado")
    res = {"abertura": None, "encerramento": None, "bu_a": None, "bu_bio": None, "bu_man": None}
    for cls, tag, v in inner:
        if cls == 2 and tag == 0 and isinstance(v, list):
            ds = [x[2].decode() for x in v if x[1] == 27 and _DT.match(x[2])]
            if len(ds) == 2:
                res["abertura"], res["encerramento"] = ds
        elif cls == 0 and tag == 2:
            res["bu_a"] = _int(v)
        elif cls == 2 and tag == 1 and isinstance(v, list):
            ints = [_int(x[2]) for x in v if x[1] == 2]
            if len(ints) == 3:
                res["bu_a_t"], res["bu_bio"], res["bu_man"] = ints
    votos = {"nominal": {}, "branco": 0, "nulo": 0, "legenda": 0}
    found = False
    for cls, tag, v in _walk(inner):
        if cls == 0 and tag == 16 and isinstance(v, list) and v and v[0][1] == 2 and v[0][0] == 0 \
                and isinstance(v[0][2], (bytes, bytearray)) and _int(v[0][2]) == cd_eleicao:
            found = True
            for c2, t2, v2 in _walk(v):
                if c2 == 0 and t2 == 16 and isinstance(v2, list):
                    kinds = {(x[0], x[1]): x[2] for x in v2}
                    if (2, 1) in kinds and (2, 2) in kinds:
                        tipo = _int(kinds[(2, 1)]); q = int.from_bytes(kinds[(2, 2)], "big")
                        if tipo == 1:
                            ident = kinds.get((2, 3))
                            code = None
                            if isinstance(ident, list):
                                ints = [_int(x[2]) for x in ident if x[1] == 2]
                                code = ints[-1] if ints else None
                            votos["nominal"][code] = votos["nominal"].get(code, 0) + q
                        elif tipo == 2:
                            votos["branco"] += q
                        elif tipo == 3:
                            votos["nulo"] += q
                        else:
                            votos["legenda"] += q
            break
    res["found"] = found
    res["votos"] = votos
    return res
