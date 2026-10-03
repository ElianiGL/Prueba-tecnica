# Cálculo de métricas por pregunta a partir de la respuesta

from __future__ import annotations
import math
import re

def _art(clave: str) -> str:
    p = clave.split(":")
    if p[0] == "CPEUM" and len(p) >= 3 and p[1] == "123":
        return ":".join(p[:3])
    return ":".join(p[:2]) if p[1] != "TRANS" else ":".join(p[:3])

def _cubre(claves_frag: list[str], oro: str, nivel: str) -> bool:
    if nivel == "articulo":
        return _art(oro) in {_art(c) for c in claves_frag}
    return oro in claves_frag

def metricas_recuperacion(fragmentos: list[dict], oro: list[str]) -> dict:
    if not oro:
        return {"recall_articulo": math.nan, "recall_fraccion": math.nan, "acierto_k": math.nan, "mrr": math.nan}
    claves = [f["claves"] for f in fragmentos]
    arts = {_art(o) for o in oro}
    enc_art = {a for a in arts if any(_cubre(c, a, "articulo") for c in claves)}
    enc_fr = [o for o in oro if any(_cubre(c, o, "fraccion") or (_art(o) == o and _cubre(c, o, "articulo"))
                                    for c in claves)]
    primer = next((r for r, c in enumerate(claves, 1) if any(_cubre(c, a, "articulo") for a in arts)), None)
    return {
        "recall_articulo": len(enc_art) / len(arts),
        "recall_fraccion": len(enc_fr) / len(oro),
        "acierto_k": float(bool(enc_art)),
        "mrr": 1.0 / primer if primer else 0.0,
    }

def precision_citas(fundamentos: list[dict], fragmentos: list[dict], oro: list[str]) -> float:
    validos = [f for f in fundamentos if f.get("valida")]
    if not validos or not oro:
        return math.nan
    por_id = {f["id"]: f for f in fragmentos}
    arts_oro = {_art(o) for o in oro}
    citados = {_art(por_id[f["fragmento_id"]]["claves"][0]) for f in validos if f["fragmento_id"] in por_id}
    return len(citados & arts_oro) / len(citados) if citados else math.nan

def fidelidad(afirmaciones: list[dict], veredictos: list[dict], fragmentos: list[dict],
              fundamentos: list[dict]) -> dict:
    factuales = [i for i, a in enumerate(afirmaciones) if a.get("tipo") == "factual"]
    n = len(factuales)
    if n == 0:
        return {"n_afirmaciones": 0, "n_respaldadas": 0, "n_contradichas": 0,
                "fidelidad": math.nan, "fidelidad_estricta": math.nan}
    ids_citados = {f["fragmento_id"] for f in fundamentos if f.get("valida")}
    pos_a_id = {f"F{i}": f["id"] for i, f in enumerate(fragmentos, 1)}
    resp = estricta = contra = 0
    for i in factuales:
        v = veredictos[i] if i < len(veredictos) else {"etiqueta": "no_respaldada", "fragmentos": []}
        if v["etiqueta"] == "respaldada":
            resp += 1
            soporte = {pos_a_id.get(re.sub(r"\s", "", s).upper()) for s in v.get("fragmentos", [])}
            estricta += bool(soporte & ids_citados)
        elif v["etiqueta"] == "contradicha":
            contra += 1
    return {"n_afirmaciones": n, "n_respaldadas": resp, "n_contradichas": contra,
            "fidelidad": resp / n, "fidelidad_estricta": estricta / n}

def correccion(calif: dict | None, n_puntos: int) -> dict:
    if not calif:
        return {"cobertura": math.nan, "contradice": math.nan, "relevancia": math.nan, "dato_inventado": math.nan}
    ver = {p.get("indice"): p.get("veredicto") for p in calif.get("puntos", [])}
    vs = [ver.get(i, "omitido") for i in range(n_puntos)]
    return {
        "cobertura": sum(v == "cubierto" for v in vs) / n_puntos if n_puntos else math.nan,
        "contradice": float(any(v == "contradicho" for v in vs)),
        "relevancia": (calif.get("relevancia", 3) - 1) / 4,  # 1..5 -> 0..1
        "dato_inventado": float(bool(calif.get("dato_inventado"))),
    }

def abstencion(estado: str, tipo_oro: str) -> dict:
    abst = estado == "sin_respuesta"
    return {
        "se_abstuvo": float(abst),
        "abstencion_correcta": float(abst) if tipo_oro == "no_respondible" else math.nan,
        "abstencion_indebida": float(abst) if tipo_oro != "no_respondible" else math.nan,
        "invento_fuera_de_alcance": float(not abst) if tipo_oro == "no_respondible" else math.nan,
    }

def exito(fila: dict) -> float:
    if fila["tipo"] == "no_respondible":
        return float(fila["estado"] == "sin_respuesta")
    if fila["estado"] == "sin_respuesta":
        return 0.0
    fid = fila.get("fidelidad")
    fid_ok = (fid is None or (isinstance(fid, float) and math.isnan(fid)) or fid >= 0.8)
    return float(fila.get("contradice", 1) == 0 and fila.get("cobertura", 0) >= 0.5 and fid_ok)