#Valida el conjunto de preguntas contra el texto descargado.

from __future__ import annotations
import json
import re
import unicodedata
from laboral.parseo import Fragmento
from .metricas import _art

def _n(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w%]+", " ", s)).strip()

def validar(preguntas: list[dict], fragmentos: list[Fragmento]) -> dict:
    textos_por_art: dict[str, str] = {}
    for f in fragmentos:
        for c in f.claves:
            textos_por_art[_art(c)] = textos_por_art.get(_art(c), "") + " " + _n(f.texto)
    problemas = []
    for q in preguntas:
        for o in q.get("articulos_esperados", []):
            if _art(o) not in textos_por_art:
                problemas.append({"id": q["id"], "tipo": "articulo_no_encontrado", "detalle": o})
        for art, ancla in (q.get("anclas") or {}).items():
            if ancla and _n(ancla) not in textos_por_art.get(_art(art), ""):
                problemas.append({"id": q["id"], "tipo": "ancla_no_encontrada", "detalle": f"{art}: '{ancla}'"})
    return {"n_preguntas": len(preguntas), "n_problemas": len(problemas), "problemas": problemas}

if __name__ == "__main__":  # python -m evaluacion.validar
    from laboral.config import DIR_DATA
    from laboral.parseo import construir_corpus
    preguntas = json.loads((DIR_DATA / "preguntas.json").read_text(encoding="utf-8"))
    print(json.dumps(validar(preguntas, construir_corpus("fraccion", 2200)), indent=2, ensure_ascii=False))