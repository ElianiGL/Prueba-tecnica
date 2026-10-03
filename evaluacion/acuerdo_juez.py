#Validación del juez contra calificación humana (reto 4).

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from laboral.config import DIR_RESULTADOS

RUTA_MUESTRA = DIR_RESULTADOS / "anotacion_manual.csv"
RUTA_JUEZ_OCULTO = DIR_RESULTADOS / "anotacion_juez_oculto.csv"

def generar_muestra(detalle_afirmaciones: pd.DataFrame, n: int = 40, semilla: int = 7,
                    sobrescribir: bool = False) -> Path:
    if RUTA_MUESTRA.exists() and not sobrescribir:
        previo = pd.read_csv(RUTA_MUESTRA)
        if previo["etiqueta_humana"].notna().any():
            print(f"[acuerdo] {RUTA_MUESTRA.name} ya tiene etiquetas humanas; no se sobrescribe.")
            return RUTA_MUESTRA
    df = detalle_afirmaciones[detalle_afirmaciones.tipo_afirmacion == "factual"].copy()
    rng = np.random.default_rng(semilla)
    no_resp = df[df.etiqueta_juez != "respaldada"]
    resp = df[df.etiqueta_juez == "respaldada"]
    k_no = min(len(no_resp), max(n // 3, n - len(resp)))
    sel = pd.concat([no_resp.sample(k_no, random_state=semilla) if k_no else no_resp.iloc[:0],
                     resp.sample(min(len(resp), n - k_no), random_state=semilla)])
    sel = sel.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    sel["item"] = [f"A{i:03d}" for i in range(len(sel))]
    sel["etiqueta_humana"] = ""
    sel["nota_humana"] = ""
    cols = ["item", "id", "version", "pregunta", "afirmacion", "fragmentos_contexto", "etiqueta_humana", "nota_humana"]
    sel[cols].to_csv(RUTA_MUESTRA, index=False)
    sel[["item", "etiqueta_juez", "justificacion_juez"]].to_csv(RUTA_JUEZ_OCULTO, index=False)
    print(f"[acuerdo] Muestra de {len(sel)} afirmaciones en {RUTA_MUESTRA}")
    return RUTA_MUESTRA

def _kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)

def calcular_acuerdo() -> dict | None:
    if not (RUTA_MUESTRA.exists() and RUTA_JUEZ_OCULTO.exists()):
        return None
    h = pd.read_csv(RUTA_MUESTRA).merge(pd.read_csv(RUTA_JUEZ_OCULTO), on="item")
    h["etiqueta_humana"] = h["etiqueta_humana"].astype(str).str.strip().str.lower()
    h = h[h.etiqueta_humana.isin(["respaldada", "no_respaldada", "contradicha"])]
    if len(h) == 0:
        return {"estado": "pendiente", "n": 0}
    hum_b = ["respaldada" if x == "respaldada" else "no" for x in h.etiqueta_humana]
    juez_b = ["respaldada" if x == "respaldada" else "no" for x in h.etiqueta_juez]
    tp = sum(x == "no" and y == "no" for x, y in zip(hum_b, juez_b))
    fp = sum(x == "respaldada" and y == "no" for x, y in zip(hum_b, juez_b))
    fn = sum(x == "no" and y == "respaldada" for x, y in zip(hum_b, juez_b))
    conf = pd.crosstab(h.etiqueta_humana, h.etiqueta_juez, rownames=["humano"], colnames=["juez"])
    fid_h = np.mean([x == "respaldada" for x in hum_b])
    fid_j = np.mean([x == "respaldada" for x in juez_b])
    res = {
        "estado": "completo",
        "n": int(len(h)),
        "acuerdo_3_clases": float((h.etiqueta_humana == h.etiqueta_juez).mean()),
        "acuerdo_binario": float(np.mean([x == y for x, y in zip(hum_b, juez_b)])),
        "kappa_cohen_binario": _kappa(hum_b, juez_b),
        "juez_precision_no_respaldada": tp / (tp + fp) if tp + fp else float("nan"),
        "juez_recall_no_respaldada": tp / (tp + fn) if tp + fn else float("nan"),
        "fidelidad_humano_en_muestra": float(fid_h),
        "fidelidad_juez_en_muestra": float(fid_j),
        "matriz_confusion": conf.to_dict(),
        "desacuerdos": h[[a != b for a, b in zip(hum_b, juez_b)]][
            ["item", "afirmacion", "etiqueta_humana", "etiqueta_juez", "justificacion_juez", "nota_humana"]
        ].to_dict(orient="records"),
    }
    (DIR_RESULTADOS / "acuerdo_juez.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return res