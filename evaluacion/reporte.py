"""Genera resultados/reporte.md (y gráficas PNG) a partir de las métricas."""

from __future__ import annotations
import json
import math
from datetime import datetime
import numpy as np
import pandas as pd
from laboral.config import DIR_RESULTADOS, MODELO_JUEZ, VERSIONES
from .errores import TIPOS
from .estadistica import efecto_minimo_detectable

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
UMBRALES = {
    "invencion_fuera_de_alcance": ("≤", 0.10, "proporción de preguntas fuera de alcance que el asistente respondió"),
    "fidelidad_micro": ("≥", 0.95, "afirmaciones factuales respaldadas por los fragmentos"),
    "contradice": ("≤", 0.05, "respuestas que contradicen la referencia"),
    "exito": ("≥", 0.85, "preguntas bien resueltas (criterio compuesto)"),
    "acierto_k": ("≥", 0.90, "preguntas donde el artículo correcto llegó al modelo"),
}

def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"

def _num(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"

def resumen_version(df: pd.DataFrame, det: pd.DataFrame, v: str) -> dict:
    d = df[df.version == v]
    dd = det[(det.version == v) & (det.tipo_afirmacion == "factual")] if len(det) else det
    resp = d[d.tipo != "no_respondible"]
    return {
        "n": len(d),
        "acierto_k": resp.acierto_k.mean(), "recall_articulo": resp.recall_articulo.mean(),
        "recall_fraccion": resp.recall_fraccion.mean(), "mrr": resp.mrr.mean(),
        "abstencion_correcta": d.abstencion_correcta.mean(),
        "invencion_fuera_de_alcance": d.invento_fuera_de_alcance.mean(),
        "abstencion_indebida": d.abstencion_indebida.mean(),
        "fidelidad_macro": d.fidelidad.mean(),
        "fidelidad_micro": (dd.etiqueta_juez == "respaldada").mean() if len(dd) else math.nan,
        "fidelidad_estricta": d.fidelidad_estricta.mean(),
        "n_afirmaciones": int(len(dd)),
        "cobertura": resp.cobertura.mean(), "contradice": resp.contradice.mean(),
        "relevancia": d.relevancia.mean(), "precision_citas": d.precision_citas.mean(),
        "citas_invalidas": (d.citas_invalidas > 0).mean(), "advertencia_cifras": d.advertencia_cifras.mean(),
        "dato_inventado": d.dato_inventado.mean(), "exito": d.exito.mean(),
        "lat_p50": d.segundos_total.median(), "lat_p95": d.segundos_total.quantile(0.95),
        "costo_medio": d.costo_usd.mean(), "costo_total": d.costo_usd.sum(),
        "costo_juez_total": d.costo_juez_usd.sum(), "tokens_entrada_medio": d.tokens_entrada.mean(),
        "tokens_salida_medio": d.tokens_salida.mean(),
    }

def _grafica_metricas(res: dict, versiones: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    claves = [("acierto_k", "Artículo correcto\nrecuperado"), ("abstencion_correcta", "Se abstiene si\nno hay respuesta"),
              ("fidelidad_micro", "Fidelidad"), ("cobertura", "Cobertura de\npuntos clave"), ("exito", "Éxito")]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ancho = 0.8 / len(versiones)
    x = np.arange(len(claves))
    for i, v in enumerate(versiones):
        vals = [res[v][k] if not math.isnan(res[v][k]) else 0 for k, _ in claves]
        barras = ax.bar(x + i * ancho - 0.4 + ancho / 2, vals, ancho - 0.02, color=COLORES[i % 4], label=v, zorder=2)
        for b, val in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{100 * val:.0f}", ha="center", va="bottom",
                    fontsize=7, color=TINTA_2)
    ax.set_xticks(x, [l for _, l in claves], fontsize=8, color=TINTA)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("proporción", color=TINTA_2, fontsize=8)
    ax.grid(axis="y", color=REJILLA, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", colors=TINTA_2, labelsize=7)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=len(versiones))
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "metricas_por_version.png")
    plt.close(fig)

def _grafica_errores(df: pd.DataFrame, versiones: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = df[df.version.isin(versiones)].groupby(["error_principal", "version"]).size().unstack(fill_value=0)
    t = t.drop(index="sin_error", errors="ignore")
    if t.empty:
        return
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.2), dpi=150)
    y = np.arange(len(t))
    alto = 0.8 / len(versiones)
    for i, v in enumerate(versiones):
        if v not in t:
            continue
        ax.barh(y + i * alto - 0.4 + alto / 2, t[v], alto - 0.04, color=COLORES[i % 4], label=v, zorder=2)
    ax.set_yticks(y, [s.replace("_", " ") for s in t.index], fontsize=8, color=TINTA)
    ax.set_xlabel("preguntas", fontsize=8, color=TINTA_2)
    ax.grid(axis="x", color=REJILLA, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "errores_por_tipo.png")
    plt.close(fig)

def recomendacion(r: dict) -> tuple[str, list[str]]:
    lineas, fallas = [], 0
    for k, (op, umbral, desc) in UMBRALES.items():
        val = r.get(k, math.nan)
        ok = (not math.isnan(val)) and (val <= umbral if op == "≤" else val >= umbral)
        fallas += not ok
        lineas.append(f"| {desc} | {_pct(val)} | {op} {_pct(umbral)} | {'✅' if ok else '❌'} |")
    if fallas == 0:
        veredicto = ("**Sí, como piloto controlado**: cumple todos los umbrales. Condiciones: aviso visible de "
                     "que no sustituye asesoría legal, mostrar siempre los artículos citados, canal para escalar "
                     "a RH/abogado en despidos y casos individuales, y monitoreo con muestras calificadas por humanos.")
    elif fallas <= 2:
        veredicto = ("**Todavía no frente a empleados sin supervisión.** Recomendado como herramienta interna "
                     "de RH (un humano revisa cada respuesta antes de enviarla) hasta corregir los umbrales en rojo.")
    else:
        veredicto = "**No.** Falla en varios umbrales críticos; usar solo para seguir iterando."
    return veredicto, lineas

def escribir_reporte(df, det, comp, comp_aa, acuerdo, val, manifiesto, versiones, segundos) -> str:
    res = {v: resumen_version(df, det, v) for v in df.version.unique()}
    _grafica_metricas(res, versiones)
    _grafica_errores(df, versiones)
    principal = versiones[0]
    md = [f"# Reporte de evaluación — asistente de derechos laborales",
          f"_Generado {datetime.now():%Y-%m-%d %H:%M} · {segundos / 60:.1f} min · juez: `{MODELO_JUEZ}`_", ""]
    md += ["## Documentos evaluados", "| Documento | Última reforma DOF | sha256 |", "|---|---|---|"]
    for k, m in (manifiesto or {}).items():
        md.append(f"| {k} | {m.get('ultima_reforma_dof', '—')} | `{m.get('sha256', '')[:16]}…` |")
    if val and val.get("n_problemas"):
        md += ["", f"⚠ **Validación del conjunto**: {val['n_problemas']} posibles desajustes con el texto vigente "
                   "(ver `validacion_dataset.json`). Revisarlos antes de interpretar los resultados de esas preguntas."]
    md += ["", "## Resumen por versión", "", "![métricas](metricas_por_version.png)", ""]
    vs = list(res)
    md.append("| Métrica | " + " | ".join(f"`{v}`" for v in vs) + " |")
    md.append("|---" * (len(vs) + 1) + "|")
    filas = [
        ("**Recuperación** — artículo correcto entre los k fragmentos (acierto@k)", "acierto_k", _pct),
        ("Recall de artículos esperados", "recall_articulo", _pct), ("Recall a nivel fracción", "recall_fraccion", _pct),
        ("MRR (posición del primer artículo correcto)", "mrr", _num),
        ("**Abstención** — se abstiene en preguntas sin respuesta", "abstencion_correcta", _pct),
        ("Inventa en preguntas sin respuesta", "invencion_fuera_de_alcance", _pct),
        ("Se abstiene indebidamente en preguntas con respuesta", "abstencion_indebida", _pct),
        ("**Fidelidad** (micro: afirmaciones respaldadas / total)", "fidelidad_micro", _pct),
        ("Fidelidad (macro: promedio por respuesta)", "fidelidad_macro", _pct),
        ("Fidelidad estricta (respaldo en fragmentos citados)", "fidelidad_estricta", _pct),
        ("Afirmaciones factuales evaluadas", "n_afirmaciones", lambda x: str(x)),
        ("**Corrección** — cobertura de puntos clave", "cobertura", _pct),
        ("Contradice la referencia", "contradice", _pct), ("Relevancia (0–1)", "relevancia", _num),
        ("Precisión de citas (artículo citado ∈ esperados)", "precision_citas", _pct),
        ("Respuestas con alguna cita inválida (descartada)", "citas_invalidas", _pct),
        ("Juez detecta dato externo inventado", "dato_inventado", _pct),
        ("**Éxito** (criterio compuesto)", "exito", _pct),
        ("Latencia p50 (s)", "lat_p50", lambda x: _num(x, 1)), ("Latencia p95 (s)", "lat_p95", lambda x: _num(x, 1)),
        ("Costo medio por pregunta (USD)", "costo_medio", lambda x: f"${x:.4f}"),
        ("Tokens de entrada / salida por pregunta", "tokens_entrada_medio", None),
        ("Costo total asistente (USD)", "costo_total", lambda x: f"${x:.3f}"),
        ("Costo total juez (USD)", "costo_juez_total", lambda x: f"${x:.3f}"),
    ]
    for nombre, k, fmt in filas:
        if fmt is None:
            celdas = [f"{res[v]['tokens_entrada_medio']:.0f} / {res[v]['tokens_salida_medio']:.0f}" for v in vs]
        else:
            celdas = [fmt(res[v][k]) for v in vs]
        md.append(f"| {nombre} | " + " | ".join(celdas) + " |")
    md += ["", "### Por estilo de pregunta (éxito / acierto@k)", "", "| Estilo | " + " | ".join(vs) + " |",
           "|---" * (len(vs) + 1) + "|"]
    for est, g in df.groupby("estilo"):
        md.append(f"| {est} (n={g.id.nunique()}) | " + " | ".join(
            f"{_pct(g[g.version == v].exito.mean())} / {_pct(g[(g.version == v) & (g.tipo != 'no_respondible')].acierto_k.mean())}"
            for v in vs) + " |")
    md += ["", "### Por categoría (éxito)", "", "| Categoría | " + " | ".join(vs) + " |", "|---" * (len(vs) + 1) + "|"]
    for cat, g in df.groupby("categoria"):
        md.append(f"| {cat} (n={g.id.nunique()}) | " + " | ".join(_pct(g[g.version == v].exito.mean()) for v in vs) + " |")
    def tabla_comp(c, a, b):
        out = [f"| Métrica | `{a}` | `{b}` | Δ (b−a) | IC 95% | p | Prueba | Veredicto |", "|---|---|---|---|---|---|---|---|"]
        for f in c:
            pct = f["metrica"] not in ("segundos_total", "costo_usd", "mrr")
            fm = (lambda x: _pct(x)) if pct else (lambda x: _num(x, 4))
            out.append(f"| {f['metrica']} | {fm(f[a])} | {fm(f[b])} | {fm(f['diferencia'])} | "
                       f"[{fm(f['ic95_inf'])}, {fm(f['ic95_sup'])}] | {f['p']:.3f} | {f['prueba']} | {f['veredicto']} |")
        return out
    if comp:
        a, b = versiones[0], versiones[1]
        n = df[df.version == a].id.nunique()
        md += ["", f"## Comparación de versiones: `{a}` vs `{b}`", "",
               f"- `{a}`: {json.dumps({k: v for k, v in VERSIONES[a].a_dict().items() if VERSIONES[a].a_dict()[k] != VERSIONES[b].a_dict()[k]}, ensure_ascii=False)}",
               f"- `{b}`: {json.dumps({k: v for k, v in VERSIONES[b].a_dict().items() if VERSIONES[a].a_dict()[k] != VERSIONES[b].a_dict()[k]}, ensure_ascii=False)}",
               "", *tabla_comp(comp, a, b), "",
               f"Con n = {n} preguntas, la diferencia mínima en la tasa de éxito que McNemar detectaría con 80% de "
               f"potencia (suponiendo ~20% de pares discordantes) es de aproximadamente "
               f"{_pct(efecto_minimo_detectable(n))}. Diferencias menores no deben interpretarse como mejoras."]
        reales = [f["metrica"] for f in comp if f["veredicto"] == "diferencia real"]
        md.append("" if not reales else f"\n**Diferencias estadísticamente reales:** {', '.join(reales)}.")
        if not reales:
            md.append("\n**Ninguna diferencia es distinguible del ruido** con este tamaño de muestra.")
    if comp_aa:
        md += ["", f"### Prueba A/A: `{versiones[0]}` contra sí misma (otra corrida, sin caché)",
               "Mide cuánto cambian las métricas solo por la no-determinancia del modelo. Si A/A muestra "
               "diferencias del mismo orden que A/B, la diferencia A/B es ruido.", "",
               *tabla_comp(comp_aa, versiones[0], f"{versiones[0]}_rep")]
    md += ["", "## ¿Dónde y por qué falla?", "", "![errores](errores_por_tipo.png)", "",
           "| Tipo de error (principal) | " + " | ".join(versiones) + " | Qué significa | Causa probable |",
           "|---" * (len(versiones) + 3) + "|"]
    conteo = df.groupby(["error_principal", "version"]).size().unstack(fill_value=0)
    for t, (desc, causa) in TIPOS.items():
        if t in conteo.index:
            md.append(f"| {t.replace('_', ' ')} | " + " | ".join(str(conteo.loc[t].get(v, 0)) for v in versiones)
                      + f" | {desc} | {causa} |")
    md.append(f"| sin error | " + " | ".join(str(conteo.loc['sin_error'].get(v, 0)) if 'sin_error' in conteo.index else '0'
                                          for v in versiones) + " | | |")
    md += ["", f"### Ejemplos (`{principal}`)", ""]
    ej = df[(df.version == principal) & (df.error_principal != "sin_error")]
    for t, g in ej.groupby("error_principal"):
        md.append(f"**{t.replace('_', ' ')}**")
        for _, f in g.head(3).iterrows():
            resp = str(f.respuesta).replace("\n", " ")
            md.append(f"- `{f.id}` _{f.pregunta}_ → estado `{f.estado}`, recall art. {_pct(f.recall_articulo)}, "
                      f"fidelidad {_pct(f.fidelidad)}, cobertura {_pct(f.cobertura)}. "
                      f"Respuesta: “{resp[:220]}{'…' if len(resp) > 220 else ''}”")
        md.append("")
    md += ["## ¿El juez coincide con una persona?", ""]
    if acuerdo and acuerdo.get("estado") == "completo":
        md += [f"Muestra de **{acuerdo['n']}** afirmaciones calificadas a mano (estratificada, sobre-representa las "
               "que el juez marcó como no respaldadas).", "",
               f"- Acuerdo binario (respaldada / no): **{_pct(acuerdo['acuerdo_binario'])}**",
               f"- Kappa de Cohen: **{_num(acuerdo['kappa_cohen_binario'])}** (≥0.6 sustancial, ≥0.8 casi perfecto)",
               f"- Acuerdo en 3 clases: {_pct(acuerdo['acuerdo_3_clases'])}",
               f"- Detección de no respaldadas por el juez: precisión {_pct(acuerdo['juez_precision_no_respaldada'])}, "
               f"recall {_pct(acuerdo['juez_recall_no_respaldada'])}",
               f"- Fidelidad en la muestra: humano {_pct(acuerdo['fidelidad_humano_en_muestra'])} vs juez "
               f"{_pct(acuerdo['fidelidad_juez_en_muestra'])}", "",
               f"Desacuerdos: {len(acuerdo['desacuerdos'])} (detalle en `acuerdo_juez.json`)."]
    else:
        md += ["**Pendiente.** Califica `resultados/anotacion_manual.csv` (o usa `notebooks/03_anotacion_manual.ipynb`) "
               "y vuelve a correr `python -m evaluacion.run` (todo viene de caché, es instantáneo) o "
               "`python -m evaluacion.acuerdo`. Hasta entonces, las cifras de fidelidad dependen de un juez no validado."]
    rp = res[principal]
    veredicto, lineas = recomendacion(rp)
    md += ["", f"## Recomendación (`{principal}`)", "", "| Criterio | Valor | Umbral | |", "|---|---|---|---|", *lineas, "",
           veredicto]
    ruta = DIR_RESULTADOS / "reporte.md"
    ruta.write_text("\n".join(md), encoding="utf-8")
    (DIR_RESULTADOS / "resumen.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False))
    return str(ruta)