#Evaluación completa con un solo comando:

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from laboral.asistente import Asistente
from laboral.config import DIR_CACHE, DIR_DATA, DIR_RESULTADOS, MODELO_JUEZ, VERSIONES
from laboral.descarga import descargar
from laboral.llm import ClienteLLM, Uso
from laboral.parseo import construir_corpus
from . import acuerdo_juez, errores, metricas
from .estadistica import comparar
from .juez import Juez
from .reporte import escribir_reporte
from .validar import validar

FECHA_REFERENCIA = "2026-10-02"
METRICAS_CONTINUAS = ["recall_articulo", "mrr", "fidelidad", "fidelidad_estricta", "cobertura", "relevancia", "precision_citas", "segundos_total", "costo_usd"]
METRICAS_BINARIAS = ["exito", "acierto_k", "abstencion_correcta", "abstencion_indebida", "contradice"]

def correr_version(nombre: str, preguntas: list[dict], llm: ClienteLLM, embedder, salida: Path) -> list[dict]:
    config = VERSIONES[nombre.replace("_rep", "")]
    asistente = Asistente.cargar(config, llm=llm, embedder=embedder, fecha=FECHA_REFERENCIA)
    filas = []
    with open(salida, "w", encoding="utf-8") as f:
        for q in tqdm(preguntas, desc=f"asistente {nombre}"):
            try:
                r = asistente.preguntar(q["pregunta"]).a_dict()
            except Exception as e:
                r = {"pregunta": q["pregunta"], "estado": "error", "respuesta": f"ERROR: {e}", "fundamentos": [],
                     "fragmentos": [], "consultas": [], "costo_usd": 0, "segundos_total": 0, "advertencias": []}
            r.update(id=q["id"], version=nombre)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            filas.append(r)
    return filas

def juzgar(r: dict, q: dict, juez: Juez) -> tuple[dict, list[dict]]:
    uso = Uso()
    fila = {"id": q["id"], "version": r["version"], "categoria": q["categoria"], "tipo": q["tipo"],
            "estilo": q["estilo"], "pregunta": q["pregunta"], "estado": r["estado"], "respuesta": r["respuesta"]}
    fila.update(metricas.metricas_recuperacion(r["fragmentos"], q["articulos_esperados"]))
    fila.update(metricas.abstencion(r["estado"], q["tipo"]))
    fila["precision_citas"] = metricas.precision_citas(r["fundamentos"], r["fragmentos"], q["articulos_esperados"])
    fila["citas_invalidas"] = sum(not f["valida"] for f in r["fundamentos"])
    fila["n_citas_validas"] = sum(f["valida"] for f in r["fundamentos"])
    fila["advertencia_cifras"] = float(any("Cifras" in a for a in r.get("advertencias", [])))
    for k in ("segundos_total", "costo_usd", "tokens_entrada", "tokens_salida", "llamadas"):
        fila[k] = r.get(k, math.nan)
    detalle = []
    afirm, ver = [], []
    if r["estado"] not in ("sin_respuesta", "error"):
        afirm = juez.extraer(q["pregunta"], r["respuesta"], uso)
        textos = [a["texto"] for a in afirm]
        ver = juez.verificar(textos, r["fragmentos"], uso)
        fila.update(metricas.fidelidad(afirm, ver, r["fragmentos"], r["fundamentos"]))
        contexto = "\n\n".join(f"[F{i}] {f['etiqueta']}\n{f['texto']}" for i, f in enumerate(r["fragmentos"], 1))
        for a, v in zip(afirm, ver):
            detalle.append({"id": q["id"], "version": r["version"], "pregunta": q["pregunta"], "afirmacion": a["texto"],
                            "tipo_afirmacion": a["tipo"], "etiqueta_juez": v["etiqueta"],
                            "fragmentos_juez": ",".join(v.get("fragmentos", [])),
                            "justificacion_juez": v.get("justificacion", ""), "fragmentos_contexto": contexto})
    else:
        fila.update(metricas.fidelidad([], [], [], []))
    if q["tipo"] != "no_respondible" and r["estado"] not in ("sin_respuesta", "error"):
        calif = juez.corregir(q["pregunta"], r["respuesta"], q["respuesta_referencia"], q["puntos_clave"], uso)
        fila.update(metricas.correccion(calif, len(q["puntos_clave"])))
        fila["comentario_juez"] = calif.get("comentario", "")
    else:
        fila.update(metricas.correccion(None, 0))
        if q["tipo"] != "no_respondible":
            fila.update(cobertura=0.0, contradice=0.0)
    fila["exito"] = metricas.exito(fila)
    fila["costo_juez_usd"] = uso.costo_usd
    fila["etiquetas_error"] = errores.etiquetar(fila)
    fila["error_principal"] = errores.principal(fila["etiquetas_error"])
    return fila, detalle

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versiones", nargs="+", default=["v1_articulo", "v2_fraccion"], choices=list(VERSIONES))
    ap.add_argument("--preguntas", default=str(DIR_DATA / "preguntas.json"))
    ap.add_argument("--limite", type=int, default=None, help="evaluar solo las primeras N preguntas")
    ap.add_argument("--aa", action="store_true", help="repetir la primera versión sin caché (prueba A/A de ruido)")
    ap.add_argument("--n-anotacion", type=int, default=40)
    ap.add_argument("--sin-cache", action="store_true", help="ignorar la caché de llamadas al modelo")
    args = ap.parse_args(argv)
    t0 = time.time()
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    preguntas = json.loads(Path(args.preguntas).read_text(encoding="utf-8"))[: args.limite]
    
    print("1) Documentos")
    manifiesto = descargar()
    
    print("2) Validación del conjunto de preguntas contra el texto vigente")
    from laboral.recuperacion import EmbedderST
    val = validar(preguntas, construir_corpus("fraccion", 2200))
    (DIR_RESULTADOS / "validacion_dataset.json").write_text(json.dumps(val, indent=2, ensure_ascii=False))
    if val["n_problemas"]:
        print(f"   ⚠ {val['n_problemas']} posibles desajustes entre preguntas y ley vigente "
              f"(ver resultados/validacion_dataset.json)")
    
    print("3) Asistente")
    llm = ClienteLLM(usar_cache=not args.sin_cache)
    embedder = EmbedderST()
    respuestas = []
    nombres = list(args.versiones)
    for v in nombres:
        respuestas += correr_version(v, preguntas, llm, embedder, DIR_RESULTADOS / f"respuestas_{v}.jsonl")
    if args.aa:
        rep = f"{nombres[0]}_rep"
        llm_rep = ClienteLLM(usar_cache=True, cache_dir=DIR_CACHE / "llm_replica")
        respuestas += correr_version(rep, preguntas, llm_rep, embedder, DIR_RESULTADOS / f"respuestas_{rep}.jsonl")
    
    print(f"4) Juez ({MODELO_JUEZ})")
    juez = Juez(llm)
    por_id = {q["id"]: q for q in preguntas}
    filas, detalle = [], []
    for r in tqdm(respuestas, desc="juez"):
        f, d = juzgar(r, por_id[r["id"]], juez)
        filas.append(f)
        detalle += d
    df = pd.DataFrame(filas)
    det = pd.DataFrame(detalle)
    df.to_csv(DIR_RESULTADOS / "metricas_por_pregunta.csv", index=False)
    det.to_csv(DIR_RESULTADOS / "afirmaciones_juez.csv", index=False)
    
    print("5) Comparación estadística")
    comp = comparar(df, nombres[0], nombres[1], METRICAS_CONTINUAS, METRICAS_BINARIAS) if len(nombres) > 1 else []
    comp_aa = comparar(df, nombres[0], f"{nombres[0]}_rep", METRICAS_CONTINUAS, METRICAS_BINARIAS) if args.aa else []
    pd.DataFrame(comp).to_csv(DIR_RESULTADOS / "comparacion.csv", index=False)
    
    print("6) Validación del juez")
    if len(det):
        acuerdo_juez.generar_muestra(det[det.version == nombres[0]], n=args.n_anotacion)
    acuerdo = acuerdo_juez.calcular_acuerdo()
    
    print("7) Reporte")
    ruta = escribir_reporte(df, det, comp, comp_aa, acuerdo, val, manifiesto, nombres, time.time() - t0)
    
    print(f"\nListo en {time.time() - t0:.0f} s -> {ruta}")
    return df

if __name__ == "__main__":
    main(sys.argv[1:])