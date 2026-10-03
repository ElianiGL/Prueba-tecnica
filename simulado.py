import re
import shutil
from pathlib import Path
from laboral.llm import Uso

AQUI = Path(__file__).parent

class LLMSimulado:
    def json(self, *, modelo, sistema, usuario, esquema, nombre, descripcion, temperatura=0.0, max_tokens=1500,
             uso: Uso | None = None, etapa=""):
        if uso is not None:
            uso.sumar(etapa, modelo, 1000, 200, 0.5, cache=False)
        if nombre == "consultas_juridicas":
            return {"consultas": ["rescisión de la relación de trabajo", "vacaciones días laborables"]}
        if nombre == "responder":
            pregunta = usuario.split("Pregunta del empleado:")[-1]
            if re.search(r"UMA|salario mínimo|ISR|Afore|Infonavit|luto|Estados Unidos|IMSS|home office", pregunta):
                return {"estado": "sin_respuesta", "respuesta": "No está en los documentos.", "fundamentos": [],
                        "no_cubierto": "todo"}
            m = re.search(r'<fragmento id="F1"[^>]*>\n(.*?)\n</fragmento>', usuario, re.S)
            palabras = m.group(1).split()[:10] if m else []
            cita = " ".join(palabras)
            if "embarazada" in pregunta:
                cita = "queda prohibido despedir a la trabajadora embarazada en cualquier caso"
            return {"estado": "respondida", "respuesta": f"Tienes derecho a 12 días. Según la ley, {cita[:60]}. Son 99 días.",
                    "fundamentos": [{"fragmento": "F1", "cita_textual": cita}], "no_cubierto": ""}
        if nombre == "afirmaciones":
            resp = usuario.split("Respuesta a descomponer:\n")[-1]
            return {"afirmaciones": [{"texto": s, "tipo": "factual"} for s in re.split(r"(?<=\.)\s+", resp) if s]}
        if nombre == "veredictos":
            n = len(re.findall(r"(?m)^\d+\. ", usuario.split("Afirmaciones")[-1]))
            return {"veredictos": [{"indice": i, "etiqueta": "respaldada" if i < n - 1 else "no_respaldada",
                                    "fragmentos": ["F1"] if i < n - 1 else [], "justificacion": "sim"}
                                   for i in range(n)]}
        if nombre == "calificacion":
            n = len(re.findall(r"(?m)^\d+\. ", usuario.split("Puntos clave")[-1].split("Respuesta del asistente")[0]))
            return {"puntos": [{"indice": i, "veredicto": "cubierto", "justificacion": ""} for i in range(n)],
                    "relevancia": 4, "dato_inventado": False, "comentario": "sim"}
        raise ValueError(nombre)

def preparar_entorno(tmp: Path, monkeypatch):
    import laboral.config as cfg
    import laboral.parseo as parseo
    import laboral.descarga as descarga
    import evaluacion.run as run
    import evaluacion.reporte as reporte
    import evaluacion.acuerdo_juez as acuerdo

    proc, res, raw = tmp / "procesado", tmp / "resultados", tmp / "raw"
    for d in (proc, res, raw):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(AQUI / "fixture_lft.txt", proc / "LFT.txt")
    shutil.copy(AQUI / "fixture_cpeum.txt", proc / "CPEUM.txt")
    monkeypatch.setattr(parseo, "DIR_PROC", proc)
    monkeypatch.setattr(descarga, "MANIFIESTO", raw / "manifiesto.json")
    for mod in (run, reporte):
        monkeypatch.setattr(mod, "DIR_RESULTADOS", res)
    monkeypatch.setattr(acuerdo, "DIR_RESULTADOS", res)
    monkeypatch.setattr(acuerdo, "RUTA_MUESTRA", res / "anotacion_manual.csv")
    monkeypatch.setattr(acuerdo, "RUTA_JUEZ_OCULTO", res / "anotacion_juez_oculto.csv")
    return proc, res