#Taxonomía de errores (reto 6): cada pregunta recibe todas las etiquetas que aplican y una principal.

from __future__ import annotations
import math

TIPOS = {
    "invencion_fuera_de_alcance": (
        "Respondió una pregunta que no está en los documentos.",
        "El modelo completa con conocimiento propio (montos, otras leyes) cuando los fragmentos recuperados "
        "son temáticamente cercanos; la instrucción de abstenerse compite con su tendencia a ser útil."),
    "fallo_recuperacion": (
        "Ningún artículo esperado llegó al modelo.",
        "Brecha de vocabulario entre la pregunta coloquial y la ley, artículos largos partidos en fragmentos "
        "sin el tema explícito, o respuestas que requieren combinar artículos lejanos (remisiones como 'en los "
        "términos del artículo 50')."),
    "abstencion_indebida": (
        "Se abstuvo aunque el artículo correcto sí estaba entre los fragmentos.",
        "Exceso de cautela: el fragmento no usa las palabras de la pregunta y el modelo no hace la conexión, "
        "o la guarda de citas descartó una cita mal copiada y forzó el 'sin respuesta'."),
    "fiel_a_fragmento_equivocado": (
        "Todo lo que dice está en los fragmentos, pero contradice la referencia.",
        "Se apoyó en un artículo vecino que no aplica (otra fracción, otro supuesto, un régimen especial "
        "o el apartado B del 123), o leyó una excepción como regla."),
    "contradice_referencia": (
        "Afirma algo incompatible con la respuesta correcta.",
        "Cálculos mal hechos a partir de una regla correcta (p. ej. antigüedad), o mezcla de reglas."),
    "afirmacion_no_respaldada": (
        "Incluye afirmaciones que los fragmentos no sostienen.",
        "Agrega matices 'de sentido común' o conocimiento previo del modelo que no están en el texto."),
    "incompleta": (
        "Cubre menos de la mitad de los puntos clave.",
        "Faltó un artículo complementario en los k fragmentos o el modelo resumió de más."),
    "poco_relevante": (
        "La respuesta no atiende directamente la pregunta.",
        "Respuesta genérica sobre el tema recuperado en lugar del caso concreto preguntado."),
    "cita_invalida": (
        "Alguna cita textual no aparece literalmente en la ley (fue descartada).",
        "El modelo parafrasea dentro de la cita o une frases de fracciones distintas."),
}

PRIORIDAD = list(TIPOS)

def _es(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and bool(v)

def etiquetar(f: dict) -> list[str]:
    e = []
    if f["tipo"] == "no_respondible":
        if f["estado"] != "sin_respuesta":
            e.append("invencion_fuera_de_alcance")
    else:
        if f.get("recall_articulo") == 0:
            e.append("fallo_recuperacion")
        if f["estado"] == "sin_respuesta" and f.get("recall_articulo", 0) > 0:
            e.append("abstencion_indebida")
        if f["estado"] != "sin_respuesta":
            if _es(f.get("contradice")):
                e.append("fiel_a_fragmento_equivocado" if f.get("fidelidad") == 1.0 else "contradice_referencia")
            if isinstance(f.get("cobertura"), float) and not math.isnan(f["cobertura"]) and f["cobertura"] < 0.5:
                e.append("incompleta")
            if isinstance(f.get("relevancia"), float) and not math.isnan(f["relevancia"]) and f["relevancia"] <= 0.25:
                e.append("poco_relevante")
    fid = f.get("fidelidad")
    if isinstance(fid, float) and not math.isnan(fid) and fid < 1.0:
        e.append("afirmacion_no_respaldada")
    if f.get("citas_invalidas", 0) > 0:
        e.append("cita_invalida")
    return e

def principal(etiquetas: list[str]) -> str:
    for t in PRIORIDAD:
        if t in etiquetas and t != "cita_invalida":
            return t
    return "cita_invalida" if "cita_invalida" in etiquetas else "sin_error"