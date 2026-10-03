#LLM como juez

from __future__ import annotations
from laboral.config import MODELO_JUEZ
from laboral.llm import ClienteLLM, Uso

SIS_EXTRAER = """Descompones respuestas en afirmaciones atómicas para auditarlas.
Una afirmación atómica es una sola idea verificable (un derecho, una cifra, un plazo, una condición, una \
prohibición). Si una oración contiene dos datos ("son 12 días y aumentan 2 por año"), sepáralos.
Clasifica cada afirmación:
- "factual": dice algo sobre lo que establece la ley o sobre los derechos/obligaciones del empleado o patrón.
- "meta": no afirma contenido legal (recomendar consultar a RH, decir que algo no está en los documentos, \
saludos, aclarar que depende del caso).
Copia el sentido de la respuesta; no agregues nada ni corrijas errores."""

ESQ_EXTRAER = {
    "type": "object",
    "properties": {"afirmaciones": {"type": "array", "items": {
        "type": "object",
        "properties": {"texto": {"type": "string"}, "tipo": {"type": "string", "enum": ["factual", "meta"]}},
        "required": ["texto", "tipo"]}}},
    "required": ["afirmaciones"],
}

SIS_VERIFICAR = """Eres un auditor jurídico estricto. Para cada afirmación decide si está respaldada por los \
fragmentos de ley proporcionados, usando ÚNICAMENTE esos fragmentos (no tu conocimiento).
Etiquetas:
- "respaldada": el contenido de la afirmación se sigue directamente de uno o más fragmentos. Se permiten \
paráfrasis en lenguaje sencillo y aritmética simple a partir de reglas del fragmento (p. ej. 12 + 2 + 2 + 2 = 18).
- "no_respaldada": ningún fragmento la sostiene (aunque pudiera ser cierta en la realidad), o agrega \
condiciones, cifras o matices que el texto no dice.
- "contradicha": algún fragmento dice lo contrario.
Indica los ids de los fragmentos que la respaldan (vacío si no está respaldada) y una justificación breve."""

ESQ_VERIFICAR = {
    "type": "object",
    "properties": {"veredictos": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "indice": {"type": "integer"},
            "etiqueta": {"type": "string", "enum": ["respaldada", "no_respaldada", "contradicha"]},
            "fragmentos": {"type": "array", "items": {"type": "string"}},
            "justificacion": {"type": "string"},
        },
        "required": ["indice", "etiqueta", "fragmentos", "justificacion"]}}},
    "required": ["veredictos"],
}

SIS_CORRECCION = """Evalúas la respuesta de un asistente de derechos laborales contra una respuesta de \
referencia escrita por una persona experta.
1) Para cada punto clave de la referencia indica:
   - "cubierto": la respuesta lo dice (con otras palabras está bien);
   - "omitido": la respuesta no lo menciona;
   - "contradicho": la respuesta dice algo incompatible (otra cifra, otro plazo, lo contrario).
2) "relevancia" de 1 a 5: ¿la respuesta atiende directamente lo que el empleado preguntó? \
(5 = responde exactamente la pregunta; 3 = responde algo relacionado pero se desvía o es genérica; \
1 = no responde la pregunta).
3) "dato_inventado": true si la respuesta afirma algún dato concreto (cifra, plazo, derecho) que no está en la \
referencia y que parece provenir de fuera de la LFT/art. 123 (otras leyes, montos vigentes, políticas)."""

ESQ_CORRECCION = {
    "type": "object",
    "properties": {
        "puntos": {"type": "array", "items": {
            "type": "object",
            "properties": {"indice": {"type": "integer"},
                           "veredicto": {"type": "string", "enum": ["cubierto", "omitido", "contradicho"]},
                           "justificacion": {"type": "string"}},
            "required": ["indice", "veredicto", "justificacion"]}},
        "relevancia": {"type": "integer", "minimum": 1, "maximum": 5},
        "dato_inventado": {"type": "boolean"},
        "comentario": {"type": "string"},
    },
    "required": ["puntos", "relevancia", "dato_inventado", "comentario"],
}

class Juez:
    def __init__(self, llm: ClienteLLM, modelo: str = MODELO_JUEZ):
        self.llm, self.modelo = llm, modelo
    def extraer(self, pregunta: str, respuesta: str, uso: Uso) -> list[dict]:
        out = self.llm.json(modelo=self.modelo, sistema=SIS_EXTRAER,
                            usuario=f"Pregunta: {pregunta}\n\nRespuesta a descomponer:\n{respuesta}",
                            esquema=ESQ_EXTRAER, nombre="afirmaciones", descripcion="Afirmaciones atómicas",
                            max_tokens=1500, uso=uso, etapa="juez_extraer")
        return [a for a in out.get("afirmaciones", []) if a.get("texto")]
    def verificar(self, afirmaciones: list[str], fragmentos: list[dict], uso: Uso) -> list[dict]:
        if not afirmaciones:
            return []
        frs = "\n\n".join(f"<fragmento id=\"F{i}\" fuente=\"{f['etiqueta']}\">\n{f['texto']}\n</fragmento>"
                          for i, f in enumerate(fragmentos, 1))
        lista = "\n".join(f"{i}. {a}" for i, a in enumerate(afirmaciones))
        out = self.llm.json(modelo=self.modelo, sistema=SIS_VERIFICAR,
                            usuario=f"<fragmentos>\n{frs}\n</fragmentos>\n\nAfirmaciones (índice. texto):\n{lista}",
                            esquema=ESQ_VERIFICAR, nombre="veredictos", descripcion="Veredicto por afirmación",
                            max_tokens=3000, uso=uso, etapa="juez_verificar")
        por_indice = {v.get("indice"): v for v in out.get("veredictos", [])}
        return [por_indice.get(i, {"indice": i, "etiqueta": "no_respaldada", "fragmentos": [], "justificacion": "sin veredicto del juez"}) for i in range(len(afirmaciones))]
    def corregir(self, pregunta: str, respuesta: str, referencia: str, puntos: list[str], uso: Uso) -> dict:
        lista = "\n".join(f"{i}. {p}" for i, p in enumerate(puntos))
        return self.llm.json(modelo=self.modelo, sistema=SIS_CORRECCION,
                             usuario=(f"Pregunta del empleado: {pregunta}\n\nRespuesta de referencia: {referencia}\n"
                                      f"Puntos clave (índice. texto):\n{lista}\n\nRespuesta del asistente:\n{respuesta}"),
                             esquema=ESQ_CORRECCION, nombre="calificacion", descripcion="Calificación de la respuesta",
                             max_tokens=1500, uso=uso, etapa="juez_correccion")