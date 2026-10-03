#El asistente: pregunta -> recuperación -> respuesta fundamentada y verificada.

from __future__ import annotations
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from .config import ConfigAsistente, VERSIONES
from .llm import ClienteLLM, Uso
from .parseo import Fragmento, construir_corpus
from .recuperacion import IndiceHibrido, expandir_con_glosario

AVISO_SIN_RESPUESTA = (
    "No encontré la respuesta a esta pregunta en la Ley Federal del Trabajo ni en el artículo 123 "
    "constitucional, que son las únicas fuentes que consulto. No voy a suponer una respuesta: "
    "Puedes consultar con recursos humanos o a la autoridad correspondiente, para mejor resolución de tu inquietud."
)

SISTEMA_REESCRITURA = """Eres un experto en derecho laboral mexicano. Tu única tarea es convertir la pregunta \
de un empleado (escrita en lenguaje cotidiano) en consultas de búsqueda redactadas con el vocabulario que usa \
la Ley Federal del Trabajo y el artículo 123 constitucional (por ejemplo: "correr" -> "despido" o "rescisión de \
la relación de trabajo"; "horas extra" -> "jornada extraordinaria"; "día festivo" -> "día de descanso \
obligatorio"; "finiquito" -> "terminación de la relación de trabajo, partes proporcionales").
No respondas la pregunta. No inventes números de artículo."""

ESQUEMA_REESCRITURA = {
    "type": "object",
    "properties": {
        "consultas": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4, "description": "2 a 4 consultas en lenguaje jurídico, cada una de 5 a 20 palabras."},
    },
    "required": ["consultas"],
}

SISTEMA_RESPUESTA = """Eres el asistente de Recursos Humanos que responde dudas de empleados sobre sus derechos \
laborales en México. Fecha de hoy: {fecha}.

Debes responder usando EXCLUSIVAMENTE los fragmentos de ley que se te entregan (Ley Federal del Trabajo y \
artículo 123 de la Constitución). Reglas:
1. Toda afirmación de tu respuesta debe estar respaldada por al menos un fragmento. No uses conocimiento \
externo: ni cifras, ni montos (salario mínimo, UMA), ni otras leyes (Seguro Social, ISR, INFONAVIT, SAR), ni \
reglamentos o políticas de la empresa.
2. Si los fragmentos NO contienen la respuesta, usa estado "sin_respuesta": en "respuesta" explica en una o dos \
oraciones qué no está en estos documentos, sin adivinar el dato. No cites fundamentos que no respondan.
3. Si los fragmentos responden solo una parte, usa estado "parcial": responde lo que sí está y describe en \
"no_cubierto" lo que falta.
4. Escribe para un empleado sin formación jurídica: segunda persona ("tienes derecho a..."), oraciones cortas, \
explica los términos legales, de 2 a 6 oraciones. Puedes hacer cálculos aritméticos simples a partir de reglas \
que estén en los fragmentos (por ejemplo, días de vacaciones según antigüedad), explicando la regla.
5. Si la respuesta depende de hechos que no conoces (por ejemplo, si hubo causa justificada), dilo.
6. En "fundamentos" incluye cada fragmento que respalda la respuesta, con una cita LITERAL copiada del \
fragmento (de 5 a 50 palabras, sin cambiar ni una palabra) y la fracción si aplica.
7. Los fragmentos de "transitorios" sirven para fechas de entrada en vigor y calendarios graduales.
8. Si la pregunta no es sobre derechos laborales, usa estado "sin_respuesta"."""

ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "estado": {"type": "string", "enum": ["respondida", "parcial", "sin_respuesta"]},
        "respuesta": {"type": "string", "description": "Respuesta en lenguaje claro para el empleado."},
        "fundamentos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fragmento": {"type": "string", "description": "Identificador, p. ej. F3"},
                    "fraccion": {"type": "string", "description": "Fracción (romano) si aplica, o vacío"},
                    "cita_textual": {"type": "string", "description": "Copia literal del fragmento"},
                },
                "required": ["fragmento", "cita_textual"],
            },
        },
        "no_cubierto": {"type": "string", "description": "Qué parte de la pregunta no está en los documentos (vacío si nada)."},
    },
    "required": ["estado", "respuesta", "fundamentos", "no_cubierto"],
}

def _formatear_fragmentos(frags: list[Fragmento]) -> str:
    bloques = []
    for i, f in enumerate(frags, 1):
        bloques.append(f"<fragmento id=\"F{i}\" fuente=\"{f.etiqueta}\" contexto=\"{f.contexto}\">\n{f.texto}\n</fragmento>")
    return "\n\n".join(bloques)

def _norm(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cita_en_texto(cita: str, texto: str, umbral: float = 92.0) -> bool:
    c, t = _norm(cita), _norm(texto)
    if not c or len(c.split()) < 3:
        return False
    if c in t:
        return True
    from rapidfuzz import fuzz
    return fuzz.partial_ratio(c, t) >= umbral

def _ubicar_fraccion(cita: str, frag: Fragmento) -> str:
    for m in re.finditer(r"(?m)^((?:[IVXLC]+)(?: (?:Bis|Ter|Quáter))?)\.\s", frag.texto):
        inicio = m.start()
        sig = re.search(r"(?m)^(?:[IVXLC]+)(?: (?:Bis|Ter|Quáter))?\.\s", frag.texto[m.end():])
        fin = m.end() + sig.start() if sig else len(frag.texto)
        if cita_en_texto(cita, frag.texto[inicio:fin]):
            return m.group(1)
    return frag.fracciones[0] if len(frag.fracciones) == 1 else ""

def _numeros_en(texto: str) -> set[int]:
    texto = re.sub(r"(?i)art[íi]culos?\s+\d+(?:-[A-Z])?", " ", texto)
    texto = re.sub(r"(?i)fracci[óo]n(?:es)?\s+[IVXLC]+", " ", texto)
    return {int(n) for n in re.findall(r"\b\d{1,4}\b", texto.replace(",", "")) if not 1900 <= int(n) <= 2100}

def numero_respaldado(n: int, texto_fuente: str) -> bool:
    from num2words import num2words
    t = _norm(texto_fuente)
    candidatos = {str(n), _norm(num2words(n, lang="es"))}
    try:
        candidatos.add(_norm(num2words(n, lang="es", to="ordinal")))
    except Exception:
        pass
    if n == 100:
        candidatos |= {"ciento por ciento", "doble"}
    if n == 200:
        candidatos |= {"doscientos por ciento", "triple"}
    return any(re.search(rf"\b{re.escape(c)}\b", t) for c in candidatos if c)

@dataclass
class Fundamento:
    etiqueta: str
    fraccion: str
    cita: str
    fragmento_id: str
    valida: bool

@dataclass
class Respuesta:
    pregunta: str
    estado: str
    respuesta: str
    fundamentos: list[Fundamento]
    no_cubierto: str
    aviso: str
    advertencias: list[str]
    fragmentos: list[Fragmento]
    consultas: list[str]
    version: str
    uso: Uso
    segundos_total: float
    estado_modelo: str = ""
    respuesta_modelo: str = ""

    @property
    def costo_usd(self) -> float:
        return self.uso.costo_usd

    def a_markdown(self) -> str:
        iconos = {"respondida": "✅ Respondida con base en la ley", "parcial": "⚠️ Respuesta parcial",
                  "sin_respuesta": "❌ No está en los documentos"}
        md = [f"### {self.pregunta}", f"**{iconos.get(self.estado, self.estado)}**", "", self.respuesta]
        if self.aviso:
            md += ["", f"> **Aviso:** {self.aviso}"]
        validos = [f for f in self.fundamentos if f.valida]
        if validos:
            md += ["", "**Fundamento legal**"]
            for f in validos:
                etiqueta = f.etiqueta + (f", fracción {f.fraccion}" if f.fraccion and "fr." not in f.etiqueta else "")
                md.append(f"- **{etiqueta}**: “{f.cita}”")
        for a in self.advertencias:
            md.append(f"\n<sub>⚠ {a}</sub>")
        md.append(f"\n<sub>{self.version} · {self.segundos_total:.1f} s · US${self.costo_usd:.4f} · "
                  f"{self.uso.llamadas} llamadas al modelo</sub>")
        return "\n".join(md)

    def mostrar(self):
        try:
            from IPython.display import Markdown, display
            display(Markdown(self.a_markdown()))
        except ImportError:  # fuera de Jupyter
            print(self.a_markdown())

    def a_dict(self) -> dict:
        return {
            "pregunta": self.pregunta, "estado": self.estado, "respuesta": self.respuesta,
            "estado_modelo": self.estado_modelo, "respuesta_modelo": self.respuesta_modelo,
            "fundamentos": [f.__dict__ for f in self.fundamentos], "no_cubierto": self.no_cubierto,
            "aviso": self.aviso, "advertencias": self.advertencias,
            "fragmentos": [{"id": f.id, "etiqueta": f.etiqueta, "claves": f.claves, "texto": f.texto}
                           for f in self.fragmentos],
            "consultas": self.consultas, "version": self.version,
            "llamadas": self.uso.llamadas, "tokens_entrada": self.uso.tokens_entrada,
            "tokens_salida": self.uso.tokens_salida, "costo_usd": round(self.uso.costo_usd, 6),
            "segundos_llm": round(self.uso.segundos, 3),
            "segundos_total": round(self.segundos_total - self.uso.segundos_vivos + self.uso.segundos, 3),
            "segundos_recuperacion_y_guardas": round(self.segundos_total - self.uso.segundos_vivos, 3),
            "desde_cache": self.uso.desde_cache,
        }

class Asistente:
    def __init__(self, config: ConfigAsistente, indice: IndiceHibrido, llm: ClienteLLM, fecha: str | None = None):
        self.config = config
        self.indice = indice
        self.llm = llm
        self.fecha = fecha or date.today().isoformat()

    @classmethod
    def cargar(cls, version: str | ConfigAsistente = "v1_articulo", llm: ClienteLLM | None = None,
               embedder=None, fecha: str | None = None) -> "Asistente":
        """Construye (o lee de caché) el corpus y el índice para la versión pedida."""
        config = VERSIONES[version] if isinstance(version, str) else version
        frags = construir_corpus(config.chunking, config.max_chars_chunk)
        indice = IndiceHibrido(frags, embedder=embedder, usar_bm25=config.usar_bm25,
                               usar_embeddings=config.usar_embeddings)
        return cls(config, indice, llm or ClienteLLM(), fecha)

    def _consultas(self, pregunta: str, uso: Uso) -> list[str]:
        consultas = [pregunta]
        if self.config.usar_glosario:
            extra = expandir_con_glosario(pregunta)
            if extra:
                consultas.append(f"{pregunta} {extra}")
        if self.config.usar_reescritura:
            out = self.llm.json(modelo=self.config.modelo, sistema=SISTEMA_REESCRITURA,
                                usuario=f"Pregunta del empleado: {pregunta}", esquema=ESQUEMA_REESCRITURA,
                                nombre="consultas_juridicas", descripcion="Consultas de búsqueda en lenguaje jurídico",
                                temperatura=0.0, max_tokens=300, uso=uso, etapa="reescritura")
            consultas += [c for c in out.get("consultas", []) if isinstance(c, str)][:4]
        return consultas

    def recuperar(self, pregunta: str, uso: Uso | None = None) -> tuple[list[Fragmento], list[str]]:
        uso = uso or Uso()
        consultas = self._consultas(pregunta, uso)
        res = self.indice.buscar(consultas, self.config.k_por_consulta, self.config.k_final, self.config.rrf_k)
        return [r.fragmento for r in res], consultas

    def preguntar(self, pregunta: str) -> Respuesta:
        t0 = time.perf_counter()
        uso = Uso()
        frags, consultas = self.recuperar(pregunta, uso)
        usuario = (f"<fragmentos>\n{_formatear_fragmentos(frags)}\n</fragmentos>\n\n"
                   f"Pregunta del empleado: {pregunta}")
        out = self.llm.json(modelo=self.config.modelo, sistema=SISTEMA_RESPUESTA.format(fecha=self.fecha),
                            usuario=usuario, esquema=ESQUEMA_RESPUESTA, nombre="responder",
                            descripcion="Respuesta fundamentada para el empleado",
                            temperatura=self.config.temperatura, max_tokens=self.config.max_tokens,
                            uso=uso, etapa="respuesta")
        estado_modelo = out.get("estado", "sin_respuesta")
        respuesta_modelo = (out.get("respuesta") or "").strip()
        estado, texto = estado_modelo, respuesta_modelo
        advertencias: list[str] = []
        fundamentos: list[Fundamento] = []
        for f in out.get("fundamentos", []) or []:
            m = re.search(r"\d+", str(f.get("fragmento", "")))
            idx = int(m.group()) - 1 if m else -1
            cita = (f.get("cita_textual") or "").strip()
            if not (0 <= idx < len(frags)):
                fundamentos.append(Fundamento("¿fragmento inexistente?", "", cita, str(f.get("fragmento")), False))
                continue
            fr = frags[idx]
            valida = cita_en_texto(cita, fr.texto) if self.config.verificar_citas else True
            fraccion = _ubicar_fraccion(cita, fr) if valida else (f.get("fraccion") or "")
            etiqueta = fr.etiqueta
            if fraccion and len(fr.fracciones) != 1:
                etiqueta = re.sub(r", fr\..*$", "", etiqueta) + f", fr. {fraccion}"
            fundamentos.append(Fundamento(etiqueta, fraccion, cita, fr.id, valida))
        invalidas = [f for f in fundamentos if not f.valida]
        if invalidas and self.config.verificar_citas:
            advertencias.append(f"{len(invalidas)} cita(s) no se encontraron literalmente en la ley y se omitieron.")
        validos = [f for f in fundamentos if f.valida]
        if estado != "sin_respuesta" and not validos and self.config.verificar_citas:
            estado, texto = "sin_respuesta", AVISO_SIN_RESPUESTA
            advertencias.append("El modelo respondió sin una cita verificable; se sustituyó por el aviso de 'sin respuesta'.")
        if self.config.verificar_numeros and estado != "sin_respuesta":
            fuente = " ".join(next(fr.texto for fr in frags if fr.id == f.fragmento_id) for f in validos)
            sueltos = sorted(n for n in _numeros_en(texto) - _numeros_en(pregunta) if not numero_respaldado(n, fuente))
            if sueltos:
                advertencias.append("Cifras que no aparecen literalmente en los artículos citados (pueden ser un "
                                    f"cálculo; verifícalas): {', '.join(map(str, sueltos))}.")
        aviso = ""
        if estado == "sin_respuesta":
            aviso = "Esta pregunta no tiene respuesta en la LFT ni en el artículo 123 constitucional."
            fundamentos = []
        elif estado == "parcial":
            aviso = "Una parte de tu pregunta no está en la LFT ni en el artículo 123 constitucional" + \
                    (f": {out.get('no_cubierto').strip()}" if (out.get("no_cubierto") or "").strip() else ".")
        return Respuesta(pregunta, estado, texto, fundamentos, (out.get("no_cubierto") or "").strip(), aviso,
                         advertencias, frags, consultas, self.config.nombre, uso, time.perf_counter() - t0,
                         estado_modelo, respuesta_modelo)