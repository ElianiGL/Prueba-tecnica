#Recuperación híbrida: BM25 (léxico) + embeddings multilingües (semántico), fusionados con RRF.

from __future__ import annotations
import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import DIR_PROC, MODELO_EMBEDDINGS
from .parseo import Fragmento

_STOP = set("""a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos
el ella ellas ellos en entre era es esa ese eso esta este esto estos estas fue fueron ha han hasta hay la las
le les lo los mas me mi mis mucho muy no nos o os otra otro para pero poco por porque que quien se sea ser si
sido sin sobre su sus tambien te tiene tienen todo todos tu tus un una uno unos unas y ya yo""".split())

try:
    import snowballstemmer
    _stem = snowballstemmer.stemmer("spanish").stemWord
except Exception:
    _stem = lambda w: w

def _sin_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def tokenizar(texto: str) -> list[str]:
    texto = _sin_acentos(texto.lower())
    return [_stem(t) for t in re.findall(r"[a-zñ0-9]+", texto) if t not in _STOP and len(t) > 1]

GLOSARIO: list[tuple[str, str]] = [
    (r"\b(corr(er|ieron|en|an)|correr|despid|liquid|me sacaron|me echaron)", "despido rescisión de la relación de trabajo indemnización reinstalación"),
    (r"\bembaraz|\bmaternidad|\bincapacidad por maternidad", "mujeres embarazadas maternidad periodo de gestación descanso de seis semanas parto"),
    (r"\bfiniquito", "terminación de la relación de trabajo parte proporcional aguinaldo vacaciones prima de antigüedad"),
    (r"\bhoras? extra", "jornada extraordinaria prolongarse horas de trabajo extraordinario ciento por ciento más"),
    (r"\bfestiv|dia feriado|puente", "días de descanso obligatorio"),
    (r"\bdomingo", "prima dominical"),
    (r"\bincapacidad|me lastim|accidente|me enferm", "riesgos de trabajo accidente de trabajo enfermedad de trabajo incapacidad temporal suspensión"),
    (r"\bacos|hostig|me grita|maltrat|humill", "hostigamiento acoso sexual malos tratos injurias rescisión sin responsabilidad para el trabajador"),
    (r"\bhome ?office|desde casa|remot|teletrabaj", "teletrabajo"),
    (r"\boutsourcing|subcontrat|agencia", "subcontratación de personal servicios especializados"),
    (r"\butilidades|ptu|reparto", "participación de los trabajadores en las utilidades de las empresas"),
    (r"\bsueldo|me pagan|pago", "salario"),
    (r"\bdescuent|me quitan del sueldo|me cobran", "descuentos en los salarios"),
    (r"\bpapa\b|paternidad|nacio mi hij", "permiso de paternidad"),
    (r"\bamamant|lactancia|dar pecho", "periodo de lactancia reposos extraordinarios"),
    (r"\bcontrato\b|no firme|sin contrato", "condiciones de trabajo constar por escrito"),
    (r"\bprueba\b", "periodo a prueba"),
    (r"\bmenor(es)? de edad|anos.*trabajar|edad.*trabajar", "menores de quince años trabajo de los menores"),
    (r"\bdemand|cuanto tiempo tengo|plazo", "prescripción acciones de trabajo"),
    (r"\brenunc", "terminación de la relación de trabajo retiro voluntario"),
    (r"\bantig", "prima de antigüedad"),
    (r"\bfalt(e|ar|as)\b|llegar tarde|retard", "faltas de asistencia sin permiso del patrón"),
    (r"\bjornada|horario|cuantas horas", "duración de la jornada de trabajo jornada máxima"),
    (r"\bvacacion", "vacaciones prima vacacional"),
    (r"\bvale|despensa|especie", "salario en moneda de curso legal mercancías vales fichas"),
]

def expandir_con_glosario(pregunta: str) -> str:
    q = _sin_acentos(pregunta.lower())
    extras = [exp for patron, exp in GLOSARIO if re.search(patron, q)]
    return " ".join(extras)

class EmbedderST:
    """Embeddings locales con sentence-transformers. e5 usa prefijos 'query:'/'passage:'."""
    def __init__(self, modelo: str = MODELO_EMBEDDINGS):
        from sentence_transformers import SentenceTransformer
        self.nombre = modelo
        self.modelo = SentenceTransformer(modelo)
        self._e5 = "e5" in modelo.lower()
    def documentos(self, textos: list[str]) -> np.ndarray:
        textos = [("passage: " + t) if self._e5 else t for t in textos]
        return self.modelo.encode(textos, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    def consultas(self, textos: list[str]) -> np.ndarray:
        textos = [("query: " + t) if self._e5 else t for t in textos]
        return self.modelo.encode(textos, normalize_embeddings=True)

class EmbedderTFIDF:
    """Respaldo sin red (solo pruebas): TF-IDF de n-gramas de caracteres. No es semántico."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.nombre = "tfidf-char"
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=lambda s: _sin_acentos(s.lower()))
        self._ajustado = False
    def documentos(self, textos):
        m = self.vec.fit_transform(textos)
        self._ajustado = True
        return _normaliza(m.toarray())
    def consultas(self, textos):
        return _normaliza(self.vec.transform(textos).toarray())

def _normaliza(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-12, None)

@dataclass
class Resultado:
    fragmento: Fragmento
    score: float
    rangos: dict

class IndiceHibrido:
    def __init__(self, fragmentos: list[Fragmento], embedder=None, usar_bm25: bool = True,
                 usar_embeddings: bool = True, cache_dir: Path = DIR_PROC):
        from rank_bm25 import BM25Okapi
        self.fragmentos = fragmentos
        self.usar_bm25, self.usar_embeddings = usar_bm25, usar_embeddings
        self.bm25 = BM25Okapi([tokenizar(f.texto_indexado) for f in fragmentos])
        self.embedder = None
        self.matriz = None
        if usar_embeddings:
            self.embedder = embedder or EmbedderST()
            textos = [f.texto_indexado for f in fragmentos]
            h = hashlib.sha1(("\n".join(textos) + self.embedder.nombre).encode()).hexdigest()[:16]
            ruta = cache_dir / f"emb_{h}.npy"
            if ruta.exists() and not isinstance(self.embedder, EmbedderTFIDF):
                self.matriz = np.load(ruta)
            else:
                self.matriz = np.asarray(self.embedder.documentos(textos), dtype=np.float32)
                if not isinstance(self.embedder, EmbedderTFIDF):
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(ruta, self.matriz)
        self._por_clave = defaultdict(list)
        for i, f in enumerate(fragmentos):
            for c in f.claves:
                self._por_clave[c].append(i)
    def _rank_bm25(self, consulta: str, k: int) -> list[int]:
        s = self.bm25.get_scores(tokenizar(consulta))
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]
    def _rank_emb(self, consulta: str, k: int) -> tuple[list[int], np.ndarray]:
        q = self.embedder.consultas([consulta])[0]
        s = self.matriz @ q
        return [int(i) for i in np.argsort(-s)[:k]], s
    def buscar(self, consultas: list[str], k_por_consulta: int = 20, k_final: int = 8, rrf_k: int = 60) -> list[Resultado]:
        puntaje = defaultdict(float)
        rangos = defaultdict(dict)
        for qi, q in enumerate(consultas):
            if not q.strip():
                continue
            listas = {}
            if self.usar_bm25:
                listas[f"bm25_q{qi}"] = self._rank_bm25(q, k_por_consulta)
            if self.usar_embeddings:
                listas[f"emb_q{qi}"] = self._rank_emb(q, k_por_consulta)[0]
            for nombre, lista in listas.items():
                for r, i in enumerate(lista):
                    puntaje[i] += 1.0 / (rrf_k + r + 1)
                    rangos[i][nombre] = r + 1
            for m in re.finditer(r"art[íi]culo\s+(\d+(?:-[A-Z])?(?:\s+Bis)?)", q, re.I):
                for i in self._por_clave.get(f"LFT:{m.group(1)}", []):
                    puntaje[i] += 1.0 / (rrf_k + 1)
                    rangos[i]["referencia_explicita"] = 1
        orden = sorted(puntaje, key=lambda i: -puntaje[i])[:k_final]
        return [Resultado(self.fragmentos[i], puntaje[i], dict(rangos[i])) for i in orden]