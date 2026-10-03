#Configuración central del asistente.

from __future__ import annotations
import os
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATA = RAIZ / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROC = DIR_DATA / "procesado"
DIR_CACHE = DIR_DATA / "cache"
DIR_RESULTADOS = RAIZ / "resultados"

if (RAIZ / ".env").exists():
    for _l in (RAIZ / ".env").read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

FUENTES = {
    "LFT": "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFT.pdf",
    "CPEUM": "https://www.diputados.gob.mx/LeyesBiblio/pdf/CPEUM.pdf",
}

MODELO_ASISTENTE = "claude-haiku-4-5"
MODELO_JUEZ = "claude-sonnet-5-5"
PRECIOS_USD_MTOK = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

MODELO_EMBEDDINGS = os.environ.get("LABORAL_EMBEDDINGS", "intfloat/multilingual-e5-small")

@dataclass(frozen=True)
class ConfigAsistente:
    nombre: str = "v1_articulo"
    chunking: str = "articulo"
    max_chars_chunk: int = 2200
    usar_bm25: bool = True
    usar_embeddings: bool = True
    usar_glosario: bool = True
    usar_reescritura: bool = True
    k_por_consulta: int = 20
    k_final: int = 8
    rrf_k: int = 60
    modelo: str = MODELO_ASISTENTE
    temperatura: float = 0.0
    max_tokens: int = 1500
    verificar_citas: bool = True
    verificar_numeros: bool = True
    def a_dict(self) -> dict:
        return asdict(self)
    def con(self, **cambios) -> "ConfigAsistente":
        return replace(self, **cambios)

VERSIONES = {
    "v1_articulo": ConfigAsistente(),
    "v2_fraccion": ConfigAsistente(nombre="v2_fraccion", chunking="fraccion"),
    "v3_sin_reescritura": ConfigAsistente(nombre="v3_sin_reescritura", usar_reescritura=False, usar_glosario=False),
    "v4_sonnet": ConfigAsistente(nombre="v4_sonnet", modelo="claude-sonnet-5-5"),
}

def costo_usd(modelo: str, tokens_entrada: int, tokens_salida: int) -> float:
    pin, pout = PRECIOS_USD_MTOK.get(modelo, (0.0, 0.0))
    return tokens_entrada / 1e6 * pin + tokens_salida / 1e6 * pout