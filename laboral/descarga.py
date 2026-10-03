#Descarga de las versiones vigentes desde el sitio oficial de la Cámara de Diputados.

from __future__ import annotations
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from .config import DIR_RAW, FUENTES

MANIFIESTO = DIR_RAW / "manifiesto.json"

def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()

def descargar(forzar: bool = False, timeout: int = 120) -> dict:
    import requests
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    manifiesto = json.loads(MANIFIESTO.read_text()) if MANIFIESTO.exists() else {}
    for clave, url in FUENTES.items():
        destino = DIR_RAW / f"{clave}.pdf"
        if destino.exists() and not forzar:
            print(f"[descarga] {destino.name} ya existe; se reutiliza.")
        else:
            print(f"[descarga] {url}")
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 asistente-laboral"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError(f"La respuesta de {url} no es un PDF.")
            destino.write_bytes(r.content)
            manifiesto.setdefault(clave, {})["descargado_utc"] = datetime.now(timezone.utc).isoformat()
        manifiesto.setdefault(clave, {}).update(url=url, archivo=destino.name, sha256=_sha256(destino))
    MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    return manifiesto

def registrar_ultima_reforma(clave: str, texto: str) -> None:
    m = re.search(r"Última Reforma DOF (\d{2}-\d{2}-\d{4})", texto)
    if not m:
        return
    manifiesto = json.loads(MANIFIESTO.read_text()) if MANIFIESTO.exists() else {}
    manifiesto.setdefault(clave, {})["ultima_reforma_dof"] = m.group(1)
    MANIFIESTO.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))