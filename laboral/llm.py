# Cliente LLM (Anthropic) con salida estructurada

from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from .config import DIR_CACHE, costo_usd

@dataclass
class Uso:
    llamadas: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    costo_usd: float = 0.0
    segundos: float = 0.0
    desde_cache: int = 0
    segundos_vivos: float = 0.0
    detalle: list = field(default_factory=list)
    def sumar(self, etapa: str, modelo: str, tin: int, tout: int, seg: float, cache: bool):
        c = costo_usd(modelo, tin, tout)
        self.llamadas += 1
        self.tokens_entrada += tin
        self.tokens_salida += tout
        self.costo_usd += c
        self.segundos += seg
        self.desde_cache += int(cache)
        self.detalle.append({"etapa": etapa, "modelo": modelo, "tokens_entrada": tin, "tokens_salida": tout, "costo_usd": round(c, 6), "segundos": round(seg, 3), "cache": cache})

class ClienteLLM:
    def __init__(self, usar_cache: bool = True, cache_dir: Path = DIR_CACHE / "llm", max_reintentos: int = 5):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta la variable de entorno ANTHROPIC_API_KEY (ver README).")
        self._cliente = anthropic.Anthropic(max_retries=max_reintentos)
        self.usar_cache = usar_cache
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
    def _clave(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.cache_dir / f"{h[:2]}/{h}.json"
    def json(self, *, modelo: str, sistema: str, usuario: str, esquema: dict, nombre: str,
             descripcion: str, temperatura: float = 0.0, max_tokens: int = 1500,
             uso: Uso | None = None, etapa: str = "") -> dict:
        payload = {"modelo": modelo, "sistema": sistema, "usuario": usuario, "esquema": esquema,
                   "nombre": nombre, "temperatura": temperatura, "max_tokens": max_tokens}
        ruta = self._clave(payload)
        if self.usar_cache and ruta.exists():
            t_cache = time.perf_counter()
            guardado = json.loads(ruta.read_text(encoding="utf-8"))
            if uso is not None:
                uso.segundos_vivos += time.perf_counter() - t_cache
                uso.sumar(etapa, modelo, guardado["tokens_entrada"], guardado["tokens_salida"],
                          guardado["segundos"], cache=True)
            return guardado["salida"]
        t0 = time.perf_counter()
        resp = self._cliente.messages.create(
            model=modelo,
            max_tokens=max_tokens,
            temperature=temperatura,
            system=sistema,
            messages=[{"role": "user", "content": usuario}],
            tools=[{"name": nombre, "description": descripcion, "input_schema": esquema}],
            tool_choice={"type": "tool", "name": nombre},)
        seg = time.perf_counter() - t0
        if uso is not None:
            uso.segundos_vivos += seg
        salida = next((b.input for b in resp.content if b.type == "tool_use"), {})
        tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        if uso is not None:
            uso.sumar(etapa, modelo, tin, tout, seg, cache=False)
        if self.usar_cache:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_text(json.dumps({"salida": salida, "tokens_entrada": tin, "tokens_salida": tout,
                                        "segundos": seg, "stop_reason": resp.stop_reason},
                                       ensure_ascii=False), encoding="utf-8")
        return salida