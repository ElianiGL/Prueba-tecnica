"""Pruebas sin red ni API: python -m pytest -q"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from laboral.asistente import Asistente, cita_en_texto, numero_respaldado
from laboral.config import ConfigAsistente
from laboral.parseo import extraer_art123, fragmentar, segmentar_lft, segmentar_transitorios
from laboral.recuperacion import EmbedderTFIDF, IndiceHibrido, expandir_con_glosario
from evaluacion import metricas, estadistica, errores

from simulado import AQUI, LLMSimulado, preparar_entorno

def _corpus(estrategia="articulo", max_chars=2200):
    arts, tr = segmentar_lft((AQUI / "fixture_lft.txt").read_text())
    a123, trc = extraer_art123((AQUI / "fixture_cpeum.txt").read_text())
    trans = segmentar_transitorios(tr, "LFT") + segmentar_transitorios(trc, "CPEUM", r"\b123\b")
    return fragmentar(arts + a123, trans, estrategia, max_chars)

def test_segmentacion_lft():
    arts, tr = segmentar_lft((AQUI / "fixture_lft.txt").read_text())
    ids = [a.articulo for a in arts]
    assert ids == ["1", "2", "3 Bis", "39-A", "47", "76", "132", "133"]
    a76 = next(a for a in arts if a.articulo == "76")
    assert "reformado" not in a76.texto and "Secretaría" not in a76.texto
    assert a76.capitulo == "Capítulo IV - Vacaciones"
    a132 = next(a for a in arts if a.articulo == "132")
    assert [f.fraccion for f in a132.fracciones] == ["I", "II", "II Bis", "III"]
    assert "TRANSITORIOS" in tr and "1970" not in next(a for a in arts if a.articulo == "133").texto

def test_art123_y_transitorios():
    a123, trc = extraer_art123((AQUI / "fixture_cpeum.txt").read_text())
    assert [a.apartado for a in a123] == ["", "A", "B"]
    assert [f.fraccion for f in a123[1].fracciones] == ["I", "II", "XI"]
    t = segmentar_transitorios(trc, "CPEUM", r"\b123\b")
    assert len(t) == 1 and t[0]["anio"] == 2026

def test_estrategias_de_fragmentacion():
    por_art = _corpus("articulo")
    por_fr = _corpus("fraccion")
    assert len(por_fr) > len(por_art)
    f = next(x for x in por_fr if "LFT:132:II Bis" in x.claves)
    assert "paternidad" in f.texto and f.etiqueta == "LFT, art. 132, fr. II Bis"

def test_glosario_y_busqueda():
    assert "rescisión" in expandir_con_glosario("¿Me pueden correr?")
    frags = _corpus("fraccion")
    idx = IndiceHibrido(frags, embedder=EmbedderTFIDF())
    res = idx.buscar(["permiso por nacimiento de mi hijo " + expandir_con_glosario("ya nació mi hijo, soy papá")], k_final=3)
    assert any("LFT:132" in r.fragmento.claves for r in res)

def test_verificacion_de_citas_y_numeros():
    texto = "disfrutarán de un periodo anual de vacaciones pagadas, que en ningún caso podrá ser inferior a doce días laborables"
    assert cita_en_texto("en ningun caso podra ser inferior a doce dias laborables", texto)
    assert not cita_en_texto("nunca menos de quince días naturales de descanso", texto)
    assert numero_respaldado(12, texto) and not numero_respaldado(15, texto)

def test_asistente_simulado(tmp_path, monkeypatch):
    preparar_entorno(tmp_path, monkeypatch)
    a = Asistente.cargar(ConfigAsistente(), llm=LLMSimulado(), embedder=EmbedderTFIDF(), fecha="2026-10-01")
    r = a.preguntar("¿Cuántos días de vacaciones me tocan?")
    assert r.estado == "respondida" and r.fundamentos and r.fundamentos[0].valida
    assert any("99" in w for w in r.advertencias)
    r2 = a.preguntar("¿Me pueden correr por estar embarazada?")
    assert r2.estado == "sin_respuesta"
    r3 = a.preguntar("¿Cuánto vale la UMA?")
    assert r3.estado == "sin_respuesta" and "no tiene respuesta" in r3.aviso
    assert r.costo_usd > 0 and "Fundamento legal" in r.a_markdown()

def test_metricas_basicas():
    frs = [{"id": "a", "claves": ["LFT:76"]}, {"id": "b", "claves": ["LFT:132", "LFT:132:XXVII Bis"]}]
    m = metricas.metricas_recuperacion(frs, ["LFT:132:XXVII Bis", "LFT:80"])
    assert m["recall_articulo"] == 0.5 and m["recall_fraccion"] == 0.5 and m["mrr"] == 0.5
    afirm = [{"texto": "x", "tipo": "factual"}, {"texto": "y", "tipo": "factual"}, {"texto": "z", "tipo": "meta"}]
    ver = [{"etiqueta": "respaldada", "fragmentos": ["F1"]}, {"etiqueta": "no_respaldada", "fragmentos": []},
           {"etiqueta": "respaldada", "fragmentos": []}]
    f = metricas.fidelidad(afirm, ver, frs, [{"fragmento_id": "b", "valida": True}])
    assert f["fidelidad"] == 0.5 and f["fidelidad_estricta"] == 0.0
    assert math.isnan(metricas.fidelidad([], [], [], [])["fidelidad"])

def test_estadistica():
    import numpy as np
    a = np.array([1] * 30 + [0] * 30, dtype=float)
    assert estadistica.mcnemar_exacto(a, a)[2] == 1.0
    b = a.copy(); b[30:50] = 1
    n01, n10, p = estadistica.mcnemar_exacto(a, b)
    assert n10 == 20 and p < 0.001

def test_evaluacion_completa_simulada(tmp_path, monkeypatch):
    """Corre evaluacion.run de punta a punta con el LLM simulado sobre los fixtures."""
    import evaluacion.run as run
    _, res = preparar_entorno(tmp_path, monkeypatch)
    monkeypatch.setattr(run, "descargar", lambda: {"LFT": {"ultima_reforma_dof": "14-05-2026", "sha256": "x" * 64}})
    monkeypatch.setattr(run, "ClienteLLM", lambda **k: LLMSimulado())
    import laboral.recuperacion as rec
    monkeypatch.setattr(rec, "EmbedderST", EmbedderTFIDF)
    df = run.main(["--limite", "12", "--aa", "--n-anotacion", "8"])
    assert set(df.version) == {"v1_articulo", "v2_fraccion", "v1_articulo_rep"}
    rep = (res / "reporte.md").read_text()
    for seccion in ("Resumen por versión", "Comparación de versiones", "Prueba A/A", "¿Dónde y por qué falla?",
                    "Recomendación"):
        assert seccion in rep
    m = pd.read_csv(res / "anotacion_manual.csv")
    oculto = pd.read_csv(res / "anotacion_juez_oculto.csv")
    m["etiqueta_humana"] = oculto["etiqueta_juez"].values
    m.loc[0, "etiqueta_humana"] = "no_respaldada" if oculto.loc[0, "etiqueta_juez"] == "respaldada" else "respaldada"
    m.to_csv(res / "anotacion_manual.csv", index=False)
    from evaluacion.acuerdo_juez import calcular_acuerdo
    ac = calcular_acuerdo()
    assert ac["n"] == len(m) and ac["acuerdo_binario"] == pytest.approx(1 - 1 / len(m))