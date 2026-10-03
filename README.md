# Asistente de consulta de derechos laborales (LFT + art. 123 constitucional)

Asistente que responde preguntas de empleados usando únicamente la Ley Federal del Trabajo y el artículo 123 de la Constitución, se realiza más de una evaluación separada que mide qué tan confiable es antes de ponerlo frente a los empleados.

```
Glosario:
asistente-laboral/
  notebooks/
    01_asistente.ipynb              Hace las preguntas
    02_evaluacion.ipynb             Corre y explora la evaluación
    03_anotacion_manual.ipynb       Califica a mano una muestra para validar al juez
  laboral/                          El asistente
    descarga.py   
    parseo.py
    recuperacion.py   
    asistente.py   
    llm.py   
    config.py
  evaluacion/                       La evaluación
    run.py  
    juez.py  
    metricas.py  
    estadistica.py  
    errores.py  
    acuerdo_juez.py  
    validar.py  
    reporte.py
  data/preguntas.json  (y .csv)     63 preguntas con artículos esperados y puntos clave
  resultados/                       Se llena al evaluar: reporte.md, CSVs, gráficas
  tests/                            Pruebas sin red ni API (LLM simulado)
    requirements.txt  
    Makefile  
    .env.example
```

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Hacer preguntas — abre `notebooks/01_asistente.ipynb` y ejecuta las celdas:
```python
r = preguntar("¿Me pueden correr por estar embarazada?")
```
La primera vez descarga `LFT.pdf` y `CPEUM.pdf` de `diputados.gob.mx/LeyesBiblio`, el modelo de embeddings (~470 MB) y construye el índice (1–3 min). Si tu red bloquea el sitio, coloca los PDF a mano en `data/raw/` con esos nombres.

Evaluación completa (un comando):
```bash
python -m evaluacion.run --aa
```
Descarga/valida documentos para correr las versiones `v1_articulo` y `v2_fraccion` (y una réplica A/A de v1) sobre las 63 preguntas, juez, métricas, comparación estadística, taxonomía de errores, `resultados/reporte.md`.
Duración estimada 15–25 min y ~US$5–8 solo la primera vez; las re-ejecuciones salen de caché (gratis, mismos números).
Otras opciones: `--versiones v1_articulo v3_sin_reescritura`, `--limite 5` (prueba rápida), `--sin-cache`.

Validar al juez: Tras la evaluación, califica `resultados/anotacion_manual.csv` (o con `notebooks/03_anotacion_manual.ipynb`) y corre `python -m evaluacion.acuerdo` (o vuelve a correr la evaluación, que lo integra al reporte).

Pruebas sin API: `python -m pytest -q` (Son 9 pruebas: Parseo, recuperación, guardas, métricas, estadística y la evaluación de punta a punta con un LLM simulado).

## Arquitectura

Pregunta: Glosario coloquial y jurídico
  Claude Haiku: Reescritura en lenguaje de ley
  Pregunta original
    Embeddings e5 por consulta, fusión RRF, top-8 fragmentos
    Claude Haiku (salida estructurada vía tool use, temperatura 0)
      Guardas deterministas: Cita literal existe en el fragmento, sin cita válida "sin respuesta", cifras de la respuesta presentes en lo citado (si no, advertencia)
      Respuesta clara + artículos/fracciones con texto citado + aviso explícito + costo, tokens y segundos por pregunta

## Cómo resolví cada reto

### 1. Los documentos son demasiado largos, segmentación estructural + recuperación híbrida

- Parseo por la estructura de la ley, no por tamaño fijo: Título → Capítulo → Artículo → Fracción (`laboral/parseo.py`). Se eliminan encabezados de página y notas de reforma ("Artículo reformado DOF …") que no son texto legal. Los artículos se detectan con una expresión que cubre `1o.-`, `3o. Bis.-`, `39-A.`, `330-E.-` y exige secuencia creciente (así una línea que empieza con "Artículo 50 de esta Ley" no corta nada).
- El título y capítulo viajan como metadato y se agregan al texto indexado ("Capítulo IV – Vacaciones"): ayuda cuando la pregunta nombra el tema y el artículo no.
- Del artículo 123 se separan apartados A y B y sus fracciones. Se conservan los transitorios de decretos desde 2012 (ahí vive, por ejemplo, el calendario gradual de la jornada de 40 horas de 2026).
- Recuperación híbrida (`laboral/recuperacion.py`): BM25 con stemming en español (acierta con términos exactos como "aguinaldo") + embeddings multilingües locales `multilingual-e5-small` (acierta con paráfrasis), fusionados con *Reciprocal Rank Fusion*, que combina rankings sin calibrar puntajes. Solo los 8 mejores fragmentos
  (~4–5 mil tokens) llegan al modelo.

### 2. Los empleados no hablan como la ley → tres puentes de vocabulario

1. Glosario determinista (costo cero): "correr" → *despido, rescisión de la relación de trabajo*; "día festivo" → *descanso obligatorio*; "finiquito" → *partes proporcionales*; "home office" → *teletrabajo*, etc.
2. Reescritura con el LLM: Haiku convierte la pregunta en 2–4 consultas redactadas como la ley (sin responderla). "¿Me pueden correr por estar embarazada?" → "prohibición de despedir a trabajadora embarazada", "rescisión de la relación de trabajo por embarazo"…
3. Embeddings para la paráfrasis residual.

Todas las consultas se buscan y se fusionan. La ablación `v3_sin_reescritura` (sin 1 ni 2) permite medir cuánto aportan: `python -m evaluacion.run --versiones v1_articulo v3_sin_reescritura`. El reporte desglosa acierto@k por estilo (coloquial vs formal), que es justo la brecha de este reto.

### 3. Evitar que invente → instrucción + estructura + verificación externa al modelo

- Instrucción: responder solo con los fragmentos; prohibido usar montos (salario mínimo, UMA) u otras leyes (IMSS, ISR, INFONAVIT); estado explícito `respondida | parcial | sin_respuesta`.
- Salida estructurada (tool use forzado): el modelo debe declarar el estado y dar una cita literal por fundamento.
- Guardas que no dependen del modelo:
  1. Cada cita se busca literalmente en el fragmento (normalizando acentos/puntuación; tolerancia difusa 92%). Citas que no existen se descartan y se avisa.
  2. Si una respuesta afirmativa se queda sin ninguna cita verificable, se sustituye por el aviso de "no está en los documentos". Preferimos un falso "no sé" a una respuesta inventada.
  3. Cifras en la respuesta que no aparecen (en número o en palabras: "quince") en los artículos citados generan una advertencia visible ("verifícala"). Puede dar falsos positivos con cálculos legítimos (12+2+2+2 = 18).
- Cuando falta todo o parte, la respuesta lo dice explícitamente (✅ / ⚠️ parcial / ❌ no está en los documentos).
- El set incluye 9 preguntas sin respuesta (salario mínimo, UMA, ISR del aguinaldo, IMSS, Afore, INFONAVIT, permiso por luto, política interna, ley de EE. UU.) y 3 parciales.

### 4. Fidelidad (faithfulness) — cálculo propio, paso a paso (`evaluacion/metricas.py::fidelidad`)

1. Descomponer: el juez (Claude Sonnet 5.5, *otro* modelo que el asistente, para reducir autoindulgencia) divide la respuesta en afirmaciones atómicas y marca cada una como `factual` (dice algo de la ley o de un derecho) o `meta` ("consulta a RH", "esto no está en los documentos").
2. Verificar: para cada afirmación factual, el juez decide `respaldada / no_respaldada / contradicha` usando solo los 8 fragmentos que vio el asistente e indica cuáles la respaldan. Se aceptan paráfrasis y aritmética simple; agregar condiciones o cifras ajenas = no respaldada. Afirmación sin veredicto = no respaldada.
3. Calcular (en código, no el juez):
   - `fidelidad(respuesta) = respaldadas / factuales`; indefinida si no hay factuales (abstenciones).
   - Micro = Σ respaldadas / Σ factuales (cada afirmación pesa igual) y macro = promedio por respuesta.
   - Fidelidad estricta: además exige que al menos un fragmento que la respalda sea uno de los que el asistente citó al empleado (respaldo visible, no solo "estaba en el contexto").
4. Validar al juez: se toma una muestra estratificada de 40 afirmaciones (sobre-representa las que el juez marcó como no respaldadas), se califica a mano sin ver al juez, y se reportan acuerdo, kappa de Cohen, y precisión/recall del juez para detectar afirmaciones sin respaldo (`evaluacion/acuerdo_juez.py`). Criterio: kappa ≥ 0.6 para confiar en las cifras de fidelidad.

### 5. Más allá de la fidelidad

Una respuesta fiel puede ser irrelevante o apoyarse en el artículo equivocado. Por eso se mide, por pregunta:

| Dimensión | Métrica | Detecta |
|---|---|---|
| Recuperación | acierto@k, recall de artículos y de fracciones, MRR (vs. artículos esperados) | si el artículo correcto llegó al modelo |
| Corrección | cobertura de puntos clave de la referencia + bandera de contradicción | fiel pero equivocada |
| Relevancia | 1–5 → 0–1: ¿atiende lo preguntado? | fiel pero fuera de tema |
| Citas | precisión (artículo citado ∈ esperados) y citas inválidas | cita el fragmento equivocado |
| Abstención | abstención correcta en no-respondibles, invención en no-respondibles, abstención indebida | inventar vs. ser inútil |
| Operación | latencia p50/p95, tokens, costo USD por pregunta (asistente y juez por separado) | viabilidad |

Un criterio compuesto de éxito (binario, usado para McNemar): no respondible → se abstuvo; respondible → no se abstuvo, no contradice, cubre ≥50% de puntos clave y fidelidad ≥0.8.

Comparación de versiones (`evaluacion/estadistica.py`): pruebas pareadas sobre las mismas preguntas — bootstrap (10 000 remuestreos, IC 95%), Wilcoxon para métricas continuas y McNemar exacto para binarias. Solo se declara "diferencia real" si el IC excluye 0 y p < 0.05. Además la prueba A/A (`--aa`) corre v1 contra sí misma sin caché: si A/A muestra diferencias del mismo orden que A/B, lo de A/B es ruido. El reporte también calcula el efecto mínimo detectable con 63 preguntas (≈ 16 puntos en tasa de éxito): diferencias menores no son evidencia.

### 6. ¿Dónde y por qué falla?

Cada pregunta recibe todas las etiquetas de error que aplican y una principal según la cadena causal (`evaluacion/errores.py`): invención fuera de alcance, fallo de recuperación, abstención indebida, fiel a fragmento equivocado, contradice la referencia, afirmación no respaldada, incompleta, poco relevante, cita inválida. El reporte cuenta cada grupo por versión, explica su causa probable y muestra ejemplos. Termina con una recomendación basada en umbrales explícitos:

| Criterio | Umbral para ponerlo frente a empleados |
|---|---|
| Inventa en preguntas sin respuesta | ≤ 10% |
| Fidelidad micro | ≥ 95% |
| Contradice la referencia | ≤ 5% |
| Éxito compuesto | ≥ 85% |
| Artículo correcto recuperado | ≥ 90% |

Todo en verde → piloto con condiciones; 1–2 en rojo → solo herramienta interna de RH con revisión humana; más → no.

---

## Resultados

Los números los produce `python -m evaluacion.run --aa` en `resultados/reporte.md` (con gráficas `metricas_por_version.png` y `errores_por_tipo.png`, y el detalle en `metricas_por_pregunta.csv` y `afirmaciones_juez.csv`). No incluyo cifras aquí porque no las he medido: en el entorno donde construí el proyecto no había acceso a diputados.gob.mx ni una API key, así que el sistema se verificó con un LLM simulado y textos de prueba con el mismo formato que los PDF oficiales (ver "Estado de la verificación"). Al correrlo, pega aquí la tabla "Resumen por versión" y la sección de recomendación del reporte.

Costo esperado (estimación previa a medir; el reporte da el real): ~5 mil tokens de entrada y ~500 de salida por pregunta con Haiku 4.5 ($1/$5 por MTok) ⇒ ≈ US$0.007 por pregunta y 2–5 s; el juez con Sonnet 5.5 cuesta ≈ US$0.03 por respuesta evaluada.

### Mi expectativa de dónde fallará (hipótesis a confirmar con el reporte)
- Preguntas que requieren combinar artículos (finiquito P39, hostigamiento P40 que remite al art. 50, embarazo P31 entre arts. 133 y 170): riesgo de cobertura incompleta.
- Cálculos (P02, vacaciones con 4 años): errores aritméticos y advertencias de cifras por diseño.
- Trampas fuera de alcance que se parecen a temas de la ley (permiso por luto P60, IMSS P28): riesgo de invención.
- Reforma de 2026 de jornada (P19): depende de que los transitorios se segmenten bien.

---

## Supuestos

Acordados contigo antes de empezar:
- Proveedor: Anthropic. Asistente `claude-haiku-4-5` (rápido y barato); juez `claude-sonnet-5-5` (distinto y más capaz que el asistente). Precios de platform.claude.com (oct-2026), configurables en `laboral/config.py`.
- Embeddings locales multilingües (`intfloat/multilingual-e5-small`), gratis y reproducibles.
- ~60 preguntas (63: 51 respondibles, 3 parciales, 9 sin respuesta; 54 coloquiales, 9 formales).
- Entrega como archivos (.zip).

Tomados por mí:
- "Versión vigente" = la que el sitio publique al correr; se registra sha256 y fecha de "Última reforma DOF" en `data/raw/manifiesto.json` y en el reporte. Al construir esto, la LFT tenía última reforma DOF 14-05-2026 (incluye la reducción gradual de jornada publicada en mayo de 2026). `evaluacion/validar.py` comprueba que cada artículo esperado y cada frase ancla del set existan en el texto descargado y marca desajustes si la ley vuelve a cambiar.
- Fecha de referencia fija (2026-10-01) en la evaluación para que "este año" signifique lo mismo en cada corrida.
- De la Constitución se usa solo el artículo 123 (apartados A y B) y sus transitorios de reforma; de ambas, los transitorios desde 2012.
- Las respuestas de referencia y los artículos esperados los redacté yo a partir del texto de la ley; no soy abogado. Antes de usar los resultados para decidir, conviene que alguien de jurídico revise `data/preguntas.csv`.
- Un "sin respuesta" ante falta de evidencia verificable es preferible a una respuesta posiblemente inventada (por eso la guarda 2 puede aumentar la abstención indebida).
- La calificación manual para validar al juez debe hacerla una persona (tú o quien evalúe); dejé el flujo listo.
- Temperatura 0 + caché ⇒ reproducibilidad. Como la API puede no ser determinista aun con temperatura 0, el ruido se mide con la prueba A/A en vez de suponerlo nulo.

## Estado de la verificación

Verificado: parseo con textos que replican el formato de los PDF oficiales (encabezados, notas de reforma, `3o. Bis`, `39-A`, fracciones `Bis`, transitorios), recuperación, guardas de citas y cifras, métricas, pruebas estadísticas, generación del reporte y de la muestra de anotación, y el cálculo de acuerdo — todo de punta a punta con un LLM simulado (`tests/`). No verificado: el parseo sobre los 457 páginas reales y las llamadas reales a la API. Lo primero que hay que revisar al correr es `data/procesado/resumen_segmentacion.json` (debe reportar ~1,000+ artículos de la LFT, del 1 al 1010) y `resultados/validacion_dataset.json` (idealmente 0 problemas).

## Qué haría distinto con más tiempo

- Reranker (cross-encoder multilingüe o el LLM) sobre los ~30 candidatos fusionados, y seguir remisiones ("en los términos del artículo 50") agregando el artículo referido al contexto.
- Más y mejores preguntas: ~200, redactadas por empleados reales de RH, con referencias revisadas por abogados laboralistas, y preguntas multi-turno; con 63 preguntas solo se detectan diferencias grandes.
- Dos jueces o juez + muestra humana mayor (100+ afirmaciones, dos anotadores para medir el acuerdo humano-humano como techo).
- Evaluar `v4_sonnet` (otro modelo) y prompt caching para bajar costo/latencia.
- Detectar preguntas que requieren hechos del caso (¿hubo causa justificada?) y responder con preguntas de aclaración en lugar de una respuesta general.
- Monitoreo en producción: registro de preguntas reales, muestreo semanal para calificación humana, alertas cuando la ley se reforme (cambio de sha256 del PDF) para reindexar y revalidar el set.