#Extracción y partición de la LFT y del artículo 123 constitucional.

from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from .config import DIR_PROC, DIR_RAW

def pdf_a_texto(ruta: Path) -> str:
    from pypdf import PdfReader
    lector = PdfReader(str(ruta))
    paginas = [(p.extract_text() or "") for p in lector.pages]
    texto = "\n".join(paginas)
    if len(texto.strip()) < 1000:
        import pdfplumber
        with pdfplumber.open(str(ruta)) as pdf:
            texto = "\n".join((p.extract_text() or "") for p in pdf.pages)
    return texto

_ENCABEZADOS = [
    r"LEY FEDERAL DEL TRABAJO",
    r"CONSTITUCI[ÓO]N POL[ÍI]TICA DE LOS ESTADOS UNIDOS MEXICANOS",
    r"C[ÁA]MARA DE DIPUTADOS DEL H\. CONGRESO DE LA UNI[ÓO]N",
    r"Secretar[íi]a General",
    r"Secretar[íi]a de Servicios Parlamentarios",
    r"[ÚU]ltima [Rr]eforma (?:publicada )?DOF \d{2}-\d{2}-\d{4}",
    r"\d{1,4} de \d{1,4}",
]
_RE_ENCABEZADO = re.compile(r"(?m)^[ \t]*(?:" + "|".join(_ENCABEZADOS) + r")[ \t]*$")
_PALABRA_NOTA = r"(?:reformad[oa]s?|adicionad[oa]s?|derogad[oa]s?|recorrid[oa]s?|recorre|reubicad[oa]s?|Fe de erratas|Aclaraci[óo]n)"
_FECHA = r"\d{2}-\d{2}-\d{4}"
_RE_NOTA = re.compile(
    rf"(?m)^[ \t]*[^\n]{{0,80}}?{_PALABRA_NOTA}[^\n]*?DOF[^\n]*$(?:\n[ \t]*(?:{_FECHA}[ ,y]*)+[ \t]*$)*")

def limpiar(texto: str) -> str:
    texto = texto.replace("\r", "").replace(" ", " ").replace("–", "-").replace("—", "-")
    texto = _RE_ENCABEZADO.sub("", texto)
    texto = _RE_NOTA.sub("", texto)
    texto = re.sub(r"[ \t]+\n", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto

_SUFIJOS = r"Bis|Ter|Qu[áa]ter|Quintus|Quinquies|Sexies|Septies|Octies|Nonies"
RE_ARTICULO = re.compile(
    rf"(?m)^[ \t]*Art[íi]culo[ \t]+(\d{{1,4}})[ \t]*(?:o\.|º\.?|°\.?)?[ \t]*"
    rf"(?:-[ \t]*([A-Z])(?![a-záéíóúñ])|[ \t]*({_SUFIJOS})\b)?[ \t]*(?:({_SUFIJOS})\b)?"
    rf"[ \t]*(?:\.[ \t]*-|\.|-)[ \t]*")
RE_TITULO = re.compile(r"(?m)^[ \t]*(T[ÍI]TULO\s+[A-ZÁÉÍÓÚ ]+?)[ \t]*$")
RE_CAPITULO = re.compile(rf"(?m)^[ \t]*(CAP[ÍI]TULO\s+(?:[IVXLC]+|[ÚU]NICO)(?:\s+(?:{_SUFIJOS}))?)[ \t]*$")
RE_TRANSITORIOS = re.compile(r"(?m)^[ \t]*(?:ART[ÍI]CULOS\s+)?TRANSITORIOS?\b")
_ROMANO = r"(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
RE_FRACCION = re.compile(
    rf"(?m)^[ \t]*((?=[IVXLC]){_ROMANO}(?:[ \t]+(?:{_SUFIJOS}))?)\.[ \t]*-?[ \t]*(?=\S)")

def _romano_a_int(s: str) -> int:
    valores = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for c in reversed(s):
        v = valores[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total

def _normaliza_lineas(texto: str) -> str:
    lineas = [l.strip() for l in texto.split("\n")]
    salida: list[str] = []
    for linea in lineas:
        if not linea:
            if salida and salida[-1] != "":
                salida.append("")
            continue
        estructural = RE_FRACCION.match(linea) or re.match(r"^[a-zñ]\)\s", linea) or re.match(r"^[A-Z]\.\s", linea)
        if salida and salida[-1] and not estructural and not re.search(r"[.:;]$", salida[-1]):
            salida[-1] = salida[-1] + (" " if not salida[-1].endswith("-") else "") + linea
        else:
            salida.append(linea)
    texto = "\n".join(salida)
    return re.sub(r"\n{2,}", "\n\n", texto).strip()

@dataclass
class Fraccion:
    fraccion: str
    texto: str

@dataclass
class Articulo:
    documento: str
    articulo: str
    titulo: str
    capitulo: str
    encabezado: str
    fracciones: list[Fraccion]
    texto: str
    apartado: str = ""

def _id_articulo(m: re.Match) -> str:
    num, letra, suf1, suf2 = m.group(1), m.group(2), m.group(3), m.group(4)
    ident = num
    if letra:
        ident += f"-{letra}"
    for s in (suf1, suf2):
        if s:
            ident += f" {s.capitalize().replace('Quater', 'Quáter')}"
    return ident

def _num(ident: str) -> int:
    return int(re.match(r"\d+", ident).group())

def _separar_fracciones(cuerpo: str) -> tuple[str, list[Fraccion]]:
    candidatos = list(RE_FRACCION.finditer(cuerpo))
    aceptados = []
    ultimo = 0
    for m in candidatos:
        rom = m.group(1).split()[0]
        v = _romano_a_int(rom)
        es_bis = len(m.group(1).split()) > 1
        if (ultimo < v <= ultimo + 10 and not es_bis) or (es_bis and v == ultimo and ultimo > 0):
            aceptados.append(m)
            ultimo = v
    if not aceptados:
        return cuerpo.strip(), []
    encabezado = cuerpo[: aceptados[0].start()].strip()
    fracs = []
    for i, m in enumerate(aceptados):
        fin = aceptados[i + 1].start() if i + 1 < len(aceptados) else len(cuerpo)
        fracs.append(Fraccion(m.group(1).strip(), cuerpo[m.start():fin].strip()))
    return encabezado, fracs

def _cap(encabezado: str) -> str:
    partes = encabezado.split()
    out = [partes[0].capitalize()]
    for p in partes[1:]:
        out.append(p if re.fullmatch(r"[IVXLC]+", p) else p.capitalize())
    return " ".join(out)

def _nombre_siguiente(texto: str, pos: int) -> str:
    resto = texto[pos:].lstrip("\n ").split("\n")
    return resto[0].strip() if resto else ""

def segmentar_lft(texto: str) -> tuple[list[Articulo], str]:
    texto = limpiar(texto)
    candidatos = list(RE_ARTICULO.finditer(texto))
    aceptados: list[re.Match] = []
    ultimo = 0
    for m in candidatos:
        n = int(m.group(1))
        if not aceptados and n != 1:
            continue
        if ultimo <= n <= ultimo + 150:
            aceptados.append(m)
            ultimo = n
    articulos: list[Articulo] = []
    titulo = capitulo = ""
    transitorios = ""
    if aceptados:
        pre = texto[: aceptados[0].start()]
        for rx in (RE_TITULO, RE_CAPITULO):
            for x in rx.finditer(pre):
                nombre = _nombre_siguiente(pre, x.end())
                if rx is RE_TITULO:
                    titulo = f"{_cap(x.group(1))} - {nombre}"
                else:
                    capitulo = f"{_cap(x.group(1))} - {nombre}"
    for i, m in enumerate(aceptados):
        fin = aceptados[i + 1].start() if i + 1 < len(aceptados) else len(texto)
        bloque = texto[m.end():fin]
        cortes = []
        for rx in (RE_TITULO, RE_CAPITULO):
            cortes += [(x.start(), x) for x in rx.finditer(bloque)]
        t = RE_TRANSITORIOS.search(bloque) if i == len(aceptados) - 1 else None
        if t:
            transitorios = bloque[t.start():]
            bloque = bloque[: t.start()]
            cortes = [c for c in cortes if c[0] < t.start()]
        cortes.sort(key=lambda c: c[0])
        cuerpo = bloque[: cortes[0][0]] if cortes else bloque
        cuerpo = _normaliza_lineas(cuerpo)
        encabezado, fracs = _separar_fracciones(cuerpo)
        ident = _id_articulo(m)
        articulos.append(Articulo("LFT", ident, titulo, capitulo, encabezado, fracs, cuerpo))
        for pos, x in cortes:
            nombre = _nombre_siguiente(bloque, x.end())
            if x.re is RE_TITULO:
                titulo, capitulo = f"{_cap(x.group(1))} - {nombre}", ""
            else:
                capitulo = f"{_cap(x.group(1))} - {nombre}"
    return articulos, transitorios

def extraer_art123(texto: str) -> tuple[list[Articulo], str]:
    texto = limpiar(texto)
    inicio = re.search(r"(?m)^[ \t]*Art[íi]culo[ \t]+123[ \t]*\.", texto)
    fin = re.search(r"(?m)^[ \t]*(?:T[ÍI]TULO S[ÉE]PTIMO|Art[íi]culo[ \t]+124[ \t]*\.)", texto[inicio.end():]) if inicio else None
    if not inicio:
        raise ValueError("No se encontró el artículo 123 en el texto de la CPEUM.")
    cuerpo = texto[inicio.end(): inicio.end() + (fin.start() if fin else 200_000)]
    cuerpo = _normaliza_lineas(cuerpo)
    m_a = re.search(r"(?m)^[ \t]*A\.[ \t]+Entre", cuerpo)
    m_b = re.search(r"(?m)^[ \t]*B\.[ \t]+Entre", cuerpo)
    articulos = []
    intro = cuerpo[: m_a.start()].strip() if m_a else ""
    if intro:
        articulos.append(Articulo("CPEUM", "123", "Título Sexto - Del Trabajo y de la Previsión Social", "",
                                  intro, [], intro, apartado=""))
    for letra, ma, mb in (("A", m_a, m_b), ("B", m_b, None)):
        if not ma:
            continue
        seg = cuerpo[ma.start(): mb.start() if mb else len(cuerpo)].strip()
        enc, fracs = _separar_fracciones(seg)
        articulos.append(Articulo("CPEUM", "123", "Título Sexto - Del Trabajo y de la Previsión Social",
                                  f"Apartado {letra}", enc, fracs, seg, apartado=letra))
    trans = ""
    m_tr = re.search(r"(?m)^[ \t]*ART[ÍI]CULOS TRANSITORIOS DE DECRETOS DE REFORMA", texto)
    if m_tr:
        trans = texto[m_tr.start():]
    return articulos, trans

_MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"

def segmentar_transitorios(texto: str, documento: str, filtro_titulo: str | None = None) -> list[dict]:
    bloques = re.split(r"(?m)^[ \t]*(?=DECRETO\b)", texto)
    salida = []
    for b in bloques:
        if not b.strip().startswith("DECRETO"):
            continue
        titulo = b.strip().split("\n\n")[0].replace("\n", " ")[:400]
        if filtro_titulo and not re.search(filtro_titulo, titulo):
            continue
        f = re.search(rf"(\d{{1,2}})\s+de\s+({_MESES})\s+de\s+(\d{{4}})", b[:1500], re.I)
        fecha = f.group(0) if f else ""
        anio = int(f.group(3)) if f else 0
        salida.append({"documento": documento, "fecha": fecha, "anio": anio, "titulo": titulo,
                       "texto": _normaliza_lineas(b)})
    return salida

@dataclass
class Fragmento:
    id: str
    documento: str
    articulo: str
    apartado: str
    fracciones: list[str]
    tipo: str
    contexto: str
    etiqueta: str
    texto: str
    claves: list[str] = field(default_factory=list)

    @property
    def texto_indexado(self) -> str:
        return f"{self.etiqueta}. {self.contexto}\n{self.texto}"
    
    def a_dict(self) -> dict:
        return asdict(self)

def _clave_art(a: Articulo) -> str:
    return f"CPEUM:123:{a.apartado}" if a.documento == "CPEUM" and a.apartado else f"{a.documento}:{a.articulo}"

def _etiqueta(a: Articulo, fracs: list[str]) -> str:
    base = "CPEUM, art. 123" + (f", apartado {a.apartado}" if a.apartado else "") if a.documento == "CPEUM" \
        else f"LFT, art. {a.articulo}"
    if fracs:
        base += ", fr. " + (fracs[0] if len(fracs) == 1 else f"{fracs[0]} a {fracs[-1]}")
    return base

def _contexto(a: Articulo) -> str:
    return " / ".join(x for x in (a.titulo, a.capitulo) if x)

def fragmentar(articulos: list[Articulo], transitorios: list[dict], estrategia: str, max_chars: int) -> list[Fragmento]:
    frags: list[Fragmento] = []
    def nuevo(a: Articulo, grupo: list[Fraccion], incluir_encabezado: bool):
        nombres = [g.fraccion for g in grupo]
        partes = []
        if a.encabezado and (incluir_encabezado or True):
            partes.append(a.encabezado if incluir_encabezado else f"[Encabezado del artículo] {a.encabezado[:400]}")
        partes += [g.texto for g in grupo]
        clave = _clave_art(a)
        claves = [clave] + [f"{clave}:{n}" for n in nombres]
        if a.documento == "CPEUM":
            claves.append("CPEUM:123")
        claves = list(dict.fromkeys(claves))
        idf = f"{clave}" + (f"#{nombres[0]}-{nombres[-1]}" if nombres else "") + f"#{len(frags)}"
        frags.append(Fragmento(idf, a.documento, a.articulo, a.apartado, nombres, "articulo", _contexto(a),
                               _etiqueta(a, nombres), "\n".join(partes).strip(), claves))
    for a in articulos:
        if not a.fracciones:
            if len(a.texto) <= max_chars:
                nuevo(a, [], True)
            else:
                parrafos, actual = a.texto.split("\n"), ""
                for p in parrafos:
                    if actual and len(actual) + len(p) > max_chars:
                        nuevo(Articulo(**{**asdict(a), "encabezado": actual, "fracciones": []}), [], True)
                        actual = ""
                    actual += ("\n" if actual else "") + p
                if actual:
                    nuevo(Articulo(**{**asdict(a), "encabezado": actual, "fracciones": []}), [], True)
            continue
        if estrategia == "fraccion":
            for i, fr in enumerate(a.fracciones):
                nuevo(a, [fr], incluir_encabezado=(i == 0 and len(a.encabezado) < 600))
        elif estrategia == "articulo":
            if len(a.texto) <= max_chars:
                nuevo(a, a.fracciones, True)
                continue
            grupo: list[Fraccion] = []
            tam = len(a.encabezado)
            primero = True
            for fr in a.fracciones:
                if grupo and tam + len(fr.texto) > max_chars:
                    nuevo(a, grupo, primero)
                    primero, grupo, tam = False, [], min(len(a.encabezado), 400)
                grupo.append(fr)
                tam += len(fr.texto)
            if grupo:
                nuevo(a, grupo, primero)
        else:
            raise ValueError(f"Estrategia desconocida: {estrategia}")
    for t in transitorios:
        texto = t["texto"]
        piezas = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)] or [texto]
        for j, p in enumerate(piezas):
            clave = f"{t['documento']}:TRANS:{t['anio']}"
            frags.append(Fragmento(f"{clave}#{len(frags)}", t["documento"], "Transitorios", "", [], "transitorio",
                                   t["titulo"][:300],
                                   f"{t['documento']}, transitorios del decreto publicado el {t['fecha']}",
                                   p, [clave]))
    return frags

def construir_corpus(estrategia: str = "articulo", max_chars: int = 2200, anio_min_transitorios: int = 2012,
                     forzar: bool = False) -> list[Fragmento]:
    from .descarga import registrar_ultima_reforma
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    salida = DIR_PROC / f"fragmentos_{estrategia}_{max_chars}.jsonl"
    if salida.exists() and not forzar:
        return [Fragmento(**json.loads(l)) for l in salida.read_text(encoding="utf-8").splitlines()]
    textos = {}
    for clave in ("LFT", "CPEUM"):
        cache_txt = DIR_PROC / f"{clave}.txt"
        if cache_txt.exists() and not forzar:
            textos[clave] = cache_txt.read_text(encoding="utf-8")
        else:
            pdf = DIR_RAW / f"{clave}.pdf"
            if not pdf.exists():
                raise FileNotFoundError(f"Falta {pdf}. Ejecuta laboral.descarga.descargar() primero.")
            textos[clave] = pdf_a_texto(pdf)
            cache_txt.write_text(textos[clave], encoding="utf-8")
        registrar_ultima_reforma(clave, textos[clave])
    arts_lft, trans_lft = segmentar_lft(textos["LFT"])
    arts_123, trans_cpeum = extraer_art123(textos["CPEUM"])
    trans = [t for t in segmentar_transitorios(trans_lft, "LFT") if t["anio"] >= anio_min_transitorios]
    trans += [t for t in segmentar_transitorios(trans_cpeum, "CPEUM", filtro_titulo=r"\b123\b")
              if t["anio"] >= anio_min_transitorios]
    frags = fragmentar(arts_lft + arts_123, trans, estrategia, max_chars)
    with open(salida, "w", encoding="utf-8") as f:
        for fr in frags:
            f.write(json.dumps(fr.a_dict(), ensure_ascii=False) + "\n")
    (DIR_PROC / "resumen_segmentacion.json").write_text(json.dumps({
        "articulos_lft": len(arts_lft),
        "primer_articulo": arts_lft[0].articulo if arts_lft else None,
        "ultimo_articulo": arts_lft[-1].articulo if arts_lft else None,
        "bloques_art123": len(arts_123),
        "decretos_transitorios": len(trans),
        f"fragmentos_{estrategia}": len(frags),
    }, indent=2, ensure_ascii=False))
    return frags

def quitar_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")