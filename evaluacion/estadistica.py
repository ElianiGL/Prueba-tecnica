#Comparación de dos versiones: ¿la diferencia es real o ruido?

from __future__ import annotations
import math
import numpy as np
from scipy import stats

def bootstrap_pareado(a: np.ndarray, b: np.ndarray, n: int = 10_000, semilla: int = 42) -> tuple[float, float, float]:
    d = b - a
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    medias = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))

def mcnemar_exacto(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = stats.binomtest(min(n01, n10), n, 0.5).pvalue
    return n01, n10, float(p)

def comparar(df, version_a: str, version_b: str, metricas_continuas: list[str], metricas_binarias: list[str]) -> list[dict]:
    filas = []
    a = df[df.version == version_a].set_index("id")
    b = df[df.version == version_b].set_index("id")
    for m in metricas_continuas + metricas_binarias:
        comunes = a.index.intersection(b.index)
        x, y = a.loc[comunes, m].astype(float), b.loc[comunes, m].astype(float)
        ok = ~(x.isna() | y.isna())
        x, y = x[ok].to_numpy(), y[ok].to_numpy()
        if len(x) < 5:
            continue
        dif, lo, hi = bootstrap_pareado(x, y)
        if m in metricas_binarias:
            n01, n10, p = mcnemar_exacto(x, y)
            prueba = f"McNemar exacto (solo {version_a}: {n01}, solo {version_b}: {n10})"
        else:
            if np.allclose(x, y):
                p, prueba = 1.0, "Wilcoxon (sin diferencias)"
            else:
                p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
                prueba = "Wilcoxon pareado"
        real = (lo > 0 or hi < 0) and p < 0.05
        filas.append({
            "metrica": m, "n": len(x), version_a: x.mean(), version_b: y.mean(),
            "diferencia": dif, "ic95_inf": lo, "ic95_sup": hi, "p": p, "prueba": prueba,
            "veredicto": "diferencia real" if real else "no distinguible de ruido",
        })
    return filas

def efecto_minimo_detectable(n: int, p_base: float = 0.8, alfa: float = 0.05, potencia: float = 0.8,
                             discordancia: float = 0.2) -> float:
    za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(potencia)
    return float((za + zb) * math.sqrt(discordancia / n))