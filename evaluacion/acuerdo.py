"""python -m evaluacion.acuerdo  -> recalcula el acuerdo juez-humano tras llenar anotacion_manual.csv."""
import json

from .acuerdo_juez import calcular_acuerdo

if __name__ == "__main__":
    res = calcular_acuerdo()
    if not res:
        print("No existe resultados/anotacion_manual.csv; corre primero python -m evaluacion.run")
    else:
        print(json.dumps({k: v for k, v in res.items() if k != "desacuerdos"}, indent=2, ensure_ascii=False, default=str))
