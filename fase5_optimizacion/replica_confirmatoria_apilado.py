# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-002: REPLICACION CONFIRMATORIA DE LA VARIANTE V2 (peso 0.5)
==============================================================================
Motivo. En analisis_apilado_transiciones.py el candidato declarado ANTES de ver
resultados (V1, historia con peso 1.0) no cumplio la regla de adopcion. La variante
V2 (historia con peso 0.5) se veia mejor, pero se miro DESPUES de ver los numeros de
2024->2025. Adoptarla directamente seria elegir al ganador a posteriori. La forma
honesta de resolverlo es replicar el experimento donde el peso 0.5 NO se eligio.

Diseno. Panel objetivo = transicion 2023->2024 (344 positivos). Historia = transiciones
2021->2022 y 2022->2023 (684 positivos), que terminan antes de 2024: ninguna fila del
periodo evaluado entra al entrenamiento. Mismo ensamble, mismos hiperparametros, mismos
pliegues (StratifiedKFold 5, shuffle, semilla 42) y mismo bootstrap pareado.

Regla de confirmacion (declarada ANTES de ejecutar)
---------------------------------------------------
V2 se considera CONFIRMADA solo si, en esta replicacion:
  C1  OOF: PR-AUC(V2) > PR-AUC(base) y ROC-AUC(V2) >= ROC-AUC(base) - 0.002.
  C2  Zona roja OOF: TP(top 1%) y TP(top 2.5%) de V2 no son menores que los de la base y
      al menos uno es estrictamente mayor.
  C3  Version agrupada (V2g, sin los NIT validados en la historia): PR-AUC(V2g) > base.
V2 se adopta por defecto en produccion solo si C1, C2 y C3 se cumplen Y, ademas, V2 ya
cumplio G1 a G4 en 2024->2025 (ver analisis_apilado_transiciones.json). Si falla algo,
no se aplica.

Notas tecnicas: ``Series.isin`` filtra los NIT validados; ``fit(sample_weight=...)`` pondera
las filas historicas; el bootstrap pareado reutiliza ``bootstrap_pareado`` (mismo remuestreo
para base y variante, con reemplazo, semilla fija).
"""

import json
import time

import joblib

import analisis_apilado_transiciones as A

base = A.base
RUTA_SALIDA = base.SALIDAS_DIR / ("confirmacion_v2_smoke.json" if A.SMOKE else "confirmacion_v2.json")


def main():
    t0 = time.time()
    print("=" * 100)
    print("REPLICACION CONFIRMATORIA: panel objetivo 2023->2024, historia 2021->22 y 2022->23")
    print("=" * 100)
    assert base.USAR_FEATURES_MULTIANIO is False

    # Columnas del PISO (tambien re-valida las aserciones de integridad del panel).
    columnas = list(base.construir_panel_y_splits()[0].columns)
    trans = {a: A.construir_transicion(a, columnas) for a in (2021, 2022, 2023)}
    hist = A.apilar(trans, (2021, 2022))
    Xp, yp, nit_p = trans[2023]["X"], trans[2023]["y"], trans[2023]["nit"]
    yp_np = yp.to_numpy()
    print(f"  Panel objetivo 2023->2024: {len(Xp):,} empresas, {int(yp.sum())} positivos | "
          f"historia: {len(hist['X']):,} filas, {int(hist['y'].sum())} positivos")
    assert 2024 not in set(hist["anio"]) and max(hist["anio"]) == 2022, "La historia no debe incluir 2023."

    cache_path = base.SALIDAS_DIR / ("cache_confirmacion_smoke.pkl" if A.SMOKE else "cache_confirmacion.pkl")
    cache = joblib.load(cache_path) if cache_path.exists() else {}

    def correr(clave, h, pw, agr):
        if clave not in cache:
            ti = time.time()
            cache[clave] = A.oof_panel(Xp, yp, nit_p, h, pw, agr)
            joblib.dump(cache, cache_path)
            print(f"    [{clave}] {time.time() - ti:.0f}s", flush=True)
        return cache[clave]

    variantes = {"base": (None, 1.0, False), "V1_peso1.0": (hist, 1.0, False),
                 "V2_peso0.5": (hist, 0.5, False), "V2g_peso0.5_agrupado": (hist, 0.5, True)}
    oof, res = {}, {"panel_objetivo": "2023->2024", "positivos_panel": int(yp.sum()),
                    "positivos_historia": int(hist["y"].sum()), "metricas": {}}
    for nombre, (h, pw, agr) in variantes.items():
        oof[nombre] = correr(nombre, h, pw, agr)
        m = A.metricas(yp_np, oof[nombre])
        res["metricas"][nombre] = m
        c = m["cortes"]
        print(f"    {nombre:22s} ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | F1max {m['f1_max']['f1']:.4f} | "
              f"top1%: {c['1.0']['tp']}/{c['1.0']['alertas']} | top2.5%: {c['2.5']['tp']}/{c['2.5']['alertas']} | "
              f"top10%: {c['10.0']['tp']}/{c['10.0']['alertas']}", flush=True)

    res["bootstrap_vs_base"] = {n: A.bootstrap_pareado(yp_np, oof["base"], oof[n])
                                for n in ("V1_peso1.0", "V2_peso0.5", "V2g_peso0.5_agrupado")}

    b, v2, v2g = (res["metricas"][k] for k in ("base", "V2_peso0.5", "V2g_peso0.5_agrupado"))
    tp1_b, tp25_b = b["cortes"]["1.0"]["tp"], b["cortes"]["2.5"]["tp"]
    tp1, tp25 = v2["cortes"]["1.0"]["tp"], v2["cortes"]["2.5"]["tp"]
    reglas = {
        "C1_pr_mejora_y_roc_no_cae": bool(v2["pr_auc"] > b["pr_auc"] and v2["roc_auc"] >= b["roc_auc"] - 0.002),
        "C2_zona_roja_no_empeora_y_mejora": bool(tp1 >= tp1_b and tp25 >= tp25_b and (tp1 > tp1_b or tp25 > tp25_b)),
        "C3_agrupado_supera_base": bool(v2g["pr_auc"] > b["pr_auc"]),
    }
    previo = json.load(open(base.SALIDAS_DIR / "analisis_apilado_transiciones.json", encoding="utf-8")) \
        if (base.SALIDAS_DIR / "analisis_apilado_transiciones.json").exists() else None
    res["reglas_confirmacion"] = reglas
    res["v2_confirmada"] = bool(all(reglas.values()))
    res["valores"] = {"pr_base_v2_v2g": [b["pr_auc"], v2["pr_auc"], v2g["pr_auc"]],
                      "roc_base_v2_v2g": [b["roc_auc"], v2["roc_auc"], v2g["roc_auc"]],
                      "tp_top1_base_v2": [tp1_b, tp1], "tp_top2_5_base_v2": [tp25_b, tp25]}
    print("\n" + "=" * 100 + "\nCONFIRMACION")
    for r, ok in reglas.items():
        print(f"  {'CUMPLE ' if ok else 'FALLA  '} {r}")
    print(f"  ==> V2 CONFIRMADA: {res['v2_confirmada']}")
    res["segundos"] = round(time.time() - t0, 1)
    with open(RUTA_SALIDA, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado {RUTA_SALIDA.name} ({res['segundos']} s)")


if __name__ == "__main__":
    main()
