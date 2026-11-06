# -*- coding: utf-8 -*-
"""
FASE 5 - ABLACION Y NO-REGRESION DE FEATURES PLURIANUALES (FEAT-005)
==============================================================================
Objetivo cientifico
--------------------
Cuantificar el aporte marginal del bloque de tendencias PLURIANUALES (ventana
2020..2024, construido en features_multianio.py) frente al PISO del modelo, SIN
tocar la definicion del panel ni de la variable objetivo ``y`` del pipeline.

Metodo (identico en espiritu a analisis_tendencias_y_seleccion.py):
  * Se REUTILIZAN por import las funciones ya validadas del pipeline activo
    (construir_panel_y_splits, entrenar_escuadron_elite, calcular_pesos_ensamble).
    El import no dispara el pipeline por el guard ``if __name__ == '__main__'``.
  * Se mide en el MISMO split estratificado (random_state=42) y con el MISMO
    escuadron/calibracion dos escenarios:
      (a) SIN features plurianuales  -> es el PISO (bandera en False).
      (b) CON features plurianuales  -> bandera en True.
  * Se reportan ROC-AUC, PR-AUC y F1 optimo en el Test reservado, mas la OOF del
    panel completo en ambos escenarios, y el delta CON - SIN.
  * Se persiste salidas/analisis_multianio.json con la comparacion y la DECISION
    de adopcion segun el criterio de no-regresion del plan.

Criterio de adopcion (autoritativo, FEAT-005)
---------------------------------------------
Adoptar las features plurianuales como DEFAULT (USAR_FEATURES_MULTIANIO=True) SOLO
si NO regresan el PISO en el Test reservado:
    PR-AUC_test >= 0.3115 - 0.005  (= 0.3065)   Y
    F1_test     >= 0.4144 - 0.005  (= 0.4094)
Si regresan, se deja la bandera en False por defecto y se documenta la comparacion.

Anti-fuga
---------
Se reafirma que ninguna columna de X contiene '2025' en ambos escenarios; las
features plurianuales usan exclusivamente anios <= 2024 (ver
features_multianio.validar_sin_fuga_multianio).

Documentacion tecnica citada:
  - sklearn.metrics.{roc_auc_score, average_precision_score, precision_recall_curve}
  - sklearn.model_selection.{StratifiedKFold, cross_val_predict}
"""

import json

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import script_cinco_optimizado as base
from script_cinco_optimizado import (
    RANDOM_STATE,
    SALIDAS_DIR,
    calcular_pesos_ensamble,
    construir_panel_y_splits,
    entrenar_escuadron_elite,
)

# PISO INTOCABLE del Test reservado (commit-ready HEAD + FEAT-001..003).
PISO_TEST = {"roc_auc": 0.8782, "pr_auc": 0.3115, "f1_optimo": 0.4144}
# Tolerancia de no-regresion (ruido de muestreo) permitida respecto al PISO.
TOLERANCIA_NO_REGRESION = 0.005


def _metricas_test(y_test, score_test) -> dict:
    """
    ROC-AUC, PR-AUC y F1 optimo (con umbral) de un vector de scores frente a y_test.

    precision_recall_curve devuelve precision/recall por umbral; el F1 por umbral es
    su media armonica y se toma el maximo (mismo criterio que el pipeline).
    """
    roc = float(roc_auc_score(y_test, score_test))
    pr = float(average_precision_score(y_test, score_test))
    precisions, recalls, thresholds = precision_recall_curve(y_test, score_test)
    f1 = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = int(np.argmax(f1))
    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "f1_optimo": round(float(f1[idx]), 4),
        "umbral_f1_optimo": round(float(thresholds[idx]), 4),
        "precision_en_f1": round(float(precisions[idx]), 4),
        "recall_en_f1": round(float(recalls[idx]), 4),
    }


def _oof_panel_completo(X, y) -> dict:
    """
    Metricas Out-Of-Fold del super-ensamble sobre el panel completo con
    StratifiedKFold(5) + cross_val_predict(method='predict_proba'), replicando la
    estrategia de calibracion del pipeline. Los pesos del ensamble se calculan con
    calcular_pesos_ensamble sobre los modelos disponibles.
    """
    modelos, _, _ = entrenar_escuadron_elite(X, y, X, y)
    pesos = calcular_pesos_ensamble(modelos.keys())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score_oof = np.zeros(len(y), dtype=float)
    for nombre, modelo in modelos.items():
        oof = cross_val_predict(clone(modelo), X, y, cv=skf, method="predict_proba")[:, 1]
        score_oof += pesos[nombre] * oof
    return {
        "roc_auc": round(float(roc_auc_score(y, score_oof)), 4),
        "pr_auc": round(float(average_precision_score(y, score_oof)), 4),
    }


def _escenario(nombre: str, usar_multianio: bool) -> dict:
    """
    Construye el panel y mide un escenario completo (Test + OOF) fijando la bandera
    base.USAR_FEATURES_MULTIANIO. Se reconstruye el panel DENTRO del escenario para
    que la bandera surta efecto en construir_matriz_caracteristicas.
    """
    base.USAR_FEATURES_MULTIANIO = usar_multianio
    print(f"\n{'='*90}\n[ESCENARIO] {nombre} (USAR_FEATURES_MULTIANIO={usar_multianio})\n{'='*90}")
    X, y, X_train, X_test, y_train, y_test, _, _ = construir_panel_y_splits()

    # Anti-fuga explicito en ambos escenarios.
    assert not any("2025" in str(c) for c in X.columns), "ALERTA: fuga de 2025 en X."
    print(f"Panel: {X.shape[0]} empresas x {X.shape[1]} features | quiebras y={int(y.sum())}")

    _, score_test, _ = entrenar_escuadron_elite(X_train, y_train, X_test, y_test)
    met_test = _metricas_test(y_test, score_test)
    oof = _oof_panel_completo(X, y)

    return {
        "n_columnas": int(X.shape[1]),
        "columnas": list(X.columns),
        "test": met_test,
        "oof_panel_completo": oof,
    }


def main():
    print("=" * 90)
    print("FEAT-005: ABLACION DE FEATURES PLURIANUALES 2020..2024 (CON vs SIN)")
    print("=" * 90)

    # Guardamos el valor de produccion para restaurarlo al final (buen ciudadano).
    valor_produccion = base.USAR_FEATURES_MULTIANIO

    # (a) SIN features plurianuales = PISO.
    sin = _escenario("SIN features plurianuales (PISO)", usar_multianio=False)
    # (b) CON features plurianuales.
    con = _escenario("CON features plurianuales", usar_multianio=True)

    # Restauramos el valor de produccion de la bandera.
    base.USAR_FEATURES_MULTIANIO = valor_produccion

    # Columnas plurianuales efectivamente anadidas (diferencia de conjuntos).
    cols_nuevas = [c for c in con["columnas"] if c not in set(sin["columnas"])]

    deltas_test = {
        "roc_auc": round(con["test"]["roc_auc"] - sin["test"]["roc_auc"], 4),
        "pr_auc": round(con["test"]["pr_auc"] - sin["test"]["pr_auc"], 4),
        "f1_optimo": round(con["test"]["f1_optimo"] - sin["test"]["f1_optimo"], 4),
    }
    deltas_oof = {
        "roc_auc": round(con["oof_panel_completo"]["roc_auc"] - sin["oof_panel_completo"]["roc_auc"], 4),
        "pr_auc": round(con["oof_panel_completo"]["pr_auc"] - sin["oof_panel_completo"]["pr_auc"], 4),
    }

    # Criterio de no-regresion sobre el PISO (NO sobre el escenario SIN medido ahora,
    # que podria diferir por ruido; el PISO es el valor comprometido en reporte_fase5).
    umbral_pr = PISO_TEST["pr_auc"] - TOLERANCIA_NO_REGRESION
    umbral_f1 = PISO_TEST["f1_optimo"] - TOLERANCIA_NO_REGRESION
    no_regresa = (con["test"]["pr_auc"] >= umbral_pr) and (con["test"]["f1_optimo"] >= umbral_f1)
    mejora = (con["test"]["pr_auc"] > PISO_TEST["pr_auc"]) or (con["test"]["f1_optimo"] > PISO_TEST["f1_optimo"])

    decision = {
        "criterio": (
            f"adoptar DEFAULT (True) solo si PR-AUC_test >= {umbral_pr:.4f} y "
            f"F1_test >= {umbral_f1:.4f} (PISO +/- {TOLERANCIA_NO_REGRESION})"
        ),
        "pr_auc_test_con": con["test"]["pr_auc"],
        "f1_test_con": con["test"]["f1_optimo"],
        "no_regresa_piso": bool(no_regresa),
        "mejora_piso": bool(mejora),
        "usar_features_multianio_recomendado": bool(no_regresa),
    }

    resultado = {
        "piso_test_referencia": PISO_TEST,
        "tolerancia_no_regresion": TOLERANCIA_NO_REGRESION,
        "columnas_plurianuales_anadidas": cols_nuevas,
        "n_columnas_sin": sin["n_columnas"],
        "n_columnas_con": con["n_columnas"],
        "test_sin_multianio": sin["test"],
        "test_con_multianio": con["test"],
        "delta_test_con_menos_sin": deltas_test,
        "oof_sin_multianio": sin["oof_panel_completo"],
        "oof_con_multianio": con["oof_panel_completo"],
        "delta_oof_con_menos_sin": deltas_oof,
        "decision_adopcion": decision,
        "estrategia_faltantes": (
            "Imputacion-neutral + flags de cobertura (Anios_Historia_Disponible, "
            "Flag_Sin_Historia_3A, Flag_Historia_Parcial). Conserva el assert de "
            "cero-NaN en X y mantiene operativo a RandomForest (que no acepta NaN). "
            "Las pendientes/variaciones/CAGR/aceleraciones toman 0.0 (variacion nula) "
            "cuando falta el saldo historico; los conteos de perdida cuentan solo los "
            "anios realmente observados. El requisito de entrada parcial se cumple via "
            "las banderas de cobertura."
        ),
        "anti_fuga": (
            "Ventana plurianual restringida a anios <= anio_t (2024 para el modelo "
            "2024->2025, 2025 como anio_t solo en la proyeccion pero usando historia "
            "<=2024). Ninguna columna de X contiene '2025'. y se deriva de columnas "
            "*_2025 que NO entran en X. validar_sin_fuga_multianio asevera anio <= anio_t."
        ),
    }

    ruta = SALIDAS_DIR / "analisis_multianio.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("RESULTADO DE LA ABLACION PLURIANUAL (Test reservado)")
    print("=" * 90)
    print(f"  SIN multi-anio -> ROC {sin['test']['roc_auc']} | PR {sin['test']['pr_auc']} | F1 {sin['test']['f1_optimo']}")
    print(f"  CON multi-anio -> ROC {con['test']['roc_auc']} | PR {con['test']['pr_auc']} | F1 {con['test']['f1_optimo']}")
    print(f"  Delta (CON-SIN) -> ROC {deltas_test['roc_auc']} | PR {deltas_test['pr_auc']} | F1 {deltas_test['f1_optimo']}")
    print(f"  Columnas plurianuales anadidas ({len(cols_nuevas)}): {cols_nuevas}")
    print(f"  NO REGRESA EL PISO: {no_regresa} | MEJORA EL PISO: {mejora}")
    print(f"  RECOMENDACION USAR_FEATURES_MULTIANIO = {no_regresa}")
    print(f"\n[OK] Guardado {ruta}")
    print("\nProceso de analisis FEAT-005 completado con exito.")
    return resultado


if __name__ == "__main__":
    main()
