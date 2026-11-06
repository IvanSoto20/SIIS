# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-004: EVALUACION HONESTA DE LA ZONA ROJA Y DECISION DE ADOPCION
==============================================================================
Objetivo de negocio (requisito del usuario)
--------------------------------------------
"Subir la zona roja": detectar MAS quiebras entre el pequeno grupo de empresas que
el modelo marca como mas peligrosas (top 0.5 / 1.0 / 2.5 por ciento), SIN regresar el
PISO publicado. Este modulo es el BANCO DE PRUEBAS que, con un unico protocolo
honesto, mide cuatro variantes contra el PISO y emite una decision de adopcion
auditable. Primero reproduce EXACTAMENTE el PISO (gate de montaje) y solo entonces
compara.

Las cuatro variantes (mismos pliegues OOF y mismo Test reservado)
-----------------------------------------------------------------
  base         PISO: ensamble plano (0.30 LGBM + 0.30 XGB + 0.20 HGB + 0.20 RF),
               sin features de anexos, sin cascada.
  anexos       PALANCA 1: USAR_FEATURES_ANEXOS activo (X gana 16 variables de
               520000/801500/801700), ensamble plano. El anti-fuga se reafirma:
               ninguna columna lleva el literal 2025 (las variables salen del corte
               2024 para el modelo 2024->2025).
  cascada      PALANCA 2: X del PISO (sin anexos), score via modelo_cascada de 2
               etapas. Se BARRE el hiperparametro PERCENTIL_COLA; el ganador del
               barrido se elige con OOF/entrenamiento, NO mirando el Test.
  anexos_cascada  PALANCA 1 + PALANCA 2: X con anexos, score via cascada (mismo
               barrido de PERCENTIL_COLA, ganador elegido por OOF).

Protocolo honesto (identico al de las tandas previas)
-----------------------------------------------------
  * OOF de 5 pliegues (StratifiedKFold shuffle random_state=42) sobre las 19937
    empresas (304 positivos). Para el ensamble plano, cross_val_predict por modelo y
    mezcla ponderada. Para la cascada, una cascada completa por pliegue
    (score_cascada_oof), de modo que ninguna fila se puntua con modelos que la vieron.
  * Test reservado de 3988 empresas (61 positivos), con el MISMO train_test_split
    estratificado (test_size=0.20, random_state=42) que el PISO.
  * metricas() y bootstrap_pareado() se REUTILIZAN de analisis_apilado_transiciones
    (mismo remuestreo pareado ~500 iteraciones, semilla fija). Se miden explicitamente
    los cortes top 0.5 / 1.0 / 2.5 por ciento y el max-F1.

REGLA DE ADOPCION (declarada ANTES de ver resultados)
=====================================================
Se declara aqui, como en replica_confirmatoria_apilado.py (C1/C2/C3) y en
analisis_apilado_transiciones.py (G1..G5), para que la decision no se tome a
posteriori. Una variante se ADOPTA por defecto SOLO si cumple G1, G2, G3 y G4; en
empate o ambiguedad NO se adopta (queda tras bandera OFF). El PISO es un suelo
intocable.

  G1  OOF no-regresion (la variante no estropea el ranking global):
      PR-AUC(variante) >= PR-AUC(base) - RUIDO_PR  Y
      ROC-AUC(variante) >= ROC-AUC(base) - RUIDO_ROC.
  G2  Zona roja OOF mejora (el objetivo del usuario):
      TP en top 1% (y/o top 0.5%) de la variante ESTRICTAMENTE mayor que base,
      SIN caer en top 2.5% respecto a base.
  G3  Test reservado no-regresion:
      F1-max(variante) >= F1-max(PISO) - RUIDO_F1  Y
      rojo-elite TP (top 1%, 40 alertas) >= PISO_TEST_TOP1_TP - RUIDO_ELITE_TP.
  G4  Bootstrap pareado vs base: la mejora tiene respaldo estadistico, es decir, la
      probabilidad de mejora en PR-AUC O en recall de la cola (top 2.5%) O en
      precision de la cola (top 1%) es >= PROB_MEJORA_MIN (el IC95 no esta dominado
      por el lado negativo).

El barrido de PERCENTIL_COLA de la cascada se decide con OOF/entrenamiento (criterio
PRIMARIO: TP en top 1% OOF; desempate por TP top 0.5% OOF y luego PR-AUC OOF), NUNCA
eligiendo el que mejor sale en el Test reservado (eso seria elegir ganador a
posteriori). El Test reservado se usa solo para la no-regresion de G3, no para elegir
el tamano de cola.

Reglas duras respetadas
-----------------------
  * Solo lectura sobre py/ y SIIS/. Este modulo solo escribe en salidas/.
  * Cero fuga en todas las variantes: X proviene de construir_panel_y_splits
    (anti-fuga ya aseverado alli); se reafirma con un assert de que ninguna columna de
    X menciona 2025, en la variante base, en la de anexos y en ambas etapas de la
    cascada (el assert vive en modelo_cascada).
  * Importar NO dispara el entrenamiento del pipeline (guard __main__). Este modulo
    tiene su propio guard __main__.
  * Codigo, comentarios e informe en espanol, sin guiones largos.

Notas tecnicas (documentacion citada)
-------------------------------------
* sklearn.model_selection.cross_val_predict(method="predict_proba") devuelve, para
  cada muestra, la probabilidad predicha cuando esa muestra estaba FUERA del pliegue
  (prediccion out-of-fold), evitando el optimismo de evaluar sobre datos vistos. Se
  usa clone para no contaminar el estimador base.
* StratifiedKFold(shuffle=True, random_state=42) reproduce los pliegues del pipeline.
* sklearn.base.clone crea un estimador nuevo sin ajustar con los mismos
  hiperparametros.
* joblib.dump/load memoiza los scores OOF costosos en salidas/*.pkl para re-ejecutar
  rapido sin recomputar.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Pipeline de produccion: panel, escuadron y utilidades del ensamble. Importar es
# seguro porque todo el entrenamiento vive bajo el guard __main__ del modulo.
import script_cinco_optimizado as base

# Arnes de evaluacion honesta ya existente: se REUTILIZAN sus metricas de zona roja y
# su bootstrap pareado, junto con las constantes de pliegues y semilla. Importar es
# seguro (su logica pesada vive bajo el guard __main__).
from analisis_apilado_transiciones import (
    N_FOLDS,
    SEMILLA,
    bootstrap_pareado,
    metricas,
)

# ------------------------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------------------------
# SMOKE reduce el barrido y el numero de pliegues para validar el codigo rapido.
SMOKE = os.environ.get("EVAL_ZONA_ROJA_SMOKE") == "1"

# Valores del PISO (ver salidas/reporte_fase5.json y context.json: piso_metricas).
# Se usan como referencia de reproduccion; las tolerancias absorben solo ruido de
# ultima decimal por versiones de librerias, nunca una regresion real.
PISO_TEST = {"roc_auc": 0.8782, "pr_auc": 0.3115, "f1_optimo": 0.4144,
             "top1pct_tp": 19, "top1pct_alertas": 40}
PISO_OOF = {"roc_auc": 0.8867, "pr_auc": 0.2534,
            "top1pct_tp_aprox": 85, "top1pct_alertas_aprox": 200}
TOLERANCIA_ROC = 0.003
TOLERANCIA_PR = 0.003

# ------------------------------------------------------------------------------
# REGLA DE ADOPCION (parametros, declarados ANTES de los resultados)
# ------------------------------------------------------------------------------
# Umbrales de ruido para los gates G1..G4. Se eligen del mismo orden de magnitud que
# las tandas previas (TOLERANCIA 0.005 en analisis_apilado, 0.002 en ROC OOF).
RUIDO_PR = 0.005          # G1: cuanto puede caer PR-AUC OOF sin considerarse regresion.
RUIDO_ROC = 0.002         # G1: cuanto puede caer ROC-AUC OOF sin considerarse regresion.
RUIDO_F1 = 0.005          # G3: cuanto puede caer F1-max del Test sin regresion.
RUIDO_ELITE_TP = 1        # G3: cuantos TP del rojo elite (top1% Test) se toleran a la baja.
PROB_MEJORA_MIN = 0.60    # G4: probabilidad minima de mejora en el bootstrap pareado.

# Barrido de PERCENTIL_COLA de la cascada. El ganador se elige por OOF/entrenamiento,
# nunca por el Test. 88 es el valor base de FEAT-003; se barren una cola mas ancha (85)
# y una mas estrecha (90) alrededor del valor base para no descartar la cascada por un
# tamano subobtimo. El barrido se acota a 3 valores (en vez de 5) para respetar el tope
# de tiempo de reloj del usuario sin sacrificar OOF de 5 pliegues ni el Test reservado;
# cada cascada OOF cuesta ~350 s, de modo que 3 valores x 2 cascadas se mantiene dentro
# del presupuesto. El valor base 88 queda siempre cubierto.
PERCENTILES_COLA = (88.0,) if SMOKE else (85.0, 88.0, 90.0)

RUTA_JSON = base.SALIDAS_DIR / ("evaluacion_zona_roja_smoke.json" if SMOKE
                                else "evaluacion_zona_roja.json")
RUTA_JSON_BASE = base.SALIDAS_DIR / "evaluacion_zona_roja_base.json"
RUTA_CACHE = base.SALIDAS_DIR / ("cache_evaluacion_zona_roja_smoke.pkl" if SMOKE
                                 else "cache_evaluacion_zona_roja.pkl")


# ------------------------------------------------------------------------------
# BLOQUE 1: ESCUADRON DEL PISO (hiperparametros identicos a entrenar_escuadron_elite)
# ------------------------------------------------------------------------------
def crear_modelos_piso() -> dict:
    """
    Devuelve el mismo escuadron de 4 modelos del PISO con sus nombres canonicos
    (los que entiende ``calcular_pesos_ensamble``: LightGBM / XGBoost /
    HistGradientBoosting / RandomForest). Reutiliza los mismos hiperparametros que
    ``entrenar_escuadron_elite`` y ``proyectar_semaforo_2026`` del pipeline, por lo
    que el banco de pruebas y la produccion comparten exactamente la misma receta.
    LightGBM y XGBoost se incluyen solo si su libreria cargo (mismo blindaje opcional
    que el pipeline, controlado por ``base.TIENE_LGBM`` / ``base.TIENE_XGB``).
    """
    modelos = {}
    if base.TIENE_LGBM:
        modelos["LightGBM"] = base.LGBMClassifier(
            n_estimators=450, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
            scale_pos_weight=7, random_state=base.RANDOM_STATE, verbose=-1, n_jobs=-1)
    if base.TIENE_XGB:
        modelos["XGBoost"] = base.XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=7,
            tree_method="hist", eval_metric="aucpr", random_state=base.RANDOM_STATE,
            n_jobs=-1)
    modelos["HistGradientBoosting"] = base.HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=25,
        l2_regularization=5.0, class_weight={0: 1, 1: 8}, random_state=base.RANDOM_STATE)
    modelos["RandomForest"] = base.RandomForestClassifier(
        n_estimators=500, max_depth=9, min_samples_leaf=8, max_features="sqrt",
        class_weight={0: 1, 1: 10}, random_state=base.RANDOM_STATE, n_jobs=-1)
    return modelos


# ------------------------------------------------------------------------------
# BLOQUE 2: SCORES HONESTOS DEL ENSAMBLE (OOF de 5 pliegues y Test reservado)
# ------------------------------------------------------------------------------
def score_oof_ensamble(X, y, modelos: dict) -> np.ndarray:
    """
    Reproduce EXACTAMENTE el OOF del PISO (el que calcula ``proyectar_semaforo_2026``):
    para cada modelo se obtiene su probabilidad out-of-fold con ``cross_val_predict``
    sobre los MISMOS pliegues (``StratifiedKFold(N_FOLDS, shuffle=True,
    random_state=SEMILLA)``) y luego se mezclan con los pesos renormalizados del
    ensamble (0.30/0.30/0.20/0.20). No hay fuga: cada muestra se puntua con modelos
    que nunca la vieron en su pliegue de entrenamiento.

    Este es el GANCHO honesto para evaluar variantes en OOF: una variante (p.ej. la de
    anexos) solo cambia ``X`` manteniendo este mismo protocolo, de modo que la
    comparacion contra el PISO sea pareja.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    oofs = {}
    for nombre, modelo in modelos.items():
        oofs[nombre] = cross_val_predict(
            clone(modelo), X, y, cv=skf, method="predict_proba")[:, 1]
    pesos = base.calcular_pesos_ensamble(oofs.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    return sum(peso * oofs[nombre] for nombre, peso in pesos.items())


def score_test_ensamble(X_train, y_train, X_test, modelos: dict) -> np.ndarray:
    """
    Reproduce EXACTAMENTE el score de Test del PISO (``entrenar_escuadron_elite``):
    entrena cada modelo en X_train y puntua X_test; luego mezcla con los pesos
    renormalizados del ensamble. Es el GANCHO honesto para evaluar variantes en el
    Test reservado de 3988 empresas (61 positivos).
    """
    scores = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        scores[nombre] = modelo.predict_proba(X_test)[:, 1]
    pesos = base.calcular_pesos_ensamble(scores.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    return sum(peso * scores[nombre] for nombre, peso in pesos.items())


# ------------------------------------------------------------------------------
# BLOQUE 3: GANCHOS DE EVALUACION DE VARIANTES
# ------------------------------------------------------------------------------
def evaluar_variante_oof(X, y, modelos: dict, etiqueta: str) -> dict:
    """Evalua una variante de ENSAMBLE PLANO en OOF de N_FOLDS pliegues y devuelve su
    score y sus metricas de zona roja. ``modelos`` define la variante; con
    ``crear_modelos_piso()`` reproduce el PISO (sobre la X que se pase)."""
    y_np = np.asarray(y)
    s = score_oof_ensamble(X, y, modelos)
    m = metricas(y_np, s)
    m["_etiqueta"] = etiqueta
    return {"score": s, "metricas": m}


def evaluar_variante_test(X_train, y_train, X_test, y_test, modelos: dict, etiqueta: str) -> dict:
    """Evalua una variante de ENSAMBLE PLANO en el Test reservado y devuelve su score y
    sus metricas de zona roja."""
    y_te_np = np.asarray(y_test)
    s = score_test_ensamble(X_train, y_train, X_test, modelos)
    m = metricas(y_te_np, s)
    m["_etiqueta"] = etiqueta
    return {"score": s, "metricas": m}


# ------------------------------------------------------------------------------
# BLOQUE 4: GANCHO DE COMPARACION PAREADA CONTRA EL PISO
# ------------------------------------------------------------------------------
def comparar_contra_piso(y, s_piso: np.ndarray, s_variante: np.ndarray) -> dict:
    """
    Cuantifica, con el ``bootstrap_pareado`` del arnes existente, si una variante
    supera al PISO por encima del ruido: deltas de PR-AUC, ROC-AUC, precision top1% y
    recall top2.5% con IC95% y probabilidad de mejora (mismo remuestreo con reemplazo y
    semilla fija para base y variante).
    """
    return bootstrap_pareado(np.asarray(y), s_piso, s_variante)


# ------------------------------------------------------------------------------
# BLOQUE 5: CONSTRUCCION DE X CON ANEXOS (anti-fuga reafirmado)
# ------------------------------------------------------------------------------
def construir_panel_con_anexos():
    """
    Reconstruye el panel del PISO con las 16 features de anexos de FEAT-002 ACTIVAS,
    SIN modificar el valor por defecto de la bandera en el modulo fuente. Se activa la
    bandera ``base.USAR_FEATURES_ANEXOS`` en runtime SOLO durante la llamada a
    ``construir_panel_y_splits`` (que delega en ``construir_matriz_caracteristicas``,
    el unico sitio que lee la bandera) y se restaura su valor original al terminar, en
    un bloque try/finally. Asi la X de la variante de anexos sale por la MISMA ruta de
    produccion (corte 2024 para el modelo 2024->2025), el anti-fuga del pipeline se
    aplica igual y el default de la bandera sigue siendo False.

    Devuelve la misma tupla que ``construir_panel_y_splits``.
    """
    valor_original = base.USAR_FEATURES_ANEXOS
    try:
        base.USAR_FEATURES_ANEXOS = True
        resultado = base.construir_panel_y_splits()
    finally:
        base.USAR_FEATURES_ANEXOS = valor_original
    assert base.USAR_FEATURES_ANEXOS is valor_original, \
        "La bandera USAR_FEATURES_ANEXOS no se restauro a su valor original."
    return resultado


def _assert_anti_fuga(X, contexto: str):
    """Reafirma el anti-fuga: ninguna columna de X puede mencionar 2025."""
    assert not any("2025" in str(c) for c in X.columns), \
        f"ALERTA CRITICA: fuga de datos (columna 2025) en X ({contexto})."


# ------------------------------------------------------------------------------
# BLOQUE 6: EVALUACION DE LA CASCADA (con barrido de PERCENTIL_COLA)
# ------------------------------------------------------------------------------
def evaluar_cascada(X, y, X_train, y_train, X_test, y_test, etiqueta: str,
                    cache: dict) -> dict:
    """
    Evalua la variante de CASCADA barriendo PERCENTIL_COLA. Para cada valor del barrido
    computa (y cachea en disco) el score OOF honesto de la cascada
    (``modelo_cascada.score_cascada_oof``) y el score de Test
    (``modelo_cascada.score_cascada_test``), y mide ambas con ``metricas``. El GANADOR
    del barrido se elige con OOF/entrenamiento (criterio primario TP top 1% OOF;
    desempate TP top 0.5% OOF, luego PR-AUC OOF), NUNCA mirando el Test.

    ``etiqueta`` distingue la cascada sobre X del PISO ("cascada") de la cascada sobre
    X con anexos ("anexos_cascada"); se usa en las claves de cache para no mezclar.
    """
    # Import local: con la cascada apagada, modelo_cascada ni se importa en produccion.
    from modelo_cascada import score_cascada_oof, score_cascada_test

    y_np, y_te_np = np.asarray(y), np.asarray(y_test)
    por_percentil = {}
    for pc in PERCENTILES_COLA:
        clave_oof = f"{etiqueta}::oof::pc{pc}"
        clave_test = f"{etiqueta}::test::pc{pc}"
        if clave_oof not in cache:
            t0 = time.time()
            cache[clave_oof] = score_cascada_oof(X, y, percentil_cola=pc)
            joblib.dump(cache, RUTA_CACHE)
            print(f"      [{clave_oof}] OOF listo en {time.time() - t0:.0f}s", flush=True)
        if clave_test not in cache:
            t0 = time.time()
            cache[clave_test] = score_cascada_test(X_train, y_train, X_test, percentil_cola=pc)
            joblib.dump(cache, RUTA_CACHE)
            print(f"      [{clave_test}] Test listo en {time.time() - t0:.0f}s", flush=True)
        s_oof, s_test = cache[clave_oof], cache[clave_test]
        m_oof, m_test = metricas(y_np, s_oof), metricas(y_te_np, s_test)
        por_percentil[str(pc)] = {
            "percentil_cola": pc,
            "oof": {"score": s_oof, "metricas": m_oof},
            "test": {"score": s_test, "metricas": m_test},
        }
        c_o, c_t = m_oof["cortes"], m_test["cortes"]
        print(f"    {etiqueta} pc={pc:5.1f} | OOF ROC {m_oof['roc_auc']:.4f} PR {m_oof['pr_auc']:.4f} "
              f"top0.5% {c_o['0.5']['tp']}/{c_o['0.5']['alertas']} top1% {c_o['1.0']['tp']}/{c_o['1.0']['alertas']} "
              f"top2.5% {c_o['2.5']['tp']}/{c_o['2.5']['alertas']} | "
              f"TEST top0.5% {c_t['0.5']['tp']}/{c_t['0.5']['alertas']} top1% {c_t['1.0']['tp']}/{c_t['1.0']['alertas']} "
              f"F1max {m_test['f1_max']['f1']:.4f}", flush=True)

    # --- Eleccion del ganador por OOF (criterio primario TP top1%, desempates) ---
    def clave_orden(pc_str):
        m = por_percentil[pc_str]["oof"]["metricas"]
        return (m["cortes"]["1.0"]["tp"], m["cortes"]["0.5"]["tp"], m["pr_auc"])

    ganador = max(por_percentil.keys(), key=clave_orden)
    print(f"    GANADOR OOF del barrido ({etiqueta}): PERCENTIL_COLA={ganador} "
          f"(criterio TP top1% OOF, desempate top0.5% OOF y PR-AUC OOF).")
    return {"por_percentil": por_percentil, "ganador": ganador}


# ------------------------------------------------------------------------------
# BLOQUE 7: EVALUACION DE LOS GATES (regla declarada arriba)
# ------------------------------------------------------------------------------
def evaluar_gates(m_oof_base, m_test_base, m_oof_var, m_test_var, boot) -> dict:
    """
    Aplica la REGLA DE ADOPCION declarada en el docstring (G1..G4) a una variante dada
    sus metricas OOF y Test y su bootstrap pareado contra la base. Devuelve el veredicto
    de cada gate, los valores que lo sustentan y la decision agregada.
    """
    pr_b, roc_b = m_oof_base["pr_auc"], m_oof_base["roc_auc"]
    pr_v, roc_v = m_oof_var["pr_auc"], m_oof_var["roc_auc"]
    tp05_b, tp1_b, tp25_b = (m_oof_base["cortes"]["0.5"]["tp"],
                             m_oof_base["cortes"]["1.0"]["tp"],
                             m_oof_base["cortes"]["2.5"]["tp"])
    tp05_v, tp1_v, tp25_v = (m_oof_var["cortes"]["0.5"]["tp"],
                             m_oof_var["cortes"]["1.0"]["tp"],
                             m_oof_var["cortes"]["2.5"]["tp"])
    f1_b, f1_v = m_test_base["f1_max"]["f1"], m_test_var["f1_max"]["f1"]
    elite_b, elite_v = m_test_base["cortes"]["1.0"]["tp"], m_test_var["cortes"]["1.0"]["tp"]

    g1 = bool(pr_v >= pr_b - RUIDO_PR and roc_v >= roc_b - RUIDO_ROC)
    g2 = bool((tp1_v > tp1_b or tp05_v > tp05_b) and tp25_v >= tp25_b)
    g3 = bool(f1_v >= f1_b - RUIDO_F1 and elite_v >= elite_b - RUIDO_ELITE_TP)

    prob_pr = boot["delta_pr_auc"]["prob_mejora"]
    prob_recall = boot["delta_recall_top2_5pct"]["prob_mejora"]
    prob_prec1 = boot["delta_precision_top1pct"]["prob_mejora"]
    g4 = bool(max(prob_pr, prob_recall, prob_prec1) >= PROB_MEJORA_MIN)

    gates = {"G1_oof_no_regresion": g1, "G2_zona_roja_oof_mejora": g2,
             "G3_test_no_regresion": g3, "G4_bootstrap_respalda_mejora": g4}
    return {
        "gates": gates,
        "adoptar": bool(all(gates.values())),
        "valores": {
            "oof_pr_base_var": [pr_b, pr_v], "oof_roc_base_var": [roc_b, roc_v],
            "oof_tp_top0_5_base_var": [tp05_b, tp05_v],
            "oof_tp_top1_base_var": [tp1_b, tp1_v],
            "oof_tp_top2_5_base_var": [tp25_b, tp25_v],
            "test_f1max_base_var": [f1_b, f1_v],
            "test_rojo_elite_tp_base_var": [elite_b, elite_v],
            "bootstrap_prob_mejora": {"pr_auc": prob_pr, "recall_top2_5pct": prob_recall,
                                      "precision_top1pct": prob_prec1},
        },
    }


# ------------------------------------------------------------------------------
# BLOQUE 8: PRINCIPAL (montaje + 4 variantes + gates + decision + persistencia)
# ------------------------------------------------------------------------------
def _resumen_metricas(m: dict) -> dict:
    """Extrae los campos clave de un dict de metricas para el JSON (sin el score)."""
    return {
        "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
        "top_0_5pct": m["cortes"]["0.5"], "top_1pct": m["cortes"]["1.0"],
        "top_2_5pct": m["cortes"]["2.5"], "f1_max": m["f1_max"],
    }


def main():
    t_ini = time.time()
    print("=" * 100)
    print(f"EVALUACION HONESTA DE LA ZONA ROJA (FEAT-004): 4 VARIANTES{' [SMOKE]' if SMOKE else ''}")
    print("=" * 100)
    assert base.USAR_FEATURES_MULTIANIO is False, \
        "El banco reproduce el PISO sin features plurianuales (bandera debe estar en False)."

    cache = joblib.load(RUTA_CACHE) if RUTA_CACHE.exists() else {}

    # ======================================================================
    # PANEL BASE (PISO) y PANEL CON ANEXOS. Ambos reutilizan el mismo split.
    # ======================================================================
    print("\n[PANEL] Reconstruyendo el panel del PISO (sin anexos)...", flush=True)
    X, y, X_train, X_test, y_train, y_test, panel, _df_2025 = base.construir_panel_y_splits()
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True); y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True); y_test = y_test.reset_index(drop=True)
    _assert_anti_fuga(X, "base")
    y_np, y_te_np = y.to_numpy(), y_test.to_numpy()
    print(f"  Panel: {len(X):,} empresas x {X.shape[1]} features | positivos={int(y.sum())} | "
          f"train={len(X_train):,}/{int(y_train.sum())} | test={len(X_test):,}/{int(y_test.sum())}")

    print("\n[PANEL] Reconstruyendo el panel con FEATURES DE ANEXOS activas (bandera en runtime)...",
          flush=True)
    Xa, ya, Xa_train, Xa_test, ya_train, ya_test, _p2, _d2 = construir_panel_con_anexos()
    Xa, ya = Xa.reset_index(drop=True), ya.reset_index(drop=True)
    Xa_train = Xa_train.reset_index(drop=True); ya_train = ya_train.reset_index(drop=True)
    Xa_test = Xa_test.reset_index(drop=True); ya_test = ya_test.reset_index(drop=True)
    _assert_anti_fuga(Xa, "anexos")
    assert base.USAR_FEATURES_ANEXOS is False, "La bandera de anexos debe seguir en False por defecto."
    n_extra = Xa.shape[1] - X.shape[1]
    assert n_extra > 0, "La variante de anexos no agrego columnas nuevas."
    # Las etiquetas y los pliegues deben coincidir con la base (mismo split estratificado).
    assert np.array_equal(ya.to_numpy(), y_np), "La y del panel con anexos difiere de la base."
    print(f"  Panel con anexos: {len(Xa):,} x {Xa.shape[1]} features ({n_extra} columnas de anexos extra).")

    # ======================================================================
    # VARIANTE base (PISO): OOF + Test del ensamble plano.
    # ======================================================================
    print("\n[VARIANTE base] OOF y Test del ensamble plano (reproduccion del PISO)...", flush=True)
    if "base::oof" not in cache:
        cache["base::oof"] = score_oof_ensamble(X, y, crear_modelos_piso()); joblib.dump(cache, RUTA_CACHE)
    if "base::test" not in cache:
        cache["base::test"] = score_test_ensamble(X_train, y_train, X_test, crear_modelos_piso())
        joblib.dump(cache, RUTA_CACHE)
    s_oof_base, s_test_base = cache["base::oof"], cache["base::test"]
    m_oof_base, m_test_base = metricas(y_np, s_oof_base), metricas(y_te_np, s_test_base)
    c = m_oof_base["cortes"]
    print(f"  OOF  -> ROC {m_oof_base['roc_auc']:.4f} PR {m_oof_base['pr_auc']:.4f} | "
          f"top0.5% {c['0.5']['tp']}/{c['0.5']['alertas']} top1% {c['1.0']['tp']}/{c['1.0']['alertas']} "
          f"top2.5% {c['2.5']['tp']}/{c['2.5']['alertas']} F1max {m_oof_base['f1_max']['f1']:.4f}")
    c = m_test_base["cortes"]
    print(f"  TEST -> ROC {m_test_base['roc_auc']:.4f} PR {m_test_base['pr_auc']:.4f} | "
          f"top0.5% {c['0.5']['tp']}/{c['0.5']['alertas']} top1% {c['1.0']['tp']}/{c['1.0']['alertas']} "
          f"top2.5% {c['2.5']['tp']}/{c['2.5']['alertas']} F1max {m_test_base['f1_max']['f1']:.4f}")

    # ---- GATE DE MONTAJE: el banco solo es valido si reproduce el PISO. ----
    assert abs(m_oof_base["roc_auc"] - PISO_OOF["roc_auc"]) < TOLERANCIA_ROC, \
        f"El arnes NO reproduce el OOF-ROC del PISO ({m_oof_base['roc_auc']} vs {PISO_OOF['roc_auc']})."
    assert abs(m_oof_base["pr_auc"] - PISO_OOF["pr_auc"]) < TOLERANCIA_PR, \
        f"El arnes NO reproduce el OOF-PR del PISO ({m_oof_base['pr_auc']} vs {PISO_OOF['pr_auc']})."
    assert abs(m_test_base["roc_auc"] - PISO_TEST["roc_auc"]) < TOLERANCIA_ROC, \
        f"El arnes NO reproduce el Test-ROC del PISO ({m_test_base['roc_auc']} vs {PISO_TEST['roc_auc']})."
    assert abs(m_test_base["pr_auc"] - PISO_TEST["pr_auc"]) < TOLERANCIA_PR, \
        f"El arnes NO reproduce el Test-PR del PISO ({m_test_base['pr_auc']} vs {PISO_TEST['pr_auc']})."
    assert m_test_base["cortes"]["1.0"]["tp"] == PISO_TEST["top1pct_tp"], \
        f"Test top1% TP = {m_test_base['cortes']['1.0']['tp']} (esperado {PISO_TEST['top1pct_tp']})."
    print("  GATE DE MONTAJE superado: el arnes reproduce el PISO en OOF y Test.")

    # ======================================================================
    # VARIANTE anexos: OOF + Test del ensamble plano sobre X con anexos.
    # ======================================================================
    print("\n[VARIANTE anexos] OOF y Test del ensamble plano sobre X con 16 features de anexos...",
          flush=True)
    if "anexos::oof" not in cache:
        cache["anexos::oof"] = score_oof_ensamble(Xa, ya, crear_modelos_piso()); joblib.dump(cache, RUTA_CACHE)
    if "anexos::test" not in cache:
        cache["anexos::test"] = score_test_ensamble(Xa_train, ya_train, Xa_test, crear_modelos_piso())
        joblib.dump(cache, RUTA_CACHE)
    s_oof_anexos, s_test_anexos = cache["anexos::oof"], cache["anexos::test"]
    m_oof_anexos, m_test_anexos = metricas(y_np, s_oof_anexos), metricas(y_te_np, s_test_anexos)
    c = m_oof_anexos["cortes"]
    print(f"  OOF  -> ROC {m_oof_anexos['roc_auc']:.4f} PR {m_oof_anexos['pr_auc']:.4f} | "
          f"top0.5% {c['0.5']['tp']}/{c['0.5']['alertas']} top1% {c['1.0']['tp']}/{c['1.0']['alertas']} "
          f"top2.5% {c['2.5']['tp']}/{c['2.5']['alertas']} F1max {m_oof_anexos['f1_max']['f1']:.4f}")
    c = m_test_anexos["cortes"]
    print(f"  TEST -> ROC {m_test_anexos['roc_auc']:.4f} PR {m_test_anexos['pr_auc']:.4f} | "
          f"top0.5% {c['0.5']['tp']}/{c['0.5']['alertas']} top1% {c['1.0']['tp']}/{c['1.0']['alertas']} "
          f"top2.5% {c['2.5']['tp']}/{c['2.5']['alertas']} F1max {m_test_anexos['f1_max']['f1']:.4f}")

    # ======================================================================
    # VARIANTE cascada: barrido de PERCENTIL_COLA sobre X del PISO.
    # ======================================================================
    print("\n[VARIANTE cascada] Barrido de PERCENTIL_COLA sobre X del PISO (sin anexos)...", flush=True)
    casc = evaluar_cascada(X, y, X_train, y_train, X_test, y_test, "cascada", cache)
    ganador_casc = casc["ganador"]
    s_oof_casc = casc["por_percentil"][ganador_casc]["oof"]["score"]
    s_test_casc = casc["por_percentil"][ganador_casc]["test"]["score"]
    m_oof_casc = casc["por_percentil"][ganador_casc]["oof"]["metricas"]
    m_test_casc = casc["por_percentil"][ganador_casc]["test"]["metricas"]

    # ======================================================================
    # VARIANTE anexos_cascada: barrido de PERCENTIL_COLA sobre X con anexos.
    # ======================================================================
    print("\n[VARIANTE anexos_cascada] Barrido de PERCENTIL_COLA sobre X con anexos...", flush=True)
    casc_a = evaluar_cascada(Xa, ya, Xa_train, ya_train, Xa_test, ya_test, "anexos_cascada", cache)
    ganador_casc_a = casc_a["ganador"]
    s_oof_casc_a = casc_a["por_percentil"][ganador_casc_a]["oof"]["score"]
    s_test_casc_a = casc_a["por_percentil"][ganador_casc_a]["test"]["score"]
    m_oof_casc_a = casc_a["por_percentil"][ganador_casc_a]["oof"]["metricas"]
    m_test_casc_a = casc_a["por_percentil"][ganador_casc_a]["test"]["metricas"]

    # ======================================================================
    # BOOTSTRAP PAREADO vs base (OOF) para cada variante.
    # ======================================================================
    print("\n[BOOTSTRAP] Comparacion pareada OOF de cada variante contra el PISO...", flush=True)
    boot = {
        "anexos": comparar_contra_piso(y_np, s_oof_base, s_oof_anexos),
        "cascada": comparar_contra_piso(y_np, s_oof_base, s_oof_casc),
        "anexos_cascada": comparar_contra_piso(y_np, s_oof_base, s_oof_casc_a),
    }

    # ======================================================================
    # GATES G1..G4 por variante.
    # ======================================================================
    gates = {
        "anexos": evaluar_gates(m_oof_base, m_test_base, m_oof_anexos, m_test_anexos, boot["anexos"]),
        "cascada": evaluar_gates(m_oof_base, m_test_base, m_oof_casc, m_test_casc, boot["cascada"]),
        "anexos_cascada": evaluar_gates(m_oof_base, m_test_base, m_oof_casc_a, m_test_casc_a,
                                        boot["anexos_cascada"]),
    }

    print("\n" + "=" * 100)
    print("VEREDICTO DE LOS GATES POR VARIANTE (regla declarada ANTES de ver resultados)")
    print("=" * 100)
    for nombre, g in gates.items():
        linea = "  ".join(f"{k.split('_')[0]}={'OK' if v else 'NO'}" for k, v in g["gates"].items())
        print(f"  {nombre:16s} {linea}  ==> ADOPTAR: {g['adoptar']}")

    # ======================================================================
    # DECISION FINAL. Adoptar por defecto SOLO una variante y solo si gana los gates.
    # Preferencia: la combinacion completa (anexos_cascada) si gana; luego cascada;
    # luego anexos. En empate o si ninguna gana, NO adoptar nada (PISO intocable).
    # ======================================================================
    adoptadas = [n for n in ("anexos_cascada", "cascada", "anexos") if gates[n]["adoptar"]]
    variante_adoptada = adoptadas[0] if adoptadas else None
    decision = {
        "variante_adoptada": variante_adoptada,
        "usar_features_anexos": variante_adoptada in ("anexos", "anexos_cascada"),
        "usar_cascada": variante_adoptada in ("cascada", "anexos_cascada"),
        "percentil_cola_elegido": (ganador_casc_a if variante_adoptada == "anexos_cascada"
                                   else ganador_casc if variante_adoptada == "cascada" else None),
    }
    print("\n" + "=" * 100)
    print(f"DECISION FINAL: variante adoptada = {variante_adoptada if variante_adoptada else 'NINGUNA (PISO intocable)'}")
    if variante_adoptada:
        print(f"  -> USAR_FEATURES_ANEXOS = {decision['usar_features_anexos']}, "
              f"USAR_CASCADA = {decision['usar_cascada']}, "
              f"PERCENTIL_COLA = {decision['percentil_cola_elegido']}")
    else:
        print("  -> Ambas banderas quedan en False; los artefactos de produccion NO se tocan.")
    print("=" * 100)

    # ======================================================================
    # PERSISTENCIA del veredicto.
    # ======================================================================
    def _barrido_resumen(casc_res):
        return {pc: {"oof": _resumen_metricas(d["oof"]["metricas"]),
                     "test": _resumen_metricas(d["test"]["metricas"])}
                for pc, d in casc_res["por_percentil"].items()}

    resultado = {
        "descripcion": ("Evaluacion honesta de la zona roja (FEAT-004). Mide 4 variantes "
                        "(base PISO, anexos, cascada, anexos+cascada) en OOF (5 pliegues) + "
                        "Test reservado + bootstrap pareado, con la regla de adopcion G1..G4 "
                        "declarada ANTES de ver resultados. El barrido de PERCENTIL_COLA se "
                        "decide por OOF, nunca por el Test."),
        "smoke": SMOKE,
        "panel": {"total": int(len(X)), "positivos": int(y.sum()),
                  "train": int(len(X_train)), "train_pos": int(y_train.sum()),
                  "test": int(len(X_test)), "test_pos": int(y_test.sum()),
                  "features_base": int(X.shape[1]), "features_anexos": int(Xa.shape[1])},
        "regla_de_adopcion": {
            "G1_oof_no_regresion": f"PR-AUC_var >= PR-AUC_base - {RUIDO_PR} y ROC-AUC_var >= ROC-AUC_base - {RUIDO_ROC}",
            "G2_zona_roja_oof_mejora": "TP top1% (o top0.5%) var > base, sin caer en top2.5%",
            "G3_test_no_regresion": f"F1max_var >= F1max_PISO - {RUIDO_F1} y rojo-elite TP >= PISO - {RUIDO_ELITE_TP}",
            "G4_bootstrap_respalda_mejora": f"prob. mejora (PR o recall top2.5% o prec top1%) >= {PROB_MEJORA_MIN}",
            "adopcion": "adoptar SOLO si G1..G4; en empate/ambiguedad NO adoptar.",
            "barrido_percentil_cola": "ganador elegido por OOF (TP top1%, desempate top0.5% y PR-AUC), nunca por el Test.",
        },
        "piso_publicado": {"oof": PISO_OOF, "test": PISO_TEST},
        "variantes": {
            "base": {"oof": _resumen_metricas(m_oof_base), "test": _resumen_metricas(m_test_base)},
            "anexos": {"oof": _resumen_metricas(m_oof_anexos), "test": _resumen_metricas(m_test_anexos)},
            "cascada": {"ganador_percentil_cola": ganador_casc,
                        "oof": _resumen_metricas(m_oof_casc), "test": _resumen_metricas(m_test_casc),
                        "barrido": _barrido_resumen(casc)},
            "anexos_cascada": {"ganador_percentil_cola": ganador_casc_a,
                               "oof": _resumen_metricas(m_oof_casc_a), "test": _resumen_metricas(m_test_casc_a),
                               "barrido": _barrido_resumen(casc_a)},
        },
        "bootstrap_vs_base_oof": boot,
        "gates_por_variante": gates,
        "decision": decision,
        "segundos_totales": round(time.time() - t_ini, 1),
    }
    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {RUTA_JSON.name} ({resultado['segundos_totales']} s).")
    print("Evaluacion honesta completada: veredicto de los gates y decision de adopcion en el JSON.")


if __name__ == "__main__":
    main()
