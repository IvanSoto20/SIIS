# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-001: ARNES DE EVALUACION HONESTA DE LA ZONA ROJA (ESQUELETO)
==============================================================================
Objetivo de negocio (requisito del usuario)
--------------------------------------------
"Subir la zona roja": detectar MAS quiebras entre el pequeno grupo de empresas que
el modelo marca como mas peligrosas (top 0.5 / 1.0 / 2.5 por ciento), sin regresar
el PISO publicado. Antes de intentar cualquier mejora (anexos nuevos de FEAT-002,
cascada de FEAT-003, decision de FEAT-004) hace falta un BANCO DE PRUEBAS honesto
que mida SIEMPRE con el mismo protocolo y que, como condicion de validez, reproduzca
EXACTAMENTE el PISO. Ese es el unico cometido de este modulo en FEAT-001.

Que hace este modulo AHORA (FEAT-001)
-------------------------------------
  1. Reconstruye el panel del PISO (2024 -> 2025) reutilizando por import
     ``construir_panel_y_splits`` del pipeline de produccion (asserts de integridad
     19937/304, 15949/243, 3988/61 embebidos alli).
  2. Calcula el score del ensamble del PISO en DOS protocolos honestos:
       a) OOF de 5 pliegues sobre las 19937 empresas, reproduciendo EXACTAMENTE la
          ruta de ``proyectar_semaforo_2026`` (``cross_val_predict`` por modelo con
          ``StratifiedKFold(shuffle=True, random_state=42)`` y luego mezcla ponderada
          0.30/0.30/0.20/0.20 renormalizada via ``calcular_pesos_ensamble``).
       b) Test reservado de 3988 empresas (61 positivos), reproduciendo EXACTAMENTE
          ``entrenar_escuadron_elite`` (entrena en X_train, puntua X_test).
  3. Calcula y guarda las metricas del PISO (ROC/PR en OOF y Test, cortes top
     0.5/1/2.5 por ciento y max-F1) reutilizando ``metricas`` del arnes ya existente
     ``analisis_apilado_transiciones``.
  4. AUTO-VALIDACION (assert): comprueba que el arnes REPRODUCE el PISO antes de
     servir como banco de pruebas (OOF ROC ~0.8867 / PR ~0.2534, top1% OOF ~85/200;
     Test ROC 0.8782 / PR 0.3115, top1% 19/40). Si no reproduce, FALLA de forma
     ruidosa en lugar de comparar variantes contra una base equivocada.
  5. Guarda salidas/evaluacion_zona_roja_base.json con el PISO reproducido.

Que NO hace todavia (se amplia en FEAT-004)
-------------------------------------------
NO evalua ninguna variante nueva (anexos, cascada). Las funciones de evaluacion de
variantes quedan como GANCHOS declarados (``evaluar_variante_oof`` /
``evaluar_variante_test``) que reutilizan el mismo protocolo honesto; en esta feature
solo se ejercitan con la configuracion del PISO para demostrar que el banco reproduce
la linea base. La REGLA DE ADOPCION declarada-antes y el ``bootstrap_pareado`` contra
el PISO se conectaran en FEAT-004.

Reglas duras respetadas
-----------------------
  * Solo lectura sobre py/ y SIIS/. Este modulo solo escribe en salidas/.
  * Cero fuga: X proviene de ``construir_panel_y_splits`` (anti-fuga ya aseverado
    alli); se reafirma con un assert de que ninguna columna de X menciona 2025.
  * Importar NO dispara el entrenamiento del pipeline (guard __main__ en ambos
    modulos importados). Este modulo tiene su propio guard __main__.
  * Codigo y comentarios en espanol, sin guiones largos.

Notas tecnicas (documentacion citada)
-------------------------------------
* ``sklearn.model_selection.cross_val_predict(method="predict_proba")`` devuelve, para
  cada muestra, la probabilidad predicha cuando esa muestra estaba FUERA del pliegue de
  entrenamiento (prediccion out-of-fold), lo que evita el optimismo de evaluar sobre
  datos vistos. Se usa ``clone`` para no contaminar el estimador base.
* ``StratifiedKFold(shuffle=True, random_state=42)`` reproduce exactamente los pliegues
  del pipeline, por eso la base reproduce el OOF del PISO.
* ``sklearn.base.clone`` crea un estimador nuevo sin ajustar con los mismos
  hiperparametros (documentacion de scikit-learn).
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
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
# CONSTANTES Y METRICAS DEL PISO PUBLICADO (fuente de verdad para la auto-validacion)
# ------------------------------------------------------------------------------
# Valores del PISO (ver salidas/reporte_fase5.json y context.json: piso_metricas).
# Se usan como referencia de reproduccion; las tolerancias absorben solo ruido de
# ultima decimal por versiones de librerias, nunca una regresion real.
PISO_TEST = {"roc_auc": 0.8782, "pr_auc": 0.3115, "f1_optimo": 0.4144,
             "top1pct_tp": 19, "top1pct_alertas": 40}
PISO_OOF = {"roc_auc": 0.8867, "pr_auc": 0.2534,
            "top1pct_tp_aprox": 85, "top1pct_alertas_aprox": 200}
TOLERANCIA_ROC = 0.003
TOLERANCIA_PR = 0.003

RUTA_JSON = base.SALIDAS_DIR / "evaluacion_zona_roja_base.json"


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

    Este es el GANCHO honesto para evaluar variantes en OOF: una variante futura
    (anexos / cascada) solo debe sustituir ``modelos`` y/o ``X`` manteniendo este
    mismo protocolo, de modo que la comparacion contra el PISO sea pareja.
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
# BLOQUE 3: GANCHOS DE EVALUACION DE VARIANTES (se amplian en FEAT-004)
# ------------------------------------------------------------------------------
# Estas dos funciones son el punto de extension del banco de pruebas. En FEAT-001
# solo se ejercitan con la configuracion del PISO (modelos = crear_modelos_piso()),
# demostrando que el arnes reproduce la linea base. En FEAT-004 recibiran las
# variantes nuevas (X con anexos, especialista de la cascada, etc.) SIN cambiar el
# protocolo honesto: mismos pliegues, mismo Test reservado, mismas metricas.
def evaluar_variante_oof(X, y, modelos: dict, etiqueta: str) -> dict:
    """Evalua una variante en OOF de N_FOLDS pliegues y devuelve sus metricas de zona
    roja (``metricas`` del arnes existente). ``modelos`` define la variante; con
    ``crear_modelos_piso()`` reproduce el PISO."""
    y_np = np.asarray(y)
    s = score_oof_ensamble(X, y, modelos)
    m = metricas(y_np, s)
    m["_etiqueta"] = etiqueta
    return {"score": s, "metricas": m}


def evaluar_variante_test(X_train, y_train, X_test, y_test, modelos: dict, etiqueta: str) -> dict:
    """Evalua una variante en el Test reservado y devuelve sus metricas de zona roja.
    ``modelos`` define la variante; con ``crear_modelos_piso()`` reproduce el PISO."""
    y_te_np = np.asarray(y_test)
    s = score_test_ensamble(X_train, y_train, X_test, modelos)
    m = metricas(y_te_np, s)
    m["_etiqueta"] = etiqueta
    return {"score": s, "metricas": m}


# ------------------------------------------------------------------------------
# BLOQUE 4: GANCHO DE COMPARACION PAREADA CONTRA EL PISO (se conecta en FEAT-004)
# ------------------------------------------------------------------------------
def comparar_contra_piso(y, s_piso: np.ndarray, s_variante: np.ndarray) -> dict:
    """
    Gancho para FEAT-004: cuantifica, con el ``bootstrap_pareado`` del arnes existente,
    si una variante supera al PISO por encima del ruido (deltas de PR-AUC, ROC-AUC,
    precision top1% y recall top2.5% con IC95% y probabilidad de mejora). En FEAT-001
    NO se usa para decidir nada; queda declarado para no duplicar logica despues.
    """
    return bootstrap_pareado(np.asarray(y), s_piso, s_variante)


# ------------------------------------------------------------------------------
# BLOQUE 5: AUTO-VALIDACION Y PERSISTENCIA
# ------------------------------------------------------------------------------
def main():
    t_ini = time.time()
    print("=" * 100)
    print("ARNES DE EVALUACION HONESTA DE LA ZONA ROJA (FEAT-001): REPRODUCCION DEL PISO")
    print("=" * 100)
    # El banco se mide sobre el PISO sin features plurianuales (igual que el publicado).
    assert base.USAR_FEATURES_MULTIANIO is False, \
        "El banco reproduce el PISO sin features plurianuales (bandera debe estar en False)."

    # ---- Panel del PISO (2024 -> 2025). Los asserts 19937/304, 15949/243, 3988/61
    #      viven dentro de construir_panel_y_splits (pipeline de produccion).
    X, y, X_train, X_test, y_train, y_test, panel, _df_2025 = base.construir_panel_y_splits()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_np = y.to_numpy()
    y_test_np = y_test.to_numpy()

    # Reafirmacion anti-fuga local (ademas de los asserts del pipeline).
    assert not any("2025" in str(c) for c in X.columns), "Fuga: columna 2025 en X."
    print(f"Panel reproducido: {len(X):,} empresas x {X.shape[1]} features | "
          f"positivos={int(y.sum())} | train={len(X_train):,}/{int(y_train.sum())} | "
          f"test={len(X_test):,}/{int(y_test.sum())}")

    # ---- Configuracion del PISO (los 4 modelos canonicos del ensamble).
    modelos_piso = crear_modelos_piso()
    print(f"Escuadron del PISO: {list(modelos_piso.keys())}")

    # ---- Protocolo honesto A: OOF de 5 pliegues sobre las 19937 empresas.
    print(f"\n[OOF] Calculando OOF de {N_FOLDS} pliegues (reproduccion del PISO)...", flush=True)
    s_oof = score_oof_ensamble(X, y, crear_modelos_piso())
    m_oof = metricas(y_np, s_oof)
    print(f"  OOF -> ROC {m_oof['roc_auc']:.4f} | PR {m_oof['pr_auc']:.4f} | "
          f"TP top0.5%: {m_oof['cortes']['0.5']['tp']}/{m_oof['cortes']['0.5']['alertas']} | "
          f"TP top1%: {m_oof['cortes']['1.0']['tp']}/{m_oof['cortes']['1.0']['alertas']} | "
          f"TP top2.5%: {m_oof['cortes']['2.5']['tp']}/{m_oof['cortes']['2.5']['alertas']} | "
          f"F1max {m_oof['f1_max']['f1']:.4f}", flush=True)

    # ---- Protocolo honesto B: Test reservado de 3988 empresas (61 positivos).
    print("\n[TEST] Entrenando en X_train y puntuando X_test (reproduccion del PISO)...", flush=True)
    s_test = score_test_ensamble(X_train, y_train, X_test, crear_modelos_piso())
    m_test = metricas(y_test_np, s_test)
    print(f"  TEST -> ROC {m_test['roc_auc']:.4f} | PR {m_test['pr_auc']:.4f} | "
          f"TP top0.5%: {m_test['cortes']['0.5']['tp']}/{m_test['cortes']['0.5']['alertas']} | "
          f"TP top1%: {m_test['cortes']['1.0']['tp']}/{m_test['cortes']['1.0']['alertas']} | "
          f"TP top2.5%: {m_test['cortes']['2.5']['tp']}/{m_test['cortes']['2.5']['alertas']} | "
          f"F1max {m_test['f1_max']['f1']:.4f}", flush=True)

    # ---- AUTO-VALIDACION: el banco solo es valido si reproduce el PISO.
    comprobaciones = {
        "oof_roc": (m_oof["roc_auc"], PISO_OOF["roc_auc"], TOLERANCIA_ROC),
        "oof_pr": (m_oof["pr_auc"], PISO_OOF["pr_auc"], TOLERANCIA_PR),
        "test_roc": (m_test["roc_auc"], PISO_TEST["roc_auc"], TOLERANCIA_ROC),
        "test_pr": (m_test["pr_auc"], PISO_TEST["pr_auc"], TOLERANCIA_PR),
    }
    for nombre_check, (obtenido, esperado, tol) in comprobaciones.items():
        assert abs(obtenido - esperado) < tol, (
            f"El arnes NO reproduce el PISO en {nombre_check}: obtenido {obtenido:.4f} "
            f"vs esperado {esperado:.4f} (tolerancia {tol}). No se puede usar como base.")
    # La zona roja del Test debe reproducir el corte emblematico top1% 19/40.
    assert m_test["cortes"]["1.0"]["tp"] == PISO_TEST["top1pct_tp"], (
        f"Test top1% TP = {m_test['cortes']['1.0']['tp']} (esperado {PISO_TEST['top1pct_tp']}).")
    assert m_test["cortes"]["1.0"]["alertas"] == PISO_TEST["top1pct_alertas"], (
        f"Test top1% alertas = {m_test['cortes']['1.0']['alertas']} "
        f"(esperado {PISO_TEST['top1pct_alertas']}).")
    print("\n  Arnes VALIDADO: reproduce el PISO en OOF y Test dentro de tolerancia.")

    # ---- Persistencia del PISO reproducido (linea base para FEAT-002/003/004).
    resultado = {
        "descripcion": ("Arnes de evaluacion honesta de la zona roja (FEAT-001). "
                        "Reproduce el PISO en OOF (5 pliegues) y en el Test reservado. "
                        "Base contra la que FEAT-004 comparara las variantes."),
        "panel": {
            "total": int(len(X)), "positivos": int(y.sum()),
            "train": int(len(X_train)), "train_pos": int(y_train.sum()),
            "test": int(len(X_test)), "test_pos": int(y_test.sum()),
            "features": int(X.shape[1]),
        },
        "escuadron": list(modelos_piso.keys()),
        "piso_reproducido": {"oof": m_oof, "test": m_test},
        "piso_publicado": {"oof": PISO_OOF, "test": PISO_TEST},
        "auto_validacion": {
            nombre_check: {"obtenido": round(float(obtenido), 4),
                           "esperado": esperado,
                           "diferencia_abs": round(abs(float(obtenido) - esperado), 4),
                           "tolerancia": tol,
                           "reproduce": bool(abs(obtenido - esperado) < tol)}
            for nombre_check, (obtenido, esperado, tol) in comprobaciones.items()
        },
        "segundos_totales": round(time.time() - t_ini, 1),
    }
    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {RUTA_JSON.name} ({resultado['segundos_totales']} s).")
    print("Arnes listo: el PISO queda reproducido y disponible como linea base honesta.")


if __name__ == "__main__":
    main()
