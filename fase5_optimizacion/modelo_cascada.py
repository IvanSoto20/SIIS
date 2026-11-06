# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-003: MODELO EN CASCADA DE 2 ETAPAS (PALANCA 2 PARA LA ZONA ROJA)
==============================================================================
Objetivo de negocio (requisito del usuario)
--------------------------------------------
"Subir la zona roja": detectar MAS quiebras entre el pequeno grupo de empresas que
el modelo marca como mas peligrosas (top 0.5 / 1.0 / 2.5 por ciento). El cuello de
botella no es separar sanas de enfermas (el ensamble plano ya lo hace bien: OOF ROC
~0.8867), sino separar, DENTRO del grupo deteriorado, las empresas que efectivamente
quiebran de las que sobreviven estresadas. Esta es la PALANCA 2: un clasificador en
CASCADA de dos etapas con un ESPECIALISTA entrenado solo sobre la cola de riesgo.

Arquitectura (dos etapas)
-------------------------
ETAPA 1 (filtro sanas vs en-riesgo). Es el mismo super-ensamble del PISO (LightGBM +
XGBoost + HistGradientBoosting + RandomForest con los hiperparametros EXACTOS y los
pesos 0.30/0.30/0.20/0.20 renormalizados a 1.0 via ``calcular_pesos_ensamble``). Su
score ordena a toda la poblacion. Se define el subconjunto "en-riesgo" como la COLA
SUPERIOR del score de etapa 1 (top ``PERCENTIL_COLA`` por ciento). El umbral de la
cola se fija SOLO con datos de entrenamiento / via OOF, NUNCA mirando el Test.

ETAPA 2 (especialista en deterioro). Un segundo super-ensamble (misma receta de 4
modelos) entrenado EXCLUSIVAMENTE sobre las filas de entrenamiento que caen en la
cola de etapa 1, con ``scale_pos_weight`` / ``class_weight`` RECALCULADOS al
desbalance real del subconjunto (que es mucho mas rico en positivos que la poblacion
total). Su tarea es la dificil: dentro del grupo deteriorado, separar quiebra de
sobrevive-estresada. La etiqueta sigue siendo y (quiebra 2025); jamas una columna
*_2025 entra como feature.

ANTI-FUGA INTERNO (critico)
---------------------------
La pertenencia al subconjunto en-riesgo EN ENTRENAMIENTO se deriva via OOF de la
etapa 1 (``cross_val_predict`` con ``StratifiedKFold(shuffle=True, random_state=42)``),
NO con el modelo de etapa 1 ya ajustado sobre esas mismas filas. Si se usara el
modelo ya ajustado, la cola de entrenamiento estaria contaminada por el optimismo de
evaluar sobre datos vistos y la etapa 2 aprenderia sobre una particion sesgada. Con
OOF, cada fila entra (o no) a la cola segun un score que NO la vio en su pliegue.

Para inferencia (Test, OOF del panel, proyeccion 2026) la pertenencia a la cola se
decide con el score de etapa 1 ya ajustado y el umbral de cola fijado en
entrenamiento (percentil del score OOF de entrenamiento), de modo que el umbral es
un parametro aprendido, no un mirar-el-futuro.

COMBINACION DEL SCORE (una sola probabilidad por empresa, monotona-compatible)
------------------------------------------------------------------------------
Se elige el RE-RANKING DE LA COLA preservando el orden global, implementado de forma
que el score resultante sea un unico numero en [0, 1] monotono-compatible con la
calibracion isotonica del pipeline:

  * Empresas FUERA de la cola de etapa 1 (las que la etapa 1 considera de bajo
    riesgo): ``P_final = u_cola * P1`` , con u_cola = umbral de la cola en [0,1].
    Como P1 < u_cola para estas filas por definicion de la cola, al multiplicar por
    u_cola (<=1) su score queda en [0, u_cola^2) subset de [0, u_cola): SIEMPRE por
    debajo de la cola. Se conserva el orden interno (monotono en P1).
  * Empresas DENTRO de la cola: ``P_final = u_cola + (1 - u_cola) * P2`` , con P2 =
    probabilidad del especialista de etapa 2. Esto mapea la cola al tramo
    [u_cola, 1], SIEMPRE por encima del resto, y la REORDENA por el criterio del
    especialista (el que de verdad separa quiebra de supervivencia estresada).

Propiedades (justificacion):
  1. UN unico score por empresa en [0, 1] (ambos tramos viven en [0,1] y no se solapan).
  2. La cola queda SIEMPRE por encima de las no-cola (u_cola es la frontera dura), de
     modo que la cascada nunca "rescata" hacia arriba a una empresa que la etapa 1
     descarto; solo REORDENA la cola, que es donde vive la zona roja (top 1-2.5%).
  3. Monotonia-compatible con IsotonicRegression: el pipeline calibra una funcion
     monotona creciente de ESTE score a la probabilidad empirica de quiebra. No se
     requiere que el score sea monotono en P1 globalmente, solo que sea un indicador
     de riesgo coherente y ordenable; el re-ranking preserva el orden dentro de cada
     tramo y respeta la frontera, que es exactamente lo que la isotonica necesita
     para ser una transformacion valida (sklearn.isotonic.IsotonicRegression ajusta
     y = f(score) monotona no decreciente por minimos cuadrados).

ENTRADA PARCIAL / CERO-NaN
--------------------------
La cascada hereda la imputacion-neutral de ``construir_matriz_caracteristicas`` del
pipeline: X nunca trae NaN (se conserva operativo RandomForest). Una empresa con
datos parciales recibe un P1 y, si cae en la cola, un P2, ambos en [0,1]; su
``P_final`` es por construccion un numero valido en [0,1]. El smoke test lo verifica.

INTEGRACION CON EL PIPELINE (bandera USAR_CASCADA)
--------------------------------------------------
Este modulo NO cambia el default de ninguna bandera. ``script_cinco_optimizado.py``
gana ``USAR_CASCADA = False`` (contrato de no-regresion): con la bandera en False el
PISO se reproduce byte a byte (este modulo ni siquiera se importa). Con la bandera en
True, ``entrenar_escuadron_elite`` y ``proyectar_semaforo_2026`` delegan el CALCULO
DEL SCORE a este modulo, SIN cambiar splits, calibracion, exportacion ni semaforo.
La cascada se combina con ``USAR_FEATURES_ANEXOS`` de forma transparente: si los
anexos estan activos, X ya trae esas columnas y la cascada las usa igual; si no,
funciona sola. FEAT-004 evaluara las variantes (a) solo anexos, (b) solo cascada,
(c) anexos + cascada reutilizando los puntos de entrada de este modulo.

Puntos de entrada para FEAT-004
-------------------------------
  * ``score_cascada_test(X_train, y_train, X_test, ...)`` -> score de Test reservado.
  * ``score_cascada_oof(X, y, ...)`` -> score OOF honesto del panel completo.
  * ``ModeloCascada`` -> clase ajustable/predecible (fit/predict_proba) para la
    proyeccion 2026 (reentrena en el panel completo y puntua empresas nuevas).

Reglas duras respetadas
-----------------------
  * Solo lectura sobre py/ y SIIS/. Este modulo solo escribe en salidas/ (smoke test).
  * Cero fuga en AMBAS etapas: y sale de columnas *_2025 que nunca son features;
    assert anti-fuga sobre las features de la etapa 2. La cola de entrenamiento se
    deriva por OOF (sin fuga del modelo etapa1 sobre sus propias filas).
  * Importar NO dispara entrenamiento (guard __main__). Codigo y comentarios en
    espanol, sin guiones largos.

Notas tecnicas (documentacion citada)
-------------------------------------
* ``sklearn.model_selection.cross_val_predict(method="predict_proba")`` devuelve, por
  muestra, la probabilidad cuando esa muestra estaba FUERA del pliegue de
  entrenamiento (out-of-fold), evitando el optimismo de puntuar datos vistos.
* ``StratifiedKFold(shuffle=True, random_state=42)`` reproduce los mismos pliegues
  que el pipeline (por eso la etapa 1 es comparable al PISO).
* ``sklearn.base.clone`` crea un estimador nuevo sin ajustar con los mismos
  hiperparametros, para no contaminar el estimador base entre pliegues.
* ``numpy.percentile`` define el umbral de la cola como el percentil del score de
  entrenamiento; ``numpy.clip`` acota el score final a [0, 1] por seguridad numerica.
* scale_pos_weight del especialista: razon negativos/positivos del subconjunto en
  riesgo (heuristica estandar para boosting con clases desbalanceadas; ver LightGBM
  ``scale_pos_weight`` y XGBoost ``scale_pos_weight``).
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Pipeline de produccion: utilidades del ensamble y constructores de modelos. Importar
# es seguro porque todo el entrenamiento del pipeline vive bajo su guard __main__.
import script_cinco_optimizado as base

# Semilla global y numero de pliegues compartidos con el pipeline (reproducibilidad).
SEMILLA = base.RANDOM_STATE
N_FOLDS = 5

# ------------------------------------------------------------------------------
# PARAMETROS DE LA CASCADA (todos aprendidos SOLO con entrenamiento / via OOF)
# ------------------------------------------------------------------------------
# PERCENTIL_COLA: tamano de la "cola superior" de la etapa 1 que define el
# subconjunto en-riesgo sobre el que se especializa la etapa 2. Se expresa como el
# PERCENTIL del score de etapa 1 por encima del cual una empresa entra a la cola. Un
# valor de 88.0 toma el top 12% mas riesgoso, suficientemente ancho para contener la
# GRAN MAYORIA de los positivos del entrenamiento (recall alto de la etapa 1) sin
# diluir demasiado al especialista. 12% del panel (~1900 filas de entrenamiento)
# concentra tipicamente > 80% de los 243 positivos de entrenamiento, dejando al
# especialista un subconjunto con un desbalance mucho mas favorable (~1 a 8 en vez de
# ~1 a 65). Es un hiperparametro; se fija aqui y NUNCA se ajusta mirando el Test.
PERCENTIL_COLA = 88.0

RUTA_JSON = base.SALIDAS_DIR / "smoke_modelo_cascada.json"


# ------------------------------------------------------------------------------
# BLOQUE 1: ESCUADRONES (hiperparametros identicos al PISO)
# ------------------------------------------------------------------------------
def _crear_escuadron(scale_pos_weight: float = 7.0, class_weight_pos: float = None) -> dict:
    """
    Devuelve el escuadron de 4 modelos del PISO con sus nombres canonicos (los que
    entiende ``calcular_pesos_ensamble``). Para la ETAPA 1 se usan los valores del
    PISO (scale_pos_weight=7, class_weight pos 8/10) de modo que la etapa 1 sea
    EXACTAMENTE comparable al ensamble plano. Para la ETAPA 2 se inyecta un
    ``scale_pos_weight`` recalculado al desbalance del subconjunto en-riesgo y un
    ``class_weight`` para la clase positiva proporcional, de modo que el especialista
    pondere la clase minoritaria segun el desbalance REAL de la cola (no el global).

    Parametros
    ----------
    scale_pos_weight : float
        Peso de la clase positiva para LightGBM/XGBoost (razon neg/pos del subconjunto).
    class_weight_pos : float | None
        Peso de la clase positiva para HistGradientBoosting/RandomForest. Si es None se
        usan los valores del PISO (8 y 10 respectivamente); si se da, se aplica el
        mismo valor a ambos (recalculo para la etapa 2).
    """
    modelos = {}
    if base.TIENE_LGBM:
        modelos["LightGBM"] = base.LGBMClassifier(
            n_estimators=450, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight, random_state=SEMILLA, verbose=-1, n_jobs=-1)
    if base.TIENE_XGB:
        modelos["XGBoost"] = base.XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
            scale_pos_weight=scale_pos_weight, tree_method="hist", eval_metric="aucpr",
            random_state=SEMILLA, n_jobs=-1)
    cw_hgb = {0: 1, 1: (8 if class_weight_pos is None else class_weight_pos)}
    cw_rf = {0: 1, 1: (10 if class_weight_pos is None else class_weight_pos)}
    modelos["HistGradientBoosting"] = base.HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=25,
        l2_regularization=5.0, class_weight=cw_hgb, random_state=SEMILLA)
    modelos["RandomForest"] = base.RandomForestClassifier(
        n_estimators=500, max_depth=9, min_samples_leaf=8, max_features="sqrt",
        class_weight=cw_rf, random_state=SEMILLA, n_jobs=-1)
    return modelos


def _ajustar_ensamble(modelos: dict, X, y):
    """Ajusta los 4 modelos sobre (X, y). Devuelve el dict de modelos ajustados."""
    for modelo in modelos.values():
        modelo.fit(X, y)
    return modelos


def _puntuar_ensamble(modelos: dict, X) -> np.ndarray:
    """Score del super-ensamble ponderado (pesos 0.30/0.30/0.20/0.20 renormalizados)."""
    pesos = base.calcular_pesos_ensamble(modelos.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    return sum(peso * modelos[nombre].predict_proba(X)[:, 1] for nombre, peso in pesos.items())


def _oof_ensamble(modelos_sin_ajustar: dict, X, y) -> np.ndarray:
    """
    Score OOF del super-ensamble: por cada modelo, probabilidad out-of-fold via
    ``cross_val_predict`` sobre los MISMOS pliegues del pipeline, luego mezcla
    ponderada. Se clona cada modelo para no contaminar el estimador base.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    oofs = {}
    for nombre, modelo in modelos_sin_ajustar.items():
        oofs[nombre] = cross_val_predict(
            clone(modelo), X, y, cv=skf, method="predict_proba")[:, 1]
    pesos = base.calcular_pesos_ensamble(oofs.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    return sum(peso * oofs[nombre] for nombre, peso in pesos.items())


# ------------------------------------------------------------------------------
# BLOQUE 2: COMBINACION DEL SCORE (re-ranking de la cola, monotona-compatible)
# ------------------------------------------------------------------------------
def combinar_scores(p1: np.ndarray, p2: np.ndarray, en_cola: np.ndarray, u_cola: float) -> np.ndarray:
    """
    Combina la probabilidad de etapa 1 (``p1``) y la del especialista de etapa 2
    (``p2``) en UN solo score por empresa en [0, 1], con la estrategia de RE-RANKING
    de la cola descrita en el encabezado del modulo:

      * No-cola: ``u_cola * p1`` (queda en [0, u_cola), por debajo de la cola).
      * Cola   : ``u_cola + (1 - u_cola) * p2`` (queda en [u_cola, 1], por encima).

    ``u_cola`` es el umbral de la cola EXPRESADO EN LA ESCALA DEL SCORE DE ETAPA 1 y
    acotado a [0, 1] (p1 ya es una probabilidad de ensamble en [0,1], asi que su
    percentil tambien vive en [0,1]). ``p2`` solo tiene sentido para las filas de la
    cola; para el resto se ignora. El resultado se clipa a [0, 1] por seguridad.
    """
    u = float(np.clip(u_cola, 0.0, 1.0))
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    en_cola = np.asarray(en_cola, dtype=bool)
    final = np.where(en_cola, u + (1.0 - u) * p2, u * p1)
    return np.clip(final, 0.0, 1.0)


# ------------------------------------------------------------------------------
# BLOQUE 3: CLASE PRINCIPAL DE LA CASCADA (fit / predict_proba)
# ------------------------------------------------------------------------------
class ModeloCascada:
    """
    Clasificador en cascada de 2 etapas con score unico calibrable en [0, 1].

    Uso
    ---
        casc = ModeloCascada(percentil_cola=88.0)
        casc.fit(X_train, y_train)
        score = casc.predict_proba(X_eval)   # score combinado en [0,1]

    Atributos ajustados
    -------------------
    modelos_etapa1_ : dict    escuadron de etapa 1 ajustado sobre (X, y) completo.
    modelos_etapa2_ : dict    escuadron especialista ajustado sobre la cola de train.
    u_cola_         : float    umbral de la cola (percentil del score OOF de etapa 1).
    scale_pos_weight_etapa2_ : float   desbalance recalculado del subconjunto.
    pos_en_cola_    : int      positivos del entrenamiento que caen en la cola (recall
                               de la etapa 1 sobre los positivos; diagnostico).
    """

    def __init__(self, percentil_cola: float = PERCENTIL_COLA):
        self.percentil_cola = float(percentil_cola)
        self.modelos_etapa1_ = None
        self.modelos_etapa2_ = None
        self.u_cola_ = None
        self.scale_pos_weight_etapa2_ = None
        self.pos_en_cola_ = None
        self.n_cola_train_ = None

    def fit(self, X, y):
        """
        Ajusta la cascada:
          1. ETAPA 1: ajusta el escuadron sobre (X, y) y calcula el score OOF de
             entrenamiento (anti-fuga del modelo etapa1 sobre sus propias filas).
          2. Define el umbral de la cola como el percentil ``percentil_cola`` del
             score OOF (SOLO con entrenamiento; nunca mira el Test).
          3. ETAPA 2: recalcula scale_pos_weight al desbalance de la cola y ajusta el
             especialista SOLO sobre las filas de entrenamiento de la cola (pertenencia
             derivada por OOF).
        """
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)

        # ANTI-FUGA (features): ninguna columna puede mencionar 2025 en ninguna etapa.
        assert not any("2025" in str(c) for c in X.columns), \
            "ALERTA CRITICA: fuga de datos (columna 2025) en las features de la cascada."

        # --- ETAPA 1 ---
        self.modelos_etapa1_ = _ajustar_ensamble(_crear_escuadron(), X, y)
        # Score OOF de entrenamiento para fijar la cola SIN fuga del modelo sobre si mismo.
        score_oof_e1 = _oof_ensamble(_crear_escuadron(), X, y)
        self.u_cola_ = float(np.percentile(score_oof_e1, self.percentil_cola))

        # Pertenencia a la cola EN ENTRENAMIENTO derivada del OOF (no del modelo ajustado).
        en_cola_train = score_oof_e1 >= self.u_cola_
        self.n_cola_train_ = int(en_cola_train.sum())
        X_cola = X.loc[en_cola_train].reset_index(drop=True)
        y_cola = y.loc[en_cola_train].reset_index(drop=True)
        self.pos_en_cola_ = int(y_cola.sum())

        # ANTI-FUGA (etapa 2): reafirmar sobre las features del subconjunto especialista.
        assert not any("2025" in str(c) for c in X_cola.columns), \
            "ALERTA CRITICA: fuga de datos (columna 2025) en las features de la ETAPA 2."
        assert self.pos_en_cola_ > 0, (
            "La cola de entrenamiento no contiene ningun positivo; el especialista no "
            "puede entrenarse. Revisar PERCENTIL_COLA (demasiado estricto).")

        # --- ETAPA 2 (especialista) ---
        # scale_pos_weight recalculado al desbalance REAL del subconjunto en-riesgo.
        neg_cola = int((y_cola == 0).sum())
        self.scale_pos_weight_etapa2_ = float(neg_cola / max(self.pos_en_cola_, 1))
        self.modelos_etapa2_ = _ajustar_ensamble(
            _crear_escuadron(
                scale_pos_weight=self.scale_pos_weight_etapa2_,
                class_weight_pos=self.scale_pos_weight_etapa2_,
            ),
            X_cola, y_cola,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Devuelve el score combinado en [0, 1] para cada fila de X (ver
        ``combinar_scores``). La pertenencia a la cola en INFERENCIA se decide con el
        score de etapa 1 ajustado y el umbral ``u_cola_`` aprendido en entrenamiento.
        """
        assert self.modelos_etapa1_ is not None, "ModeloCascada no ajustado; llame a fit()."
        X = pd.DataFrame(X)
        assert not any("2025" in str(c) for c in X.columns), \
            "ALERTA CRITICA: fuga de datos (columna 2025) en inferencia de la cascada."
        p1 = _puntuar_ensamble(self.modelos_etapa1_, X)
        en_cola = p1 >= self.u_cola_
        # El especialista solo aporta sobre la cola; para el resto su valor se ignora.
        # Se puntua TODO X con la etapa 2 (barato) y se usa solo donde en_cola es True.
        p2 = _puntuar_ensamble(self.modelos_etapa2_, X)
        final = combinar_scores(p1, p2, en_cola, self.u_cola_)
        assert np.all((final >= 0.0) & (final <= 1.0)), "Score de cascada fuera de [0,1]."
        return final


# ------------------------------------------------------------------------------
# BLOQUE 4: PUNTOS DE ENTRADA PARA EL PIPELINE Y PARA FEAT-004
# ------------------------------------------------------------------------------
def score_cascada_test(X_train, y_train, X_test, percentil_cola: float = PERCENTIL_COLA) -> np.ndarray:
    """
    GANCHO Test reservado: ajusta la cascada sobre (X_train, y_train) y puntua X_test.
    Reemplaza el score del ensamble plano de ``entrenar_escuadron_elite`` cuando
    USAR_CASCADA esta activo, SIN cambiar splits ni semaforo.
    """
    casc = ModeloCascada(percentil_cola=percentil_cola)
    casc.fit(X_train, y_train)
    return casc.predict_proba(X_test)


def score_cascada_oof(X, y, percentil_cola: float = PERCENTIL_COLA) -> np.ndarray:
    """
    GANCHO OOF honesto del panel completo: score out-of-fold de la CASCADA sobre las
    19937 empresas. Para cada pliegue se ajusta una cascada completa (etapa 1 + cola
    + especialista) en el train del pliegue y se puntua el fold de validacion, de modo
    que ninguna fila se puntua con modelos que la vieron. Es el score OOF que alimenta
    la calibracion isotonica de ``proyectar_semaforo_2026`` cuando USAR_CASCADA activo
    y el que usara FEAT-004 para comparar la cascada contra el PISO en OOF.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        casc = ModeloCascada(percentil_cola=percentil_cola)
        casc.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = casc.predict_proba(X.iloc[va])
    assert np.all((oof >= 0.0) & (oof <= 1.0)), "Score OOF de cascada fuera de [0,1]."
    return oof


# ------------------------------------------------------------------------------
# BLOQUE 5: SMOKE TEST (guard __main__)
# ------------------------------------------------------------------------------
def _metricas_basicas(y: np.ndarray, s: np.ndarray) -> dict:
    """ROC, PR-AUC y cortes top-k% reutilizando el arnes de analisis_apilado."""
    from analisis_apilado_transiciones import metricas as _metricas
    return _metricas(np.asarray(y), np.asarray(s))


def main():
    t_ini = time.time()
    print("=" * 100)
    print("SMOKE TEST: MODELO EN CASCADA DE 2 ETAPAS (FEAT-003)")
    print("=" * 100)
    # La cascada se mide sobre el PISO sin features plurianuales (igual que el publicado).
    assert base.USAR_FEATURES_MULTIANIO is False, \
        "El smoke de la cascada corre sobre el PISO sin features plurianuales."

    # ---- Panel del PISO (2024 -> 2025). Asserts 19937/304, 15949/243, 3988/61 viven
    #      dentro de construir_panel_y_splits (pipeline de produccion).
    X, y, X_train, X_test, y_train, y_test, panel, _df_2025 = base.construir_panel_y_splits()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    print(f"Panel: {len(X):,} empresas x {X.shape[1]} features | positivos={int(y.sum())} | "
          f"train={len(X_train):,}/{int(y_train.sum())} | test={len(X_test):,}/{int(y_test.sum())}")
    print(f"USAR_FEATURES_ANEXOS vigente: {base.USAR_FEATURES_ANEXOS}  |  PERCENTIL_COLA={PERCENTIL_COLA}")

    # ---- CHECKPOINT 1: ajuste de la cascada sobre el entrenamiento, con diagnostico
    #      de la cola y del especialista.
    print("\n[CHECKPOINT 1/4] Ajustando cascada sobre X_train...", flush=True)
    casc = ModeloCascada(percentil_cola=PERCENTIL_COLA)
    casc.fit(X_train, y_train)
    recall_cola = casc.pos_en_cola_ / int(y_train.sum())
    print(f"  Umbral de cola (u_cola, escala score etapa1): {casc.u_cola_:.4f}")
    print(f"  Filas de entrenamiento en la cola           : {casc.n_cola_train_:,} "
          f"({casc.n_cola_train_ / len(X_train):.1%} del train)")
    print(f"  Positivos capturados por la cola (recall e1): {casc.pos_en_cola_}/{int(y_train.sum())} "
          f"({recall_cola:.1%})")
    print(f"  scale_pos_weight recalculado para la etapa 2: {casc.scale_pos_weight_etapa2_:.2f} "
          f"(vs 7.0 del PISO)")

    # ---- CHECKPOINT 2: score de Test reservado.
    print("\n[CHECKPOINT 2/4] Puntuando el Test reservado (3988 empresas)...", flush=True)
    s_test = casc.predict_proba(X_test)
    assert np.all((s_test >= 0.0) & (s_test <= 1.0)), "Score de Test fuera de [0,1]."
    m_test = _metricas_basicas(y_test.to_numpy(), s_test)
    print(f"  TEST -> ROC {m_test['roc_auc']:.4f} | PR {m_test['pr_auc']:.4f} | "
          f"TP top1%: {m_test['cortes']['1.0']['tp']}/{m_test['cortes']['1.0']['alertas']} | "
          f"TP top2.5%: {m_test['cortes']['2.5']['tp']}/{m_test['cortes']['2.5']['alertas']} | "
          f"F1max {m_test['f1_max']['f1']:.4f}")

    # ---- CHECKPOINT 3: score OOF honesto del panel completo (cascada por pliegue).
    print(f"\n[CHECKPOINT 3/4] OOF de {N_FOLDS} pliegues de la cascada (19937 empresas)...", flush=True)
    s_oof = score_cascada_oof(X, y)
    assert np.all((s_oof >= 0.0) & (s_oof <= 1.0)), "Score OOF fuera de [0,1]."
    m_oof = _metricas_basicas(y.to_numpy(), s_oof)
    print(f"  OOF -> ROC {m_oof['roc_auc']:.4f} | PR {m_oof['pr_auc']:.4f} | "
          f"TP top1%: {m_oof['cortes']['1.0']['tp']}/{m_oof['cortes']['1.0']['alertas']} | "
          f"TP top2.5%: {m_oof['cortes']['2.5']['tp']}/{m_oof['cortes']['2.5']['alertas']} | "
          f"F1max {m_oof['f1_max']['f1']:.4f}")

    # ---- CHECKPOINT 4: entrada parcial. Primero el contrato global (X sin NaN y score
    #      finito en [0,1] para toda fila del panel). Luego una empresa de ENTRADA
    #      PARCIAL REAL: se toma una fila real del maestro 2025, se le vacian cuentas de
    #      detalle (las que la ruta de produccion imputa de forma neutral) y se
    #      reconstruye X por la MISMA ruta de produccion (construir_matriz_caracteristicas,
    #      corte t=2025, historia <=2024); la cascada ajustada debe darle un score en
    #      [0,1] sin excepcion. Es exactamente el contrato que pide el usuario.
    print("\n[CHECKPOINT 4/4] Verificacion de entrada parcial (score valido en [0,1])...", flush=True)
    assert X.isnull().sum().sum() == 0, "X trae NaN; la imputacion-neutral deberia evitarlo."
    s_todo = casc.predict_proba(X)
    assert np.all(np.isfinite(s_todo)), "Score de cascada no finito en alguna fila."
    assert np.all((s_todo >= 0.0) & (s_todo <= 1.0)), "Score de cascada fuera de [0,1] en el panel."
    print(f"  Score de cascada para las {len(X):,} empresas del panel: "
          f"min={s_todo.min():.4f}, max={s_todo.max():.4f} (todo en [0,1], sin NaN).")

    # Empresa de entrada parcial reconstruida por la ruta real de produccion.
    cuentas_detalle_a_vaciar = [
        "Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)",
        "Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)",
        "Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)",
        "Gastos de administración (AdministrativeExpense)",
        "Gastos de ventas (DistributionCosts)",
    ]
    df_2025 = pd.read_pickle(base.BASE_DIR / "df_maestro_con_ratios.pkl")
    sanas = df_2025["Estado actual"].isin(["ACTIVA", "EN ETAPA PREOPERATIVA"]) & \
        (df_2025["Flag_Quiebra_Patrimonial"] == 0)
    empresa_parcial = df_2025[sanas].head(1).copy()
    for cuenta in cuentas_detalle_a_vaciar:
        if cuenta in empresa_parcial.columns:
            empresa_parcial[cuenta] = np.nan
    prev_2024 = base.obtener_saldos_periodo_anterior(base.SIIS_DIR, "prev_2024.pkl")
    X_parcial = base.construir_matriz_caracteristicas(
        empresa_parcial, prev_2024, series_multianio=None, anio_t=2025
    )[X.columns]
    assert X_parcial.isnull().sum().sum() == 0, "La reconstruccion parcial produjo NaN en X."
    s_parcial = casc.predict_proba(X_parcial)
    n_faltantes = int(empresa_parcial[cuentas_detalle_a_vaciar].isna().sum(axis=1).iloc[0])
    assert np.all(np.isfinite(s_parcial)) and 0.0 <= float(s_parcial[0]) <= 1.0, \
        "La empresa de entrada parcial no recibio score en [0,1]."
    print(f"  Empresa de ENTRADA PARCIAL ({n_faltantes} cuentas de detalle vaciadas): "
          f"score de cascada = {float(s_parcial[0]):.6f} (en [0,1], sin excepcion).")

    # ---- Persistencia del smoke como evidencia.
    resultado = {
        "descripcion": ("Smoke test del modelo en cascada de 2 etapas (FEAT-003). Mide ROC/PR "
                        "OOF y Test y cortes top1%/top2.5%, y confirma score en [0,1]."),
        "percentil_cola": PERCENTIL_COLA,
        "usar_features_anexos": bool(base.USAR_FEATURES_ANEXOS),
        "diagnostico_cola": {
            "u_cola": round(float(casc.u_cola_), 6),
            "filas_cola_train": int(casc.n_cola_train_),
            "pct_cola_train": round(casc.n_cola_train_ / len(X_train), 4),
            "positivos_en_cola": int(casc.pos_en_cola_),
            "recall_etapa1_sobre_positivos": round(float(recall_cola), 4),
            "scale_pos_weight_etapa2": round(float(casc.scale_pos_weight_etapa2_), 4),
        },
        "test": m_test,
        "oof": m_oof,
        "score_panel_rango": {"min": round(float(s_todo.min()), 6), "max": round(float(s_todo.max()), 6)},
        "entrada_parcial": {
            "cuentas_detalle_vaciadas": n_faltantes,
            "score_cascada": round(float(s_parcial[0]), 6),
            "en_rango_0_1": bool(0.0 <= float(s_parcial[0]) <= 1.0),
        },
        "segundos_totales": round(time.time() - t_ini, 1),
    }
    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {RUTA_JSON.name} ({resultado['segundos_totales']} s).")
    print("Smoke test de la cascada completado con exito: score unico en [0,1] en OOF y Test.")


if __name__ == "__main__":
    main()
