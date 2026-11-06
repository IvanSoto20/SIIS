# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-002: ¿ES BUENA ESTRATEGIA APILAR TRANSICIONES HISTORICAS?
==============================================================================
Cuello de botella: el panel 2024 -> 2025 tiene solo 304 positivos (61 en el Test),
lo que vuelve ruidosa la zona roja. Palanca evaluada: sumar al entrenamiento las
transiciones 2021->2022, 2022->2023 y 2023->2024 (misma representacion de 90
variables, misma definicion de y_estricto) para multiplicar los positivos.

Este modulo NO cambia el pipeline de produccion: solo MIDE la estrategia y emite una
decision auditable (salidas/analisis_apilado_transiciones.json). Solo lectura sobre
py/ y SIIS/.

Evidencia en tres niveles (de mas a menos decisiva)
---------------------------------------------------
NIVEL 1  Backtest hacia adelante (forward-in-time). Se entrena SOLO con transiciones
         que terminan en 2024 o antes y se evalua en las 19,937 empresas de 2024->2025
         (304 positivos). Ninguna fila de 2025 toca el entrenamiento. Curva de
         aprendizaje con 1, 2 y 3 anios de historia. Replica en 2023->2024.
NIVEL 2  Protocolo del PISO. (a) Test reservado de 3,988 empresas (61 positivos), igual
         al publicado; (b) OOF de 5 pliegues sobre las 19,937 empresas (304 positivos),
         comparando entrenar solo con el panel vs. panel + historia.
NIVEL 3  Control anti-memorizacion. Mismo OOF, pero excluyendo del historico a las
         empresas (NIT) que se estan validando en ese pliegue.

Regla de adopcion (declarada ANTES de ver resultados)
-----------------------------------------------------
Candidato = apilado con peso muestral 1.0 (todos los anios por igual).
  G1  OOF: PR-AUC(candidato) > PR-AUC(base) y ROC-AUC no cae mas de 0.002.
  G2  OOF zona roja: TP en el top 1% y TP en el top 2.5% no empeoran, y al menos uno
      mejora estrictamente.
  G3  Test publicado (61 positivos): ROC, PR-AUC y F1 no caen mas de 0.005 del PISO.
  G4  Version agrupada (sin los NIT validados): PR-AUC OOF sigue superando a la base.
  G5  Backtest hacia adelante: PR-AUC con 3 anios de historia >= PR-AUC con 1 anio.
Se adopta por defecto solo si se cumplen G1 a G5. Si no, queda sin aplicar.

Notas tecnicas (documentacion citada)
-------------------------------------
* sklearn.model_selection.StratifiedKFold(shuffle=True, random_state=42): mismos
  pliegues que ``cross_val_predict`` del pipeline, por eso la base reproduce el OOF.
* ``fit(X, y, sample_weight=w)``: LightGBM, XGBoost, HistGradientBoosting y
  RandomForest aceptan pesos por muestra; ``None`` equivale a peso 1.
* sklearn.metrics.average_precision_score: PR-AUC sin interpolar (preferida con
  desbalance 65 a 1). precision_recall_curve entrega el umbral de maximo F1.
* numpy.random.default_rng + Generator.integers: remuestreo bootstrap reproducible.
* numpy.argpartition: top-k en O(n) sin ordenar todo el vector.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import script_cinco_optimizado as base

# ------------------------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------------------------
SMOKE = os.environ.get("APILADO_SMOKE") == "1"      # prueba rapida del codigo
SEMILLA = base.RANDOM_STATE
ESTADOS_SANOS = ["ACTIVA", "EN ETAPA PREOPERATIVA"]
ANIOS_HIST = (2021, 2022, 2023)
CORTES_PCT = (0.5, 0.7, 1.0, 1.2, 2.5, 5.0, 10.0)
N_BOOT = 40 if SMOKE else 500
N_FOLDS = 2 if SMOKE else 5
TOLERANCIA = 0.005
PISO_TEST = {"roc_auc": 0.8782, "pr_auc": 0.3115, "f1_optimo": 0.4144}
PISO_OOF = {"roc_auc": 0.8867, "pr_auc": 0.2534}

RUTA_CACHE = base.SALIDAS_DIR / ("cache_apilado_smoke.pkl" if SMOKE else "cache_apilado.pkl")
RUTA_JSON = base.SALIDAS_DIR / ("analisis_apilado_smoke.json" if SMOKE else "analisis_apilado_transiciones.json")


# ------------------------------------------------------------------------------
# BLOQUE 1: TRANSICIONES HISTORICAS (misma representacion que el PISO)
# ------------------------------------------------------------------------------
def _cargar_maestro(anio: int) -> pd.DataFrame:
    """Maestro con ratios Fase-4 de un anio: 2021-2023 reconstruidos (FEAT-001),
    2024 y 2025 son los .pkl originales de Fase 4 (solo lectura)."""
    if anio in ANIOS_HIST:
        return pd.read_pickle(base.SALIDAS_DIR / f"maestro_ratios_{anio}.pkl")
    if anio == 2024:
        return pd.read_pickle(base.BASE_DIR / "df_maestro_con_ratios_2024.pkl")
    raise ValueError(f"Anio sin maestro disponible: {anio}")


def construir_transicion(anio_t: int, columnas_piso: list) -> dict:
    """
    Construye la transicion t -> t+1 replicando EXACTAMENTE el diseno del PISO:
      1. Poblacion sana en t (estado ACTIVA/PREOPERATIVA y Flag_Quiebra_Patrimonial == 0).
      2. Cruce inner por NIT con t+1 (``merge(validate="one_to_one")`` falla si hay
         duplicados). Las columnas de t+1 solo aportan la etiqueta, nunca entran en X.
      3. y_estricto = colapso legal en t+1 OR Flag_Quiebra_Patrimonial en t+1.
      4. X con ``construir_matriz_caracteristicas`` usando solo t y su Periodo Anterior
         (t-1), y la misma poda de colinealidad. Las columnas deben coincidir 1 a 1 con
         las del PISO.
    """
    df_t = _cargar_maestro(anio_t)
    df_t1 = _cargar_maestro(anio_t + 1)
    sanas = df_t["Estado actual"].isin(ESTADOS_SANOS) & (df_t["Flag_Quiebra_Patrimonial"] == 0)
    df_sanas = df_t[sanas].copy()
    info = df_t1[["NIT", "Estado actual", "Flag_Quiebra_Patrimonial"]]
    panel = df_sanas.merge(info, on="NIT", how="inner", suffixes=("", "_tp1"), validate="one_to_one")

    colapso_legal = ~panel["Estado actual_tp1"].isin(ESTADOS_SANOS)
    colapso_patrimonial = panel["Flag_Quiebra_Patrimonial_tp1"] == 1
    y = (colapso_legal | colapso_patrimonial).astype(int)

    # Saldos del Periodo Anterior (t-1) de la carpeta del anio t. Se reutiliza la extraccion
    # del pipeline: carga la cache si existe y, si no, la genera desde los anexos (solo lectura).
    prev = base.obtener_saldos_periodo_anterior(
        base.SIIS_DIR / f"pymes_individuales-{anio_t}", f"prev_{anio_t - 1}.pkl")
    X = base.construir_matriz_caracteristicas(
        panel[list(df_sanas.columns)], prev, validar_totales_no_nulos=True
    )
    if base.APLICAR_SELECCION_COLINEALIDAD:
        X = X.drop(columns=[c for c in base.COLUMNAS_COLINEALES_ELIMINADAS if c in X.columns])

    # --- Auditorias (assert) ---
    assert list(X.columns) == list(columnas_piso), f"Columnas de {anio_t} difieren del PISO."
    assert X.isnull().sum().sum() == 0, f"Nulos en X de {anio_t}."
    assert not any(str(c).endswith("_tp1") or str(anio_t + 1) in str(c) for c in X.columns), \
        f"Fuga: columna de t+1 en X ({anio_t})."
    assert panel["NIT"].is_unique and set(np.unique(y)) <= {0, 1}
    assert y.sum() > 0, f"Transicion {anio_t} sin positivos."

    print(f"  Transicion {anio_t}->{anio_t + 1}: sanas_t={len(df_sanas):,} | panel={len(panel):,} | "
          f"positivos={int(y.sum())} ({y.mean():.3%}) "
          f"[legal={int(colapso_legal.sum())}, patrimonial={int(colapso_patrimonial.sum())}]")
    return {
        "X": X.reset_index(drop=True),
        "y": y.reset_index(drop=True),
        "nit": panel["NIT"].reset_index(drop=True),
        "anio": np.full(len(X), anio_t),
        "resumen": {"anio_t": anio_t, "empresas": int(len(panel)), "positivos": int(y.sum()),
                    "tasa_pct": round(float(y.mean() * 100), 4),
                    "positivos_legales": int(colapso_legal.sum()),
                    "positivos_patrimoniales": int(colapso_patrimonial.sum())},
    }


def apilar(transiciones: dict, anios) -> dict:
    """Concatena transiciones de varios anios en una sola tabla (indice reiniciado)."""
    partes = [transiciones[a] for a in anios]
    return {
        "X": pd.concat([p["X"] for p in partes], ignore_index=True),
        "y": pd.concat([p["y"] for p in partes], ignore_index=True),
        "nit": pd.concat([p["nit"] for p in partes], ignore_index=True),
        "anio": np.concatenate([p["anio"] for p in partes]),
    }


# ------------------------------------------------------------------------------
# BLOQUE 2: ENSAMBLE (hiperparametros identicos a entrenar_escuadron_elite)
# ------------------------------------------------------------------------------
def crear_modelos() -> dict:
    """Mismo escuadron del PISO. En SMOKE se reduce el numero de arboles."""
    r = 20 if SMOKE else None
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=r or 450, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
            scale_pos_weight=7, random_state=SEMILLA, verbose=-1, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=r or 500, learning_rate=0.02, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=7,
            tree_method="hist", eval_metric="aucpr", random_state=SEMILLA, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=r or 300, learning_rate=0.03, max_depth=4, min_samples_leaf=25,
            l2_regularization=5.0, class_weight={0: 1, 1: 8}, random_state=SEMILLA),
        "RandomForest": RandomForestClassifier(
            n_estimators=r or 500, max_depth=9, min_samples_leaf=8, max_features="sqrt",
            class_weight={0: 1, 1: 10}, random_state=SEMILLA, n_jobs=-1),
    }


def ajustar_predecir(X_tr, y_tr, X_ev, w=None) -> np.ndarray:
    """Entrena los 4 modelos y devuelve el score del super-ensamble ponderado
    (pesos 0.30/0.30/0.20/0.20 via ``calcular_pesos_ensamble``)."""
    scores = {}
    for nombre, modelo in crear_modelos().items():
        modelo.fit(X_tr, y_tr, sample_weight=w)
        scores[nombre] = modelo.predict_proba(X_ev)[:, 1]
    pesos = base.calcular_pesos_ensamble(scores.keys())
    return sum(p * scores[n] for n, p in pesos.items())


# ------------------------------------------------------------------------------
# BLOQUE 3: METRICAS DE ZONA ROJA Y BOOTSTRAP PAREADO
# ------------------------------------------------------------------------------
def metricas(y: np.ndarray, s: np.ndarray) -> dict:
    """ROC, PR-AUC, tabla de cortes top-k% (k = ceil(n*pct)) y punto de maximo F1."""
    n, P = len(y), int(y.sum())
    tasa = P / n
    acum = np.cumsum(y[np.argsort(-s, kind="stable")])
    cortes = {}
    for pct in CORTES_PCT:
        k = int(np.ceil(n * pct / 100.0))
        tp = int(acum[k - 1])
        cortes[str(pct)] = {"alertas": k, "tp": tp, "precision_pct": round(100 * tp / k, 2),
                            "recall_pct": round(100 * tp / P, 2), "lift": round((tp / k) / tasa, 2)}
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return {
        "n": n, "positivos": P,
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
        "cortes": cortes,
        "f1_max": {"f1": round(float(f1[i]), 4), "precision_pct": round(float(pr[i]) * 100, 2),
                   "recall_pct": round(float(rc[i]) * 100, 2), "umbral": round(float(th[i]), 4),
                   "alertas": int((s >= th[i]).sum())},
    }


def bootstrap_pareado(y: np.ndarray, s_base: np.ndarray, s_nuevo: np.ndarray) -> dict:
    """
    Bootstrap pareado sobre las MISMAS empresas: en cada remuestreo (con reemplazo) se
    calcula la diferencia nuevo - base en 4 KPIs. Devuelve media, IC95% percentil y la
    fraccion de remuestreos con mejora. Cuantifica si la diferencia supera el ruido.
    """
    rng = np.random.default_rng(SEMILLA)
    n = len(y)
    k1, k25 = int(np.ceil(n * 0.01)), int(np.ceil(n * 0.025))

    def kpis(yb, sb):
        t1 = np.argpartition(-sb, k1 - 1)[:k1]
        t25 = np.argpartition(-sb, k25 - 1)[:k25]
        return np.array([average_precision_score(yb, sb), roc_auc_score(yb, sb),
                         yb[t1].mean(), yb[t25].sum() / yb.sum()])

    difs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        difs.append(kpis(yb, s_nuevo[idx]) - kpis(yb, s_base[idx]))
    d = np.array(difs)
    nombres = ["delta_pr_auc", "delta_roc_auc", "delta_precision_top1pct", "delta_recall_top2_5pct"]
    return {nom: {"media": round(float(d[:, i].mean()), 4),
                  "ic95": [round(float(np.percentile(d[:, i], 2.5)), 4),
                           round(float(np.percentile(d[:, i], 97.5)), 4)],
                  "prob_mejora": round(float((d[:, i] > 0).mean()), 3)}
            for i, nom in enumerate(nombres)}


# ------------------------------------------------------------------------------
# BLOQUE 4: PROTOCOLOS DE EVALUACION
# ------------------------------------------------------------------------------
def _cache_get(cache: dict, clave: str, fn):
    """Memoiza resultados costosos en disco para poder reanudar si se interrumpe."""
    if clave not in cache:
        t0 = time.time()
        cache[clave] = fn()
        joblib.dump(cache, RUTA_CACHE)
        print(f"    [{clave}] listo en {time.time() - t0:.0f}s", flush=True)
    return cache[clave]


def entrenar_con_historia(X_tr, y_tr, X_ev, hist, peso_hist, nits_excluir=None):
    """Entrena con (X_tr + historia ponderada). Si ``nits_excluir`` se da, quita de la
    historia esas empresas (variante agrupada anti-memorizacion)."""
    w = None
    if hist is not None:
        Xh, yh, nh = hist["X"], hist["y"], hist["nit"]
        if nits_excluir is not None:
            libres = (~nh.isin(list(nits_excluir))).to_numpy()
            Xh, yh = Xh.loc[libres], yh.loc[libres]
        X_tr = pd.concat([X_tr, Xh], ignore_index=True)
        y_tr = pd.concat([y_tr.reset_index(drop=True), yh], ignore_index=True)
        w = np.concatenate([np.ones(len(X_tr) - len(Xh)), np.full(len(Xh), peso_hist)])
    return ajustar_predecir(X_tr, y_tr, X_ev, w)


def oof_panel(Xp, yp, nit_p, hist, peso_hist, agrupado):
    """OOF de N_FOLDS pliegues sobre el panel 2024->2025 (mismos pliegues que el PISO)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    oof = np.zeros(len(Xp))
    for tr, va in skf.split(Xp, yp):
        excluir = set(nit_p.iloc[va]) if agrupado else None
        oof[va] = entrenar_con_historia(Xp.iloc[tr], yp.iloc[tr], Xp.iloc[va], hist, peso_hist, excluir)
    return oof


def main():
    t_ini = time.time()
    print("=" * 100)
    print(f"ANALISIS DE LA ESTRATEGIA 'APILAR TRANSICIONES HISTORICAS'{' [SMOKE]' if SMOKE else ''}")
    print("=" * 100)
    assert base.USAR_FEATURES_MULTIANIO is False, "Se mide sobre el PISO sin features plurianuales."

    # ---- Panel PISO (2024->2025) y sus aserciones de integridad (19937/304, 15949/243, 3988/61)
    Xp, yp, X_tr, X_te, y_tr, y_te, panel, _ = base.construir_panel_y_splits()
    nit_p = panel["NIT"].reset_index(drop=True)
    Xp, yp = Xp.reset_index(drop=True), yp.reset_index(drop=True)
    columnas = list(Xp.columns)
    yp_np = yp.to_numpy()

    # ---- Transiciones historicas
    print("\nConstruyendo transiciones historicas (misma representacion de 90 variables)...")
    trans = {a: construir_transicion(a, columnas) for a in ANIOS_HIST}
    hist_all = apilar(trans, ANIOS_HIST)
    pos_hist = int(hist_all["y"].sum())
    print(f"  Historia apilada: {len(hist_all['X']):,} filas | {pos_hist} positivos "
          f"(el panel 2024->2025 aporta {int(yp.sum())}) -> x{(pos_hist + yp.sum()) / yp.sum():.2f} positivos")

    cache = joblib.load(RUTA_CACHE) if RUTA_CACHE.exists() else {}
    res = {"transiciones": [trans[a]["resumen"] for a in ANIOS_HIST],
           "positivos_panel_2024_2025": int(yp.sum()), "positivos_historia": pos_hist,
           "factor_positivos_total": round((pos_hist + int(yp.sum())) / int(yp.sum()), 3),
           "smoke": SMOKE}

    # ============== NIVEL 1: BACKTEST HACIA ADELANTE ==============
    print("\n[NIVEL 1] Backtest hacia adelante: evaluacion en 2024->2025 SIN usar ninguna fila de 2025")
    fwd = {}
    for nombre, anios in {"F1_1anio_2023": (2023,), "F2_2anios_2022_2023": (2022, 2023),
                          "F3_3anios_2021_2023": (2021, 2022, 2023)}.items():
        h = apilar(trans, anios)
        s = _cache_get(cache, f"fwd2025::{nombre}", lambda h=h: ajustar_predecir(h["X"], h["y"], Xp))
        fwd[nombre] = s
        res.setdefault("nivel1_eval_2024_2025", {})[nombre] = {
            "anios_entrenamiento": list(anios), "positivos_entrenamiento": int(h["y"].sum()),
            "metricas": metricas(yp_np, s)}
        m = res["nivel1_eval_2024_2025"][nombre]["metricas"]
        print(f"    {nombre:22s} pos_train={int(h['y'].sum()):4d} | ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | "
              f"TP top1%: {m['cortes']['1.0']['tp']}/{m['cortes']['1.0']['alertas']} | F1max {m['f1_max']['f1']:.4f}")
    res["nivel1_bootstrap_vs_F1"] = {
        n: bootstrap_pareado(yp_np, fwd["F1_1anio_2023"], fwd[n]) for n in ("F2_2anios_2022_2023", "F3_3anios_2021_2023")}

    # Replica un anio atras: evaluar en 2023->2024 con historia 2021-2022
    y23 = trans[2023]["y"].to_numpy()
    rep = {}
    for nombre, anios in {"R1_1anio_2022": (2022,), "R2_2anios_2021_2022": (2021, 2022)}.items():
        h = apilar(trans, anios)
        s = _cache_get(cache, f"fwd2024::{nombre}",
                       lambda h=h: ajustar_predecir(h["X"], h["y"], trans[2023]["X"]))
        rep[nombre] = s
        res.setdefault("nivel1_replica_eval_2023_2024", {})[nombre] = {
            "anios_entrenamiento": list(anios), "positivos_entrenamiento": int(h["y"].sum()),
            "metricas": metricas(y23, s)}
        m = res["nivel1_replica_eval_2023_2024"][nombre]["metricas"]
        print(f"    (replica 2023->24) {nombre:20s} | ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | "
              f"TP top1%: {m['cortes']['1.0']['tp']}/{m['cortes']['1.0']['alertas']}")
    res["nivel1_replica_bootstrap"] = bootstrap_pareado(y23, rep["R1_1anio_2022"], rep["R2_2anios_2021_2022"])

    # ============== NIVEL 2a: TEST RESERVADO DEL PISO (61 positivos) ==============
    print("\n[NIVEL 2a] Test reservado del PISO (3,988 empresas, 61 positivos)")
    y_te_np = y_te.to_numpy()
    nits_te = set(nit_p.iloc[X_te.index])
    variantes_split = {
        "V0_base": (None, 1.0, None),
        "V1_apilado_peso1.0": (hist_all, 1.0, None),
        "V2_apilado_peso0.5": (hist_all, 0.5, None),
        "V1g_apilado_agrupado": (hist_all, 1.0, nits_te),
    }
    sp = {}
    res["nivel2a_test_reservado"] = {}
    for nombre, (h, pw, excl) in variantes_split.items():
        s = _cache_get(cache, f"split::{nombre}",
                       lambda h=h, pw=pw, excl=excl: entrenar_con_historia(X_tr, y_tr, X_te, h, pw, excl))
        sp[nombre] = s
        m = metricas(y_te_np, s)
        res["nivel2a_test_reservado"][nombre] = m
        print(f"    {nombre:22s} | ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | F1max {m['f1_max']['f1']:.4f} | "
              f"TP top1%: {m['cortes']['1.0']['tp']}/{m['cortes']['1.0']['alertas']}")
    res["nivel2a_bootstrap_vs_base"] = {n: bootstrap_pareado(y_te_np, sp["V0_base"], sp[n])
                                        for n in ("V1_apilado_peso1.0", "V2_apilado_peso0.5")}

    # ============== NIVEL 2b / 3: OOF 5 PLIEGUES (304 positivos) ==============
    print(f"\n[NIVEL 2b y 3] OOF de {N_FOLDS} pliegues sobre 19,937 empresas (304 positivos)")
    variantes_oof = {
        "V0_base": (None, 1.0, False),
        "V1_apilado_peso1.0": (hist_all, 1.0, False),
        "V1g_apilado_agrupado": (hist_all, 1.0, True),
        "V2_apilado_peso0.5": (hist_all, 0.5, False),
        "V2g_apilado_agrupado": (hist_all, 0.5, True),
    }
    oof = {}
    res["nivel2b_oof"] = {}
    for nombre, (h, pw, agr) in variantes_oof.items():
        s = _cache_get(cache, f"oof::{nombre}",
                       lambda h=h, pw=pw, agr=agr: oof_panel(Xp, yp, nit_p, h, pw, agr))
        oof[nombre] = s
        m = metricas(yp_np, s)
        res["nivel2b_oof"][nombre] = m
        print(f"    {nombre:22s} | ROC {m['roc_auc']:.4f} | PR {m['pr_auc']:.4f} | F1max {m['f1_max']['f1']:.4f} | "
              f"TP top1%: {m['cortes']['1.0']['tp']}/{m['cortes']['1.0']['alertas']} | "
              f"TP top2.5%: {m['cortes']['2.5']['tp']}/{m['cortes']['2.5']['alertas']}", flush=True)
    res["nivel2b_bootstrap_vs_base"] = {n: bootstrap_pareado(yp_np, oof["V0_base"], oof[n])
                                        for n in ("V1_apilado_peso1.0", "V1g_apilado_agrupado",
                                                  "V2_apilado_peso0.5", "V2g_apilado_agrupado")}

    # ============== VALIDEZ DEL ARNES: la base debe reproducir lo publicado ==============
    b_test, b_oof = res["nivel2a_test_reservado"]["V0_base"], res["nivel2b_oof"]["V0_base"]
    res["reproduccion_base"] = {
        "test": {"obtenido": [b_test["roc_auc"], b_test["pr_auc"], b_test["f1_max"]["f1"]],
                 "publicado": [PISO_TEST["roc_auc"], PISO_TEST["pr_auc"], PISO_TEST["f1_optimo"]]},
        "oof": {"obtenido": [b_oof["roc_auc"], b_oof["pr_auc"]],
                "publicado": [PISO_OOF["roc_auc"], PISO_OOF["pr_auc"]]}}
    if not SMOKE:
        assert abs(b_test["roc_auc"] - PISO_TEST["roc_auc"]) < 0.002, "El arnes no reproduce el ROC del PISO."
        assert abs(b_test["pr_auc"] - PISO_TEST["pr_auc"]) < 0.003, "El arnes no reproduce el PR-AUC del PISO."
        assert abs(b_oof["roc_auc"] - PISO_OOF["roc_auc"]) < 0.003, "El arnes no reproduce el OOF-ROC del PISO."
        assert abs(b_oof["pr_auc"] - PISO_OOF["pr_auc"]) < 0.003, "El arnes no reproduce el OOF-PR del PISO."
        print("\n  Arnes validado: la base reproduce los numeros publicados del PISO.")

    # ============== DECISION (regla declarada arriba) ==============
    c_oof, c_g = res["nivel2b_oof"]["V1_apilado_peso1.0"], res["nivel2b_oof"]["V1g_apilado_agrupado"]
    c_te = res["nivel2a_test_reservado"]["V1_apilado_peso1.0"]
    tp1_b, tp25_b = b_oof["cortes"]["1.0"]["tp"], b_oof["cortes"]["2.5"]["tp"]
    tp1_c, tp25_c = c_oof["cortes"]["1.0"]["tp"], c_oof["cortes"]["2.5"]["tp"]
    f_pr1 = res["nivel1_eval_2024_2025"]["F1_1anio_2023"]["metricas"]["pr_auc"]
    f_pr3 = res["nivel1_eval_2024_2025"]["F3_3anios_2021_2023"]["metricas"]["pr_auc"]
    gates = {
        "G1_oof_pr_mejora_y_roc_no_cae": bool(c_oof["pr_auc"] > b_oof["pr_auc"]
                                              and c_oof["roc_auc"] >= b_oof["roc_auc"] - 0.002),
        "G2_zona_roja_oof_no_empeora_y_mejora": bool(tp1_c >= tp1_b and tp25_c >= tp25_b
                                                     and (tp1_c > tp1_b or tp25_c > tp25_b)),
        "G3_test_publicado_dentro_de_tolerancia": bool(
            c_te["roc_auc"] >= PISO_TEST["roc_auc"] - TOLERANCIA
            and c_te["pr_auc"] >= PISO_TEST["pr_auc"] - TOLERANCIA
            and c_te["f1_max"]["f1"] >= PISO_TEST["f1_optimo"] - TOLERANCIA),
        "G4_agrupado_sigue_superando_base": bool(c_g["pr_auc"] > b_oof["pr_auc"]),
        "G5_backtest_3anios_ge_1anio": bool(f_pr3 >= f_pr1),
    }
    res["decision"] = {
        "candidato": "V1_apilado_peso1.0",
        "gates": gates,
        "valores": {"oof_pr_base": b_oof["pr_auc"], "oof_pr_cand": c_oof["pr_auc"],
                    "oof_roc_base": b_oof["roc_auc"], "oof_roc_cand": c_oof["roc_auc"],
                    "tp_top1_base_cand": [tp1_b, tp1_c], "tp_top2_5_base_cand": [tp25_b, tp25_c],
                    "test_cand": [c_te["roc_auc"], c_te["pr_auc"], c_te["f1_max"]["f1"]],
                    "oof_pr_agrupado": c_g["pr_auc"], "fwd_pr_1anio_vs_3anios": [f_pr1, f_pr3]},
        "adoptar_por_defecto": bool(all(gates.values())),
    }
    print("\n" + "=" * 100)
    print("DECISION")
    for g, ok in gates.items():
        print(f"  {'CUMPLE ' if ok else 'FALLA  '} {g}")
    print(f"  ==> ADOPTAR POR DEFECTO: {res['decision']['adoptar_por_defecto']}")
    res["segundos_totales"] = round(time.time() - t_ini, 1)

    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {RUTA_JSON.name} ({res['segundos_totales']} s)")


if __name__ == "__main__":
    main()
