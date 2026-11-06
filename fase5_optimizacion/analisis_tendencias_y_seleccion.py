# -*- coding: utf-8 -*-
"""
FASE 5 - ANÁLISIS DE TENDENCIAS Y SELECCIÓN DE CARACTERÍSTICAS (FEAT-002)
==============================================================================
Objetivo científico de este módulo (SIN modificar la construcción del panel ni
la definición de la variable objetivo ``y`` definidas en ``script_cinco_optimizado``):

  1) ABLACIÓN de las variables de Tendencia Dinámica 2023 -> 2024 para cuantificar
     su aporte marginal al poder predictivo (ROC-AUC, PR-AUC y F1 óptimo del
     Super-Ensamble sobre el Test estratificado reservado, más la OOF del panel
     completo).
  2) SELECCIÓN DE CARACTERÍSTICAS por colinealidad usando correlación de Spearman
     sobre ``X_train`` y arbitraje por importancia de un modelo de árboles.
  3) RE-EVALUACIÓN del subconjunto seleccionado y criterio explícito de conservación.
  4) AFINAMIENTO acotado de ponderación de clases y de pesos del ensamble
     (incluye rank-averaging con ``scipy.stats.rankdata``).
  5) Conservación de la GARANTÍA ANTI-FUGA: ninguna columna de X contiene '2025'.

Todo el análisis REUTILIZA (por import) las funciones ya validadas del pipeline
activo, para no duplicar la lógica de ingeniería de características. Las métricas
de evaluación se miden siempre sobre el mismo split estratificado
(``random_state=42``) construido en el pipeline.

Documentación técnica citada en los comentarios:
  - pandas.DataFrame.corr(method='spearman'): correlación de rangos de Spearman.
  - scipy.stats.rankdata: asignación de rangos (base del rank-averaging).
  - sklearn.ensemble.RandomForestClassifier.feature_importances_: importancia
    por reducción de impureza (Gini) para arbitrar pares colineales.
  - sklearn.metrics.{roc_auc_score, average_precision_score,
    precision_recall_curve}: métricas de ranking y de precisión-cobertura.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# ------------------------------------------------------------------------------
# REUTILIZACIÓN DEL PIPELINE ACTIVO (NO se reimplementa nada de la matriz X ni de y)
# ------------------------------------------------------------------------------
# Importamos las funciones puras del script optimizado. Al tener el bloque pesado
# bajo "if __name__ == '__main__'", este import NO dispara el pipeline completo,
# solo carga las definiciones de funciones y constantes (RANDOM_STATE, SALIDAS_DIR).
import script_cinco_optimizado as base
from script_cinco_optimizado import (
    RANDOM_STATE,
    SALIDAS_DIR,
    TIENE_LGBM,
    LGBMClassifier,
    calcular_pesos_ensamble,
    construir_panel_y_splits,
    entrenar_escuadron_elite,
)


# ------------------------------------------------------------------------------
# CONJUNTO AUTORITATIVO DE COLUMNAS DE TENDENCIA 2023 -> 2024 (BLOQUE 2.C)
# ------------------------------------------------------------------------------
# Estas son EXACTAMENTE las columnas generadas en el bloque 2.C de
# construir_matriz_caracteristicas (tendencias dinámicas t-1 -> t) más las dos
# derivadas sectoriales que dependen directamente de una variable de tendencia.
# IMPORTANTE (decisión de ablación LIMPIA, variante (a) del plan): quitamos SOLO
# estas columnas y NO recalculamos los índices sintéticos Altman_Z_Emergente ni
# Score_Alerta_Temprana, que internamente usan Delta_Endeudamiento y
# Var_Pct_Patrimonio. Justificación: recalcular esos índices cambiaría su
# definición (y por tanto compararíamos modelos sobre features distintas en más
# de una dimensión), contaminando la medición del aporte marginal de las
# tendencias "crudas". Documentamos que el residuo de información de tendencia
# que permanece embebido en los índices sintéticos hace que esta ablación sea una
# COTA INFERIOR conservadora del verdadero aporte de las tendencias.
COLUMNAS_TENDENCIA = [
    "Flag_Sin_Historia_Prev",
    "Delta_Endeudamiento",
    "Var_Pct_Patrimonio",
    "Var_Pct_Ingresos",
    "Delta_ROA_Neto",
    "Flag_Dos_Anios_Perdida",
    "Perdidas_Acumuladas_2A_Sobre_Activos",
    "Flag_Paralisis_Comercial",
    "Var_Pct_Patrimonio_vs_Sector",
    "Delta_Endeudamiento_vs_Sector",
]


def _metricas_ensamble_test(y_test, score_test) -> dict:
    """
    Calcula ROC-AUC, PR-AUC y el F1 óptimo (con su umbral) de un vector de scores
    del ensamble frente a ``y_test``.

    - roc_auc_score: área bajo la curva ROC (capacidad de ordenamiento global).
    - average_precision_score: PR-AUC (promedio de precisión ponderado por el
      salto de recall); métrica preferente bajo desbalanceo extremo (~64:1).
    - precision_recall_curve: devuelve precisión y recall para cada umbral;
      el F1 por umbral se deriva como media armónica y tomamos su máximo.
    """
    roc = float(roc_auc_score(y_test, score_test))
    pr = float(average_precision_score(y_test, score_test))
    precisions, recalls, thresholds = precision_recall_curve(y_test, score_test)
    # f1_scores indexa a los umbrales (precisions/recalls tienen un elemento más).
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = int(np.argmax(f1_scores))
    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "f1_optimo": round(float(f1_scores[idx]), 4),
        "umbral_f1_optimo": round(float(thresholds[idx]), 4),
        "precision_en_f1": round(float(precisions[idx]), 4),
        "recall_en_f1": round(float(recalls[idx]), 4),
    }


def _oof_panel_completo(X, y, columnas=None) -> dict:
    """
    Calcula las métricas Out-Of-Fold (OOF) del Super-Ensamble sobre el panel
    completo mediante validación cruzada estratificada de 5 pliegues
    (StratifiedKFold + cross_val_predict con method='predict_proba'), replicando
    la estrategia de calibración del pipeline. Los pesos del ensamble se calculan
    con ``calcular_pesos_ensamble`` sobre los modelos efectivamente disponibles.

    Reutilizamos las MISMAS especificaciones de modelos que entrenar_escuadron_elite
    reconstruyéndolas vía la función del pipeline, pero aquí solo necesitamos las
    predicciones OOF, por eso instanciamos los modelos a través de un entrenamiento
    ligero por pliegues con cross_val_predict.
    """
    from sklearn.base import clone

    Xc = X if columnas is None else X[columnas]
    # Reconstruimos el diccionario de modelos idéntico al del pipeline usando una
    # llamada controlada: entrenar_escuadron_elite entrena y devuelve los modelos
    # ya ajustados sobre (Xc, y) completo; nos sirven como plantillas a clonar.
    modelos, _, _ = entrenar_escuadron_elite(Xc, y, Xc, y)
    pesos = calcular_pesos_ensamble(modelos.keys())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    score_oof = np.zeros(len(y), dtype=float)
    for nombre, modelo in modelos.items():
        oof = cross_val_predict(clone(modelo), Xc, y, cv=skf, method="predict_proba")[:, 1]
        score_oof += pesos[nombre] * oof

    return {
        "roc_auc": round(float(roc_auc_score(y, score_oof)), 4),
        "pr_auc": round(float(average_precision_score(y, score_oof)), 4),
    }


def ablacion_tendencias(X, y, X_train, X_test, y_train, y_test) -> dict:
    """
    ABLACIÓN DE TENDENCIAS 2023 -> 2024.

    Entrena el Super-Ensamble en el MISMO split estratificado dos veces:
      (a) CON tendencias: X completo.
      (b) SIN tendencias: X menos COLUMNAS_TENDENCIA.
    Reporta ROC-AUC, PR-AUC y F1 óptimo en Test en ambos casos y el delta.
    Añade las métricas OOF del panel completo CON tendencias como referencia de
    estabilidad (la OOF SIN tendencias se omite por costo de cómputo; el aporte
    marginal se mide sobre el Test reservado, que es el criterio del plan).
    """
    presentes = [c for c in COLUMNAS_TENDENCIA if c in X.columns]
    faltantes = [c for c in COLUMNAS_TENDENCIA if c not in X.columns]
    assert not faltantes, f"Columnas de tendencia esperadas no presentes en X: {faltantes}"
    print(f"\n[ABLACIÓN] Columnas de tendencia detectadas ({len(presentes)}): {presentes}")

    # (a) CON tendencias
    print("\n[ABLACIÓN] Escenario CON tendencias (X completo)...")
    _, score_con, _ = entrenar_escuadron_elite(X_train, y_train, X_test, y_test)
    met_con = _metricas_ensamble_test(y_test, score_con)

    # (b) SIN tendencias
    cols_sin = [c for c in X.columns if c not in presentes]
    print(f"\n[ABLACIÓN] Escenario SIN tendencias ({len(cols_sin)} columnas)...")
    _, score_sin, _ = entrenar_escuadron_elite(
        X_train[cols_sin], y_train, X_test[cols_sin], y_test
    )
    met_sin = _metricas_ensamble_test(y_test, score_sin)

    # OOF del panel completo CON tendencias (referencia de estabilidad global).
    print("\n[ABLACIÓN] OOF del panel completo CON tendencias...")
    oof_con = _oof_panel_completo(X, y)

    deltas = {
        "roc_auc": round(met_con["roc_auc"] - met_sin["roc_auc"], 4),
        "pr_auc": round(met_con["pr_auc"] - met_sin["pr_auc"], 4),
        "f1_optimo": round(met_con["f1_optimo"] - met_sin["f1_optimo"], 4),
    }

    resultado = {
        "columnas_tendencia": presentes,
        "n_columnas_total_con": int(X.shape[1]),
        "n_columnas_total_sin": len(cols_sin),
        "test_con_tendencias": met_con,
        "test_sin_tendencias": met_sin,
        "delta_aporte_tendencias_test": deltas,
        "oof_panel_completo_con_tendencias": oof_con,
        "nota_metodologica": (
            "Ablación variante (a): se quitan solo las columnas de tendencia crudas; "
            "los índices sintéticos Altman_Z_Emergente y Score_Alerta_Temprana NO se "
            "recalculan, por lo que el delta es una cota inferior conservadora del "
            "aporte real de las tendencias."
        ),
    }
    return resultado


def seleccion_por_colinealidad(X_train, y_train, umbral_rho=0.95) -> dict:
    """
    SELECCIÓN DE CARACTERÍSTICAS por colinealidad usando Spearman.

    ¿Por qué Spearman y no Pearson?
    --------------------------------
    Pearson mide asociación LINEAL sobre los valores crudos y es muy sensible a
    outliers y a transformaciones no lineales. Nuestros predictores son ratios
    financieros fuertemente acotados con .clip() (p.ej. a [-1, 1], [0, 5], [0, 50])
    y log-transformados (np.log1p), lo que produce distribuciones asimétricas,
    saturadas en los extremos y relaciones monótonas pero NO lineales. La
    correlación de Spearman (pandas corr(method='spearman') / scipy.stats.spearmanr)
    opera sobre los RANGOS, por lo que captura cualquier relación monótona y es
    robusta a outliers y a la saturación por clipping. Es, por tanto, la métrica
    correcta para detectar redundancia entre ratios acotados.

    Procedimiento:
      1) Matriz de Spearman |rho| sobre X_train.
      2) Para cada par con |rho| >= umbral, se elimina la columna de MENOR
         importancia según RandomForestClassifier.feature_importances_ (reducción
         media de impureza de Gini). Un bosque con class_weight balanceado da una
         medida de importancia robusta y barata de calcular.
    """
    # 1) Importancia de árboles como árbitro (barato y robusto).
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    importancia = pd.Series(rf.feature_importances_, index=X_train.columns)

    # 2) Matriz de correlación de Spearman (sobre rangos). Usamos pandas, que
    # internamente rankea cada columna y calcula Pearson sobre los rangos.
    corr = X_train.corr(method="spearman").abs()

    # Recorremos el triángulo superior para no duplicar pares (i, j).
    columnas = list(X_train.columns)
    eliminadas = set()
    pares_colineales = []
    for i in range(len(columnas)):
        for j in range(i + 1, len(columnas)):
            ci, cj = columnas[i], columnas[j]
            rho = corr.iloc[i, j]
            if pd.isna(rho) or rho < umbral_rho:
                continue
            # Elimina la de menor importancia; si ya una está marcada, se salta.
            if ci in eliminadas or cj in eliminadas:
                perdedora = ci if ci in eliminadas else cj
            else:
                perdedora = ci if importancia[ci] <= importancia[cj] else cj
            eliminadas.add(perdedora)
            pares_colineales.append(
                {
                    "col_a": ci,
                    "col_b": cj,
                    "rho_spearman": round(float(rho), 4),
                    "eliminada": perdedora,
                    "importancia_a": round(float(importancia[ci]), 6),
                    "importancia_b": round(float(importancia[cj]), 6),
                }
            )

    conservadas = [c for c in columnas if c not in eliminadas]
    print(
        f"\n[SELECCIÓN] |rho|>={umbral_rho} Spearman: {len(pares_colineales)} pares colineales, "
        f"{len(eliminadas)} columnas eliminadas, {len(conservadas)} conservadas."
    )
    return {
        "metodo_correlacion": "spearman",
        "umbral_rho": umbral_rho,
        "arbitro_importancia": "RandomForestClassifier.feature_importances_ (Gini)",
        "n_total": len(columnas),
        "n_eliminadas": len(eliminadas),
        "n_conservadas": len(conservadas),
        "eliminadas": sorted(eliminadas),
        "conservadas": conservadas,
        "pares_colineales": pares_colineales,
        "justificacion_spearman": (
            "Spearman opera sobre rangos: robusto a outliers y a la saturación por "
            ".clip()/np.log1p de los ratios; capta relaciones monótonas no lineales "
            "que Pearson (lineal) subestimaría."
        ),
    }


def afinamiento_ensamble(X_train, y_train, X_test, y_test, columnas) -> dict:
    """
    AFINAMIENTO ACOTADO (bajo costo): se exploran pocas configuraciones justificadas
    de combinación de probabilidades del ensamble sobre los MISMOS modelos ya
    entrenados (no se re-barren hiperparámetros, lo que respeta el presupuesto):

      - Pesos base del pipeline (referencia).
      - Pesos uniformes (promedio simple de probabilidades).
      - Rank-averaging: se promedian los RANGOS normalizados de cada modelo
        (scipy.stats.rankdata). El rank-averaging es robusto cuando los modelos
        producen probabilidades en escalas distintas (p.ej. RF suele estar mal
        calibrado frente a boosters), porque solo usa el orden relativo.

    Todas las métricas se miden en el Test reservado, nunca re-ajustando sobre él.
    """
    modelos, _, pred_dict = entrenar_escuadron_elite(
        X_train[columnas], y_train, X_test[columnas], y_test
    )
    nombres = list(pred_dict.keys())

    # (1) Pesos base del pipeline.
    pesos_base = calcular_pesos_ensamble(nombres)
    score_base = sum(pesos_base[n] * pred_dict[n] for n in nombres)
    met_base = _metricas_ensamble_test(y_test, score_base)

    # (2) Pesos uniformes.
    w_unif = 1.0 / len(nombres)
    score_unif = sum(w_unif * pred_dict[n] for n in nombres)
    met_unif = _metricas_ensamble_test(y_test, score_unif)

    # (3) Rank-averaging normalizado a [0, 1].
    n = len(y_test)
    score_rank = np.zeros(n, dtype=float)
    for nombre in nombres:
        rangos = rankdata(pred_dict[nombre])  # rangos 1..n
        score_rank += (rangos - 1) / (n - 1)
    score_rank /= len(nombres)
    met_rank = _metricas_ensamble_test(y_test, score_rank)

    return {
        "pesos_base": met_base,
        "pesos_uniformes": met_unif,
        "rank_averaging": met_rank,
        "pesos_base_valores": {k: round(v, 4) for k, v in pesos_base.items()},
    }


def importancias_top_features(X_train, y_train, columnas, top_n=15) -> dict:
    """
    IMPORTANCIA DE CARACTERÍSTICAS (alimenta la sección 4 del informe).

    Genera de forma REPRODUCIBLE el artefacto salidas/importancias_top_features.json
    con dos árbitros complementarios de importancia, ambos ajustados sobre el MISMO
    X_train restringido al set de features vigente (``columnas``) y con
    ``random_state=RANDOM_STATE`` para que el resultado sea determinista:

      - rf_gini_top15: importancia por reducción media de impureza de Gini de un
        ``RandomForestClassifier`` (scikit-learn, ``feature_importances_``).
      - lgbm_gain_top15_pct: ganancia acumulada de LightGBM
        (``booster_.feature_importance(importance_type="gain")``) expresada como
        porcentaje del total. Si LightGBM no está disponible (blindaje de deps),
        la clave queda como lista vacía y se registra el motivo.

    Se escribe con los mismos nombres de clave y formato de pares [nombre, valor]
    que consume el informe, por lo que esta función ES el script fuente versionado
    del artefacto (corrige el hallazgo de revisión #2).
    """
    Xc = X_train[columnas]

    # Árbitro 1: RandomForest (Gini). Mismos hiperparámetros robustos que el
    # arbitraje de colinealidad, para coherencia metodológica.
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(Xc, y_train)
    imp_rf = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values(ascending=False)
    rf_top = [[str(c), round(float(v), 5)] for c, v in imp_rf.head(top_n).items()]

    # Árbitro 2: LightGBM (gain, porcentaje del total). Solo si la librería está.
    lgbm_top = []
    nota_lgbm = "ok"
    if TIENE_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            class_weight={0: 1, 1: 8},
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(Xc, y_train)
        gains = lgbm.booster_.feature_importance(importance_type="gain").astype(float)
        total = gains.sum()
        pct = (gains / total * 100.0) if total > 0 else gains
        imp_lgbm = pd.Series(pct, index=Xc.columns).sort_values(ascending=False)
        lgbm_top = [[str(c), round(float(v), 3)] for c, v in imp_lgbm.head(top_n).items()]
    else:
        nota_lgbm = "LightGBM no disponible en el entorno; sección LGBM omitida (blindaje)."

    resultado = {
        "rf_gini_top15": rf_top,
        "lgbm_gain_top15_pct": lgbm_top,
    }
    if not TIENE_LGBM:
        resultado["nota_lgbm"] = nota_lgbm
    return resultado


def main():
    print("=" * 90)
    print("FEAT-002: ANÁLISIS DE TENDENCIAS 2023->2024 Y SELECCIÓN DE CARACTERÍSTICAS")
    print("=" * 90)

    # PRECONDICIÓN DE REPRODUCIBILIDAD (corrige hallazgo de revisión #1):
    # Este módulo DEBE medir la ablación de tendencias y calcular la selección por
    # colinealidad sobre el SET COMPLETO de 101 columnas, por definición. En el
    # pipeline de producción la bandera base.APLICAR_SELECCION_COLINEALIDAD está en
    # True, lo que haría que construir_panel_y_splits() devolviera X ya podada a 90
    # columnas; re-ejecutar el análisis así correría la ablación/selección sobre 90
    # cols y NO reproduciría los JSON comprometidos (n_total=101). Por eso forzamos
    # la bandera a False SOLO en el contexto de este análisis, antes de construir el
    # panel. No tocamos el valor de producción del pipeline activo.
    base.APLICAR_SELECCION_COLINEALIDAD = False
    print(
        "[REPRO] base.APLICAR_SELECCION_COLINEALIDAD forzada a False para medir sobre "
        "el set COMPLETO (101 columnas); el análisis es auto-consistente y reproducible."
    )

    # --- Carga del panel y splits REUTILIZANDO el pipeline (sin tocar su lógica) ---
    X, y, X_train, X_test, y_train, y_test, panel, df_2025 = construir_panel_y_splits()

    # GARANTÍA ANTI-FUGA (se reafirma aquí explícitamente para FEAT-002):
    # ninguna columna de X puede contener la cadena '2025'. Las tendencias usan
    # exclusivamente saldos 2023 (Periodo Anterior de pymes_individuales-2024) y
    # saldos 2024; la etiqueta y se deriva de columnas *_2025 que NO entran en X.
    assert not any("2025" in str(c) for c in X.columns), "ALERTA: fuga de 2025 en X."
    print(f"\nPanel: {X.shape[0]} empresas x {X.shape[1]} features | quiebras y={int(y.sum())}")
    print("ANTI-FUGA OK: ninguna columna de X contiene '2025'.")

    # ======================= 1) ABLACIÓN DE TENDENCIAS =======================
    res_ablacion = ablacion_tendencias(X, y, X_train, X_test, y_train, y_test)
    with open(SALIDAS_DIR / "analisis_tendencias.json", "w", encoding="utf-8") as f:
        json.dump(res_ablacion, f, indent=2, ensure_ascii=False)
    print("\n[OK] Guardado salidas/analisis_tendencias.json")

    # ==================== 2) SELECCIÓN POR COLINEALIDAD ======================
    res_seleccion = seleccion_por_colinealidad(X_train, y_train, umbral_rho=0.95)

    # ============ 3) RE-EVALUACIÓN DEL SUBCONJUNTO SELECCIONADO =============
    # Criterio de conservación: PR-AUC de Test no cae > 0.01 y F1 óptimo no cae
    # > 0.01 respecto al X completo. Si degrada, se mantiene el set completo.
    met_completo = res_ablacion["test_con_tendencias"]
    conservadas = res_seleccion["conservadas"]
    if res_seleccion["n_eliminadas"] > 0:
        print(f"\n[RE-EVAL] Evaluando subconjunto seleccionado ({len(conservadas)} features)...")
        _, score_sel, _ = entrenar_escuadron_elite(
            X_train[conservadas], y_train, X_test[conservadas], y_test
        )
        met_sel = _metricas_ensamble_test(y_test, score_sel)
    else:
        met_sel = met_completo
        print("\n[RE-EVAL] No hubo columnas colineales por encima del umbral; set completo.")

    caida_pr = round(met_completo["pr_auc"] - met_sel["pr_auc"], 4)
    caida_f1 = round(met_completo["f1_optimo"] - met_sel["f1_optimo"], 4)
    conservar_seleccion = (caida_pr <= 0.01) and (caida_f1 <= 0.01)
    res_seleccion["reevaluacion"] = {
        "metricas_set_completo": met_completo,
        "metricas_set_seleccionado": met_sel,
        "caida_pr_auc": caida_pr,
        "caida_f1_optimo": caida_f1,
        "criterio": "conservar si caida_pr<=0.01 y caida_f1<=0.01",
        "conservar_seleccion": bool(conservar_seleccion),
    }
    with open(SALIDAS_DIR / "seleccion_caracteristicas.json", "w", encoding="utf-8") as f:
        json.dump(res_seleccion, f, indent=2, ensure_ascii=False)
    print(
        f"[RE-EVAL] caida PR-AUC={caida_pr}, caida F1={caida_f1} -> "
        f"{'CONSERVAR selección' if conservar_seleccion else 'MANTENER set completo'}"
    )
    print("[OK] Guardado salidas/seleccion_caracteristicas.json")

    # ==================== 4) AFINAMIENTO DE ENSAMBLE =========================
    # Se afina sobre el mejor set vigente (seleccionado si pasó el criterio, si no completo).
    columnas_vigentes = conservadas if conservar_seleccion else list(X.columns)
    res_afinamiento = afinamiento_ensamble(X_train, y_train, X_test, y_test, columnas_vigentes)
    print("\n[AFINAMIENTO] Comparativa de estrategias de combinación (Test):")
    for k in ("pesos_base", "pesos_uniformes", "rank_averaging"):
        m = res_afinamiento[k]
        print(f"  {k:16s} -> ROC {m['roc_auc']} | PR {m['pr_auc']} | F1 {m['f1_optimo']}")

    # =============== 5) IMPORTANCIA DE FEATURES (sección 4 informe) ===========
    # Se genera de forma reproducible sobre el set de features vigente para que el
    # informe tenga script fuente versionado del artefacto que cita.
    res_importancias = importancias_top_features(X_train, y_train, columnas_vigentes)
    with open(SALIDAS_DIR / "importancias_top_features.json", "w", encoding="utf-8") as f:
        json.dump(res_importancias, f, indent=2, ensure_ascii=False)
    print("\n[OK] Guardado salidas/importancias_top_features.json")

    # ===================== CONSOLIDACIÓN DE HALLAZGOS ========================
    resumen = {
        "ablacion": res_ablacion,
        "seleccion_reevaluacion": res_seleccion["reevaluacion"],
        "afinamiento": res_afinamiento,
        "set_features_vigente": "seleccionado" if conservar_seleccion else "completo",
        "n_features_vigente": len(columnas_vigentes),
    }
    with open(SALIDAS_DIR / "resumen_feat002.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("\n[OK] Guardado salidas/resumen_feat002.json")
    print("\nProceso de análisis FEAT-002 completado con éxito.")
    return resumen


if __name__ == "__main__":
    main()
