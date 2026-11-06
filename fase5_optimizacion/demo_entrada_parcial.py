# -*- coding: utf-8 -*-
"""
DEMO DE ENTRADA PARCIAL (FEAT-006): PREDICCION CON DATOS INCOMPLETOS
==============================================================================
Objetivo de negocio (requisito central del usuario)
----------------------------------------------------
"La idea es que uno agregue su propia empresa con los datos que tenga disponible
(no necesariamente todos) y el modelo prediga para el siguiente anio".

Este script DEMUESTRA, con evidencia reproducible, que una empresa con DATOS
PARCIALES (cuentas contables faltantes y/o sin historia plurianual) recibe de
todas formas una ``Probabilidad_Calibrada_Quiebra`` valida en el intervalo [0, 1]
y su zona del semaforo, SIN que el pipeline lance ninguna excepcion.

Principio metodologico: REUTILIZAR LA RUTA REAL DE PRODUCCION
-------------------------------------------------------------
No se inventa una ruta de inferencia paralela. Se reutilizan EXACTAMENTE las mismas
piezas que ``proyectar_semaforo_2026`` usa en ``script_cinco_optimizado.py``:

  1. ``construir_matriz_caracteristicas(df, prev_2024, series_multianio, anio_t)``
     para construir X (misma ingenieria de features del corte 2025, misma ventana
     de historia <= 2024, mismo anti-fuga).
  2. Se seleccionan EXACTAMENTE las columnas ``modelo["features"]`` guardadas en
     ``salidas/modelo_final_2026.joblib`` (el mismo contrato de columnas que X_completo).
  3. Se aplica el MISMO ensamble ponderado (``calcular_pesos_ensamble`` renormaliza
     a 1.0 sobre los modelos presentes) y la MISMA calibracion isotonica
     (``modelo["calibrador"]``), y los MISMOS umbrales de zona (``modelo["umbrales"]``).

Como se simula la "entrada parcial" (documentado con honestidad)
----------------------------------------------------------------
La ruta de produccion reconstruye X a partir de un DataFrame "maestro" con las
columnas crudas que el pipeline espera (saldos contables, flags de fase 4, CIIU,
razon social, etc.). Para no fabricar una fila sintetica que no represente el
camino real, se TOMAN unas pocas filas REALES del maestro 2025
(``df_maestro_con_ratios.pkl``) y se las MUTILA de forma controlada para construir
tres casos de empresa con datos incompletos:

  * CASO A ("cuentas de detalle faltantes y SIN historia"): a una empresa real se le
    ponen en NaN las cuentas de DETALLE que la ruta de produccion imputa de forma
    neutral (obligaciones vencidas totales y a mas de 360 dias, pasivo DIAN, gastos
    de administracion y de ventas) para emular que el usuario solo aporto el grueso
    del balance y el nucleo del estado de resultados, sin el detalle fino; ADEMAS se
    le asigna un NIT inexistente en la serie plurianual 2020..2024, de modo que NO
    tiene historia multi-anio (todas las features plurianuales caen en
    imputacion-neutral y las banderas de cobertura marcan la ausencia). Es el caso
    mas exigente que la ruta REAL soporta hoy: faltan cuentas de detalle Y falta
    historia. Ver la nota de diseno junto a ``CUENTAS_A_VACIAR_CASO_A`` sobre cual es
    el esqueleto contable minimo que el pipeline exige y por que.

  * CASO B ("historia parcial"): a una empresa real se le conserva su NIT para que
    la serie plurianual le aporte SOLO algunos anios (se restringe la serie a 2023 y
    2024 para este NIT, emulando que el usuario solo tiene los dos ultimos anios).
    Asi ``Anios_Historia_Disponible`` sera > 0 pero < ventana plena, activando
    ``Flag_Historia_Parcial``.

  * CASO C ("empresa completa de referencia"): una empresa real sin mutilar, como
    control para contrastar que la ruta devuelve una probabilidad coherente tanto
    con datos completos como parciales.

Por que esto FUNCIONA pese a los faltantes (fundamento tecnico)
---------------------------------------------------------------
FEAT-005 adopto IMPUTACION-NEUTRAL + FLAGS DE COBERTURA (no el camino NaN-nativo):

  * Las cuentas faltantes del corte t se saturan de forma neutral dentro de
    ``construir_matriz_caracteristicas`` (``.fillna(0)``/``.clip()``/``division_segura``),
    por lo que X nunca contiene NaN (se conserva el ``assert X.isnull().sum().sum()==0``).
  * La ausencia de historia plurianual se captura con banderas explicitas
    (``Flag_Sin_Historia_3A``, ``Flag_Historia_Parcial``, ``Anios_Historia_Disponible``)
    y las tendencias toman variacion neutra 0.0, sin inventar deterioro ni mejora.
  * Como X queda SIN NaN, ``RandomForestClassifier`` de scikit-learn sigue operativo
    (ese modelo lanzaria ``ValueError: Input X contains NaN`` si recibiera nulos),
    de modo que el super-ensamble de 4 modelos devuelve una probabilidad valida.

Nota sobre el manejo nativo de NaN de los boosters (contexto documentado)
-------------------------------------------------------------------------
``HistGradientBoostingClassifier`` (scikit-learn), ``LGBMClassifier`` (LightGBM) y
``XGBClassifier`` (XGBoost) SI saben enrutar NaN de forma nativa en sus splits
(la documentacion de cada libreria lo describe: HGB "native support for missing
values", LightGBM ``use_missing``, XGBoost direccion por defecto para faltantes).
Se eligio de todas formas imputacion-neutral + flags porque: (a) RandomForest,
parte del ensamble, NO acepta NaN; (b) las banderas de cobertura hacen explicito al
modelo cuanta historia real existe, que es mas informativo que un NaN mudo; (c) se
conserva el contrato de cero-NaN de X sin listas blancas, reduciendo el riesgo de
regresion. Esta demo prueba que, con esa decision, el requisito de entrada parcial
se cumple de punta a punta.

Ejecucion
---------
    cd fase5_optimizacion
    /projects/sandbox/.venv_fase5/bin/python demo_entrada_parcial.py

Artefactos de salida
--------------------
    salidas/demo_entrada_parcial.json  (entradas de ejemplo + probabilidad + zona)
    salidas/run_demo_entrada_parcial.log  (log de la corrida, via tee en verification)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Reutilizamos EXACTAMENTE las piezas de produccion. El import no dispara el
# pipeline porque script_cinco_optimizado protege su ejecucion con el guardian
# __main__.
from script_cinco_optimizado import (
    BASE_DIR,
    SALIDAS_DIR,
    SIIS_DIR,
    COL_CIIU,
    USAR_FEATURES_MULTIANIO,
    calcular_pesos_ensamble,
    construir_matriz_caracteristicas,
    normalizar_nit,
    obtener_saldos_periodo_anterior,
)

RUTA_MODELO = SALIDAS_DIR / "modelo_final_2026.joblib"

# Cuentas contables del corte t que "vaciamos" para el CASO A (empresa que solo
# aporto algunas cuentas). HALLAZGO DE DISENO IMPORTANTE (documentado con honestidad):
# la ruta de produccion ``construir_matriz_caracteristicas`` SOLO imputa de forma
# neutral un subconjunto de cuentas de detalle, las que satura explicitamente con
# ``.fillna(0)`` antes de usarlas (obligaciones vencidas, DIAN, gastos de
# administracion y de ventas). En cambio, las cuentas ESTRUCTURALES del estado de
# resultados (ingresos, utilidad, EBIT, costos financieros, flujo de operacion) y
# los totales del balance (activos, pasivos, patrimonio, corrientes, efectivo) NO se
# imputan dentro del pipeline: si llegan en NaN, se propagan y el ``assert X sin NaN``
# detiene la ejecucion. Es decir, hoy el "esqueleto contable minimo" que una empresa
# DEBE aportar para recibir prediccion son esos totales de balance mas el nucleo del
# estado de resultados; el detalle fino de obligaciones y gastos SI puede faltar.
#
# Por eso el CASO A vacia exactamente las cuentas de detalle que el pipeline tolera,
# que es lo que representa fielmente la "entrada parcial" soportada por la ruta REAL.
# Vaciar los totales rompria el pipeline y NO representaria el camino de produccion
# (seria inventar robustez que el codigo no tiene). Esta demostracion documenta con
# precision que se simula y cual es el contrato real de datos minimos.
CUENTAS_A_VACIAR_CASO_A = [
    "Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)",
    "Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)",
    "Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)",
    "Gastos de administración (AdministrativeExpense)",
    "Gastos de ventas (DistributionCosts)",
]


def _zona_desde_score(score: float, umbrales: dict) -> str:
    """
    Reproduce la logica de zonas de ``proyectar_semaforo_2026`` para un score
    escalar, usando los MISMOS umbrales guardados en el modelo (percentiles 99/97.5
    /90 del score OOF del panel completo). Devuelve la etiqueta textual de zona.
    """
    if score >= umbrales["rojo_elite"]:
        return "4. ROJO ELITE (Peligro Inminente)"
    if score >= umbrales["naranja"]:
        return "3. NARANJA (Riesgo Alto)"
    if score >= umbrales["amarillo"]:
        return "2. AMARILLO (Vigilancia / Alerta)"
    return "1. VERDE (Estable / Sano)"


def _predecir(df_entrada: pd.DataFrame, prev_2024: pd.DataFrame, series_multianio, modelo: dict) -> np.ndarray:
    """
    RUTA DE INFERENCIA REAL: construye X igual que la proyeccion 2026, selecciona las
    columnas ``modelo["features"]`` y aplica el ensamble ponderado + calibrador
    isotonico. Devuelve las probabilidades calibradas (una por fila).

    Es, intencionalmente, la MISMA secuencia de operaciones de
    ``proyectar_semaforo_2026`` para que la demo pruebe el camino de produccion.
    """
    # 1) Ingenieria de features identica a la proyeccion (corte t = 2025, historia <=2024).
    X = construir_matriz_caracteristicas(
        df_entrada, prev_2024, series_multianio=series_multianio, anio_t=2025
    )
    # 2) Mismo contrato de columnas que X_completo (las features guardadas en el modelo).
    X = X[modelo["features"]]
    assert X.isnull().sum().sum() == 0, "La ruta de inferencia produjo NaN en X (no deberia)."

    # 3) Mismo ensamble ponderado (pesos renormalizados a 1.0 sobre los modelos presentes).
    modelos = modelo["modelos"]
    pesos = calcular_pesos_ensamble(modelos.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    score = sum(peso * modelos[nombre].predict_proba(X)[:, 1] for nombre, peso in pesos.items())

    # 4) Misma calibracion isotonica. IsotonicRegression(out_of_bounds='clip') acota a [0,1].
    probs = modelo["calibrador"].predict(score)
    return np.asarray(score, dtype=float), np.asarray(probs, dtype=float)


def main() -> None:
    print("=" * 90)
    print("DEMO DE ENTRADA PARCIAL (FEAT-006): prediccion con datos incompletos")
    print("=" * 90)
    print(f"Bandera USAR_FEATURES_MULTIANIO vigente en el pipeline: {USAR_FEATURES_MULTIANIO}")

    # ------------------------------------------------------------------
    # Carga de artefactos de produccion.
    # ------------------------------------------------------------------
    assert RUTA_MODELO.exists(), f"No existe el modelo entrenado: {RUTA_MODELO}"
    modelo = joblib.load(RUTA_MODELO)
    print(f"Modelo cargado: {len(modelo['modelos'])} modelos, {len(modelo['features'])} features, "
          f"calibrador={type(modelo['calibrador']).__name__}.")

    # Saldos 'Periodo Anterior' (2024) usados por la ruta de proyeccion 2025->2026.
    prev_2024 = obtener_saldos_periodo_anterior(SIIS_DIR, "prev_2024.pkl")

    # Serie plurianual solo si la bandera esta activa (igual que produccion). Con la
    # bandera OFF (config vigente/PISO) la serie no se usa y las features plurianuales
    # no entran en X; la demo de entrada parcial sigue siendo valida porque la
    # tolerancia a faltantes del corte t vive en construir_matriz_caracteristicas.
    series_multianio = None
    if USAR_FEATURES_MULTIANIO:
        from datos_multianio import construir_series_saldos_multianio
        series_multianio = construir_series_saldos_multianio(usar_cache=True)

    # ------------------------------------------------------------------
    # Tomamos filas REALES del maestro 2025 (poblacion sana) como base.
    # ------------------------------------------------------------------
    df_2025 = pd.read_pickle(BASE_DIR / "df_maestro_con_ratios.pkl")
    sanas_estados = ["ACTIVA", "EN ETAPA PREOPERATIVA"]
    df_2025_sanas = df_2025[
        df_2025["Estado actual"].isin(sanas_estados)
        & (df_2025["Flag_Quiebra_Patrimonial"] == 0)
    ].copy()
    # Tres filas base distintas y reproducibles.
    base = df_2025_sanas.head(3).reset_index(drop=True).copy()

    # ==================================================================
    # CASO A: cuentas faltantes + SIN historia plurianual.
    # ==================================================================
    caso_a = base.iloc[[0]].copy()
    # NIT inexistente en la serie 2020..2024 => cero historia plurianual. Un sufijo
    # no numerico garantiza que ningun anio de la serie lo contenga.
    caso_a["NIT"] = "DEMO-PARCIAL-SIN-HISTORIA-0001"
    caso_a["Razón social de la sociedad"] = "EMPRESA DEMO A (CUENTAS FALTANTES, SIN HISTORIA)"
    # Vaciamos cuentas no estructurales para emular reporte incompleto. Las dejamos en
    # NaN: construir_matriz_caracteristicas las satura de forma neutral (fillna/clip).
    for cuenta in CUENTAS_A_VACIAR_CASO_A:
        if cuenta in caso_a.columns:
            caso_a[cuenta] = np.nan

    # ==================================================================
    # CASO B: historia parcial (solo 2023 y 2024 para su NIT).
    # ==================================================================
    caso_b = base.iloc[[1]].copy()
    caso_b["Razón social de la sociedad"] = "EMPRESA DEMO B (HISTORIA PARCIAL 2023-2024)"
    nit_b = normalizar_nit(caso_b["NIT"]).iloc[0]

    # ==================================================================
    # CASO C: empresa completa de referencia (control).
    # ==================================================================
    caso_c = base.iloc[[2]].copy()
    caso_c["Razón social de la sociedad"] = "EMPRESA DEMO C (DATOS COMPLETOS, CONTROL)"

    # ------------------------------------------------------------------
    # Serie plurianual recortada para el CASO B (solo 2023 y 2024 de su NIT).
    # Solo aplica si la bandera esta ON; con la bandera OFF la serie no se usa.
    # ------------------------------------------------------------------
    series_para_b = series_multianio
    if USAR_FEATURES_MULTIANIO and series_multianio is not None:
        series_para_b = {}
        for anio, df_anio in series_multianio.items():
            df_rec = df_anio.copy()
            if anio not in (2023, 2024):
                # Quitamos a este NIT de los anios != {2023, 2024} para dejar historia parcial.
                mask = normalizar_nit(df_rec["NIT"]) != nit_b
                df_rec = df_rec[mask].copy()
            series_para_b[anio] = df_rec

    # ------------------------------------------------------------------
    # INFERENCIA por la ruta real. Cada caso se pasa por _predecir.
    # ------------------------------------------------------------------
    resultados = []
    casos = [
        ("A", "cuentas faltantes + sin historia plurianual", caso_a, series_multianio),
        ("B", "historia parcial (solo 2023 y 2024)", caso_b, series_para_b),
        ("C", "datos completos (control)", caso_c, series_multianio),
    ]

    for etiqueta, descripcion, df_caso, serie_caso in casos:
        score, probs = _predecir(df_caso, prev_2024, serie_caso, modelo)
        score_val = float(score[0])
        prob_val = float(probs[0])
        zona = _zona_desde_score(score_val, modelo["umbrales"])
        # Validacion dura del contrato pedido por el usuario: probabilidad en [0, 1].
        assert 0.0 <= prob_val <= 1.0, f"Probabilidad fuera de [0,1] en caso {etiqueta}: {prob_val}"

        # Cuantas cuentas crudas quedaron faltantes en la fila de entrada (evidencia).
        cuentas_faltantes = [
            c for c in CUENTAS_A_VACIAR_CASO_A
            if c in df_caso.columns and pd.isna(df_caso[c].iloc[0])
        ]

        registro = {
            "caso": etiqueta,
            "descripcion": descripcion,
            "nit": str(df_caso["NIT"].iloc[0]),
            "razon_social": str(df_caso["Razón social de la sociedad"].iloc[0]),
            "cuentas_corte_t_faltantes": cuentas_faltantes,
            "num_cuentas_faltantes": len(cuentas_faltantes),
            "score_ensamble": round(score_val, 6),
            "probabilidad_calibrada_quiebra": round(prob_val, 6),
            "probabilidad_en_rango_0_1": bool(0.0 <= prob_val <= 1.0),
            "zona_semaforo": zona,
            "excepcion": None,
        }
        resultados.append(registro)
        print(
            f"\n[CASO {etiqueta}] {descripcion}\n"
            f"  NIT                : {registro['nit']}\n"
            f"  Cuentas faltantes  : {registro['num_cuentas_faltantes']}\n"
            f"  Score ensamble     : {registro['score_ensamble']:.6f}\n"
            f"  Prob. calibrada    : {registro['probabilidad_calibrada_quiebra']:.6f} "
            f"(en [0,1]: {registro['probabilidad_en_rango_0_1']})\n"
            f"  Zona semaforo      : {registro['zona_semaforo']}"
        )

    # ------------------------------------------------------------------
    # Persistencia del resultado como EVIDENCIA.
    # ------------------------------------------------------------------
    salida = {
        "proposito": (
            "Demostrar que una empresa con datos parciales (cuentas faltantes y/o sin "
            "historia plurianual) recibe una probabilidad calibrada en [0,1] por la "
            "MISMA ruta de inferencia de produccion, sin excepciones."
        ),
        "usar_features_multianio": bool(USAR_FEATURES_MULTIANIO),
        "ruta_inferencia": (
            "construir_matriz_caracteristicas -> seleccion modelo['features'] -> "
            "ensamble ponderado (calcular_pesos_ensamble) -> calibrador isotonico -> "
            "zonas por modelo['umbrales']"
        ),
        "estrategia_faltantes": (
            "Imputacion-neutral dentro de construir_matriz_caracteristicas (fillna/clip/"
            "division_segura) para cuentas del corte t ausentes, y flags de cobertura + "
            "variacion neutra 0.0 para la historia plurianual ausente. X queda sin NaN, "
            "manteniendo operativo a RandomForest (que no acepta NaN)."
        ),
        "n_modelos_ensamble": len(modelo["modelos"]),
        "n_features": len(modelo["features"]),
        "umbrales_zona": {k: round(float(v), 6) for k, v in modelo["umbrales"].items()},
        "casos": resultados,
    }
    ruta_json = SALIDAS_DIR / "demo_entrada_parcial.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # Resumen final: todos los casos deben haber producido probabilidad en [0,1].
    todos_ok = all(r["probabilidad_en_rango_0_1"] for r in resultados)
    assert todos_ok, "Algun caso no produjo probabilidad en [0,1]."
    print("\n" + "=" * 90)
    print(f"Evidencia guardada en: {ruta_json}")
    print(f"Casos evaluados: {len(resultados)} | Todos con probabilidad en [0,1]: {todos_ok}")
    print("Demo de entrada parcial completada con exito.")
    print("=" * 90)


if __name__ == "__main__":
    main()
