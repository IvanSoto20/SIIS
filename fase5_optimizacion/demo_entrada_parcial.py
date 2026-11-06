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
import script_cinco_optimizado as pipeline
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
from features_multianio import (
    PROFUNDIDAD_HISTORIA,
    construir_features_multianio,
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

# Totales estructurales del balance y nucleo del estado de resultados que, TRAS la
# ampliacion de la imputacion-neutral (HALLAZGO #2), tambien se toleran ausentes: la
# ruta de produccion los satura con ``.fillna(0)`` antes de usarlos, de modo que una
# empresa con balance realmente incompleto recibe igual prediccion. El CASO D vacia
# un subconjunto representativo (efectivo, flujo de operacion, EBIT y costos
# financieros) para DEMOSTRAR que ahora SI se obtiene probabilidad en [0,1] donde
# antes el ``assert X sin NaN`` habria detenido el pipeline.
CUENTAS_ESTRUCTURALES_A_VACIAR_CASO_D = [
    "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)",
    "Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)",
    "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)",
    "Costos financieros (FinanceCosts)",
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


def _predecir(
    df_entrada: pd.DataFrame,
    prev_2024: pd.DataFrame,
    series_multianio,
    modelo: dict,
    forzar_multianio: bool = False,
) -> np.ndarray:
    """
    RUTA DE INFERENCIA REAL: construye X igual que la proyeccion 2026, selecciona las
    columnas ``modelo["features"]`` y aplica el ensamble ponderado + calibrador
    isotonico. Devuelve las probabilidades calibradas (una por fila).

    Es, intencionalmente, la MISMA secuencia de operaciones de
    ``proyectar_semaforo_2026`` para que la demo pruebe el camino de produccion.

    Parametro ``forzar_multianio``
    ------------------------------
    Cuando es True, se activa TEMPORALMENTE la bandera maestra
    ``USAR_FEATURES_MULTIANIO`` del modulo pipeline durante la construccion de X, de
    modo que ``construir_matriz_caracteristicas`` ejecute el BLOQUE E plurianual
    (``construir_features_multianio``) INDEPENDIENTEMENTE del valor de produccion de
    la bandera. Esto permite ejercer de forma explicita la ruta multi-anual (CASO B
    con historia parcial y CASO A sin historia) como pide el HALLAZGO #1, sin cambiar
    el DEFAULT del pipeline (que sigue en False por el gate de no-regresion). El valor
    original se restaura siempre en el ``finally``. Nota: cuando se fuerza el bloque
    plurianual, X trae columnas adicionales que NO estan en ``modelo['features']``
    (el modelo de produccion se entreno con 90 features, sin las plurianuales), por lo
    que para la prediccion se seleccionan solo las columnas conocidas por el modelo;
    el objetivo de la ruta forzada es DEMOSTRAR que el bloque plurianual se construye
    sin excepcion y que la empresa parcial igual obtiene prob en [0,1].
    """
    valor_original = pipeline.USAR_FEATURES_MULTIANIO
    try:
        if forzar_multianio:
            pipeline.USAR_FEATURES_MULTIANIO = True
        # 1) Ingenieria de features (corte t = 2025, historia <=2024). Con la ruta
        #    forzada se ejecuta ademas el bloque plurianual de features_multianio.
        X = construir_matriz_caracteristicas(
            df_entrada, prev_2024, series_multianio=series_multianio, anio_t=2025
        )
    finally:
        pipeline.USAR_FEATURES_MULTIANIO = valor_original

    # 2) Mismo contrato de columnas que X_completo (las features guardadas en el modelo).
    #    Si se forzo el bloque plurianual, X trae columnas extra; se descartan aqui para
    #    predecir con el contrato real del modelo de produccion (90 features).
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


def _inspeccionar_bloque_multianio(
    df_entrada: pd.DataFrame,
    series_multianio,
    saldos_t_cols: dict,
) -> pd.DataFrame:
    """
    Ejerce DIRECTAMENTE ``construir_features_multianio`` (la ruta plurianual) sobre una
    fila de entrada y devuelve el bloque de 11 columnas plurianuales, para poder
    ASEVERAR la cobertura de historia (``Anios_Historia_Disponible`` y las banderas).

    Reutiliza la misma funcion de produccion; no inventa una ruta paralela. ``saldos_t``
    se arma igual que en ``construir_matriz_caracteristicas`` (saldos del corte t ya
    clipeados a >= 0) para alimentar el punto final de las pendientes.
    """
    activos = df_entrada[saldos_t_cols["activos"]].fillna(0).clip(lower=0).astype(float)
    pasivos = df_entrada[saldos_t_cols["pasivos"]].fillna(0).clip(lower=0).astype(float)
    patrimonio = df_entrada[saldos_t_cols["patrimonio"]].fillna(0).clip(lower=0).astype(float)
    ingresos = df_entrada[saldos_t_cols["ingresos"]].fillna(0).clip(lower=0).astype(float)
    utilidad = df_entrada[saldos_t_cols["utilidad"]].fillna(0).astype(float)
    saldos_t = {
        "activos": activos,
        "pasivos": pasivos,
        "patrimonio": patrimonio,
        "ingresos": ingresos,
        "utilidad": utilidad,
    }
    bloque = construir_features_multianio(
        df_maestro_t=df_entrada,
        series_multianio=series_multianio,
        anio_t=2025,
        saldos_t=saldos_t,
    )
    return bloque


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

    # Serie plurianual. A diferencia de la version anterior (que solo la cargaba si la
    # bandera de produccion estaba ON), AHORA la cargamos SIEMPRE porque esta demo debe
    # ejercer EXPLICITAMENTE la ruta multi-anual (HALLAZGO #1), independientemente del
    # valor de produccion de la bandera. El DEFAULT del pipeline NO cambia: sigue en
    # USAR_FEATURES_MULTIANIO=False por el gate del Test; solo la evidencia forzada de
    # esta demo activa el bloque plurianual de forma temporal y acotada.
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
    # Cuatro filas base distintas y reproducibles.
    base = df_2025_sanas.head(4).reset_index(drop=True).copy()

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

    # ==================================================================
    # CASO D: balance realmente incompleto (totales estructurales ausentes).
    # Demuestra el salvavidas de la ampliacion de imputacion-neutral (HALLAZGO #2):
    # antes, un total estructural en NaN detenia el pipeline en el assert; ahora la
    # empresa recibe igual prediccion.
    # ==================================================================
    caso_d = base.iloc[[3]].copy()
    caso_d["Razón social de la sociedad"] = "EMPRESA DEMO D (BALANCE INCOMPLETO, TOTALES AUSENTES)"
    for cuenta in CUENTAS_ESTRUCTURALES_A_VACIAR_CASO_D:
        if cuenta in caso_d.columns:
            caso_d[cuenta] = np.nan

    # ------------------------------------------------------------------
    # Serie plurianual recortada para el CASO B (solo 2023 y 2024 de su NIT).
    # Se construye SIEMPRE (no condicionada a la bandera) porque la evidencia forzada
    # de la ruta multi-anual (abajo) la usa para demostrar historia parcial real.
    # ------------------------------------------------------------------
    series_para_b = {}
    for anio, df_anio in series_multianio.items():
        df_rec = df_anio.copy()
        if anio not in (2023, 2024):
            # Quitamos a este NIT de los anios != {2023, 2024} para dejar historia parcial.
            mask = normalizar_nit(df_rec["NIT"]) != nit_b
            df_rec = df_rec[mask].copy()
        series_para_b[anio] = df_rec

    # ------------------------------------------------------------------
    # INFERENCIA por la ruta real (bandera de produccion, hoy OFF = PISO).
    # Cada caso se pasa por _predecir con la bandera en su valor de produccion.
    # ------------------------------------------------------------------
    resultados = []
    casos = [
        ("A", "cuentas faltantes + sin historia plurianual", caso_a, series_multianio),
        ("B", "historia parcial (solo 2023 y 2024)", caso_b, series_para_b),
        ("C", "datos completos (control)", caso_c, series_multianio),
        ("D", "balance incompleto: totales estructurales ausentes", caso_d, series_multianio),
    ]

    # Columnas crudas de los totales estructurales que la ampliacion ahora tolera en
    # NaN; se usan para contar cuantos totales quedaron faltantes por fila (evidencia).
    cuentas_vigiladas = list(CUENTAS_A_VACIAR_CASO_A) + list(CUENTAS_ESTRUCTURALES_A_VACIAR_CASO_D)

    for etiqueta, descripcion, df_caso, serie_caso in casos:
        score, probs = _predecir(df_caso, prev_2024, serie_caso, modelo)
        score_val = float(score[0])
        prob_val = float(probs[0])
        zona = _zona_desde_score(score_val, modelo["umbrales"])
        # Validacion dura del contrato pedido por el usuario: probabilidad en [0, 1].
        assert 0.0 <= prob_val <= 1.0, f"Probabilidad fuera de [0,1] en caso {etiqueta}: {prob_val}"

        # Cuantas cuentas crudas quedaron faltantes en la fila de entrada (evidencia).
        cuentas_faltantes = [
            c for c in cuentas_vigiladas
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

    # ==================================================================
    # EVIDENCIA EXPLICITA DE LA RUTA MULTI-ANUAL (HALLAZGO #1)
    # ==================================================================
    # La corrida anterior usa la bandera de produccion (hoy OFF), de modo que el bloque
    # plurianual NO se ejecuta y los casos atraviesan el camino identico al PISO. Para
    # dejar EVIDENCIA de que una fila SIN historia (CASO A) y una con HISTORIA PARCIAL
    # (CASO B) atraviesan ``construir_features_multianio`` SIN excepcion y producen una
    # probabilidad en [0,1], aqui se FUERZA la ruta multi-anual (forzar_multianio=True)
    # independientemente del valor de produccion de la bandera. El DEFAULT del pipeline
    # NO cambia; esto es evidencia local y acotada (el valor se restaura en _predecir).
    print("\n" + "-" * 90)
    print("EVIDENCIA FORZADA DE LA RUTA MULTI-ANUAL (construir_features_multianio)")
    print("-" * 90)

    # Columnas crudas de los saldos del corte t para inspeccionar el bloque plurianual.
    saldos_t_cols = {
        "activos": "Total de activos (Assets)",
        "pasivos": "Total pasivos (Liabilities)",
        "patrimonio": "Patrimonio total (Equity)",
        "ingresos": "Ingresos de actividades ordinarias (Revenue)",
        "utilidad": "Ganancia (pérdida) (ProfitLoss)",
    }
    ventana_plena = PROFUNDIDAD_HISTORIA  # anios de la ventana [anio_t-4, anio_t-1]

    evidencia_multianio = []
    casos_ma = [
        ("A", "sin historia plurianual", caso_a, series_multianio, 0, 0),
        ("B", "historia parcial (solo 2023 y 2024)", caso_b, series_para_b, 1, ventana_plena - 1),
    ]
    for etiqueta, descripcion, df_caso, serie_caso, min_esp, max_esp in casos_ma:
        # 1) Inspeccion directa del bloque plurianual (cobertura de historia).
        bloque = _inspeccionar_bloque_multianio(df_caso, serie_caso, saldos_t_cols)
        anios_disp = int(bloque["Anios_Historia_Disponible"].iloc[0])
        flag_sin = int(bloque["Flag_Sin_Historia_3A"].iloc[0])
        flag_parcial = int(bloque["Flag_Historia_Parcial"].iloc[0])
        # El bloque plurianual nunca debe traer NaN (imputacion-neutral + flags).
        assert bloque.isnull().sum().sum() == 0, f"NaN en el bloque plurianual (caso {etiqueta})."

        # 2) Prediccion por la ruta multi-anual FORZADA (bloque E activo) para confirmar
        #    que la fila atraviesa inferencia sin excepcion y produce prob en [0,1].
        score, probs = _predecir(df_caso, prev_2024, serie_caso, modelo, forzar_multianio=True)
        prob_val = float(probs[0])
        assert 0.0 <= prob_val <= 1.0, f"Prob fuera de [0,1] en ruta multi-anual (caso {etiqueta}): {prob_val}"

        # 3) Aserciones de cobertura segun el caso.
        if etiqueta == "A":
            # Sin historia: cero anios disponibles y bandera de sin-historia activa.
            assert anios_disp == 0, f"CASO A deberia tener 0 anios de historia, tiene {anios_disp}."
            assert flag_sin == 1, "CASO A deberia activar Flag_Sin_Historia_3A."
        if etiqueta == "B":
            # HALLAZGO #4: aseverar historia ESTRICTAMENTE parcial (ni ausente ni plena).
            assert 0 < anios_disp < ventana_plena, (
                f"CASO B deberia tener 0 < Anios_Historia_Disponible < {ventana_plena}, "
                f"tiene {anios_disp}."
            )
            assert flag_parcial == 1, "CASO B deberia activar Flag_Historia_Parcial."
            assert flag_sin == 0, "CASO B no deberia activar Flag_Sin_Historia_3A."

        registro_ma = {
            "caso": etiqueta,
            "descripcion": descripcion,
            "ruta": "multi-anual FORZADA (construir_features_multianio ejercido)",
            "anios_historia_disponible": anios_disp,
            "flag_sin_historia_3a": flag_sin,
            "flag_historia_parcial": flag_parcial,
            "ventana_plena": int(ventana_plena),
            "probabilidad_calibrada_quiebra": round(prob_val, 6),
            "probabilidad_en_rango_0_1": bool(0.0 <= prob_val <= 1.0),
            "sin_excepcion": True,
        }
        evidencia_multianio.append(registro_ma)
        print(
            f"\n[RUTA MULTI-ANUAL - CASO {etiqueta}] {descripcion}\n"
            f"  Anios_Historia_Disponible : {anios_disp} (ventana plena={ventana_plena})\n"
            f"  Flag_Sin_Historia_3A      : {flag_sin}\n"
            f"  Flag_Historia_Parcial     : {flag_parcial}\n"
            f"  Prob. calibrada (ruta MA) : {prob_val:.6f} (en [0,1]: {0.0 <= prob_val <= 1.0})\n"
            f"  construir_features_multianio ejercido SIN excepcion: True"
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
            "division_segura) para TODAS las cuentas del corte t ausentes, incluidos los "
            "totales estructurales del balance y el nucleo del estado de resultados "
            "(ampliacion HALLAZGO #2), y flags de cobertura + variacion neutra 0.0 para "
            "la historia plurianual ausente. X queda sin NaN, manteniendo operativo a "
            "RandomForest (que no acepta NaN)."
        ),
        "n_modelos_ensamble": len(modelo["modelos"]),
        "n_features": len(modelo["features"]),
        "umbrales_zona": {k: round(float(v), 6) for k, v in modelo["umbrales"].items()},
        "casos": resultados,
        "evidencia_ruta_multianual_forzada": {
            "proposito": (
                "Ejercer EXPLICITAMENTE la ruta multi-anual (construir_features_multianio) "
                "para una fila SIN historia (CASO A) y una con HISTORIA PARCIAL (CASO B), "
                "forzando el bloque plurianual INDEPENDIENTEMENTE del valor de produccion "
                "de la bandera (que sigue en False por defecto). Demuestra que ambas filas "
                "atraviesan la construccion plurianual sin excepcion y producen prob en [0,1]."
            ),
            "bandera_produccion_sin_cambios": bool(USAR_FEATURES_MULTIANIO),
            "casos": evidencia_multianio,
        },
    }
    ruta_json = SALIDAS_DIR / "demo_entrada_parcial.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # Resumen final: todos los casos deben haber producido probabilidad en [0,1],
    # tanto por la ruta de produccion como por la ruta multi-anual forzada.
    todos_ok = all(r["probabilidad_en_rango_0_1"] for r in resultados)
    todos_ma_ok = all(r["probabilidad_en_rango_0_1"] for r in evidencia_multianio)
    assert todos_ok, "Algun caso no produjo probabilidad en [0,1]."
    assert todos_ma_ok, "Algun caso de la ruta multi-anual no produjo probabilidad en [0,1]."
    print("\n" + "=" * 90)
    print(f"Evidencia guardada en: {ruta_json}")
    print(f"Casos evaluados (ruta produccion): {len(resultados)} | Todos con prob en [0,1]: {todos_ok}")
    print(f"Casos evaluados (ruta multi-anual forzada): {len(evidencia_multianio)} | "
          f"Todos con prob en [0,1]: {todos_ma_ok}")
    print("Demo de entrada parcial completada con exito.")
    print("=" * 90)


if __name__ == "__main__":
    main()
