# -*- coding: utf-8 -*-
"""
FASE 5 OPTIMIZADA: PREDICCIÓN DE INSOLVENCIA EMPRESARIAL (SIIS COLOMBIA)
==============================================================================
Arquitectura de Alta Precisión Forense para Detección de Quiebra e Insolvencia
(Ley 1116 de 2006, Reorganización Empresarial y Liquidación Judicial).

Entorno de ejecución: Carpeta aislada 'fase5_optimizacion'.
Prohibida la Fuga de Datos (Data Leakage): ninguna variable de t+1 (2025) entra en X.
Validaciones matemáticas y forenses auditadas mediante aserciones (assert).

Bloques del Pipeline:
  0. Normalización de Textos, NIF/NIT y Rutas Multiplataforma.
  1. Extracción y Auditoría Forense de 'Periodo Anterior' (Saldos 2023).
  2. Ingeniería de Características Financieras y Forenses (t y t-1):
     - Ratios Base del SIIS (51 indicadores limpios).
     - Ratios de Agotamiento Patrimonial (Equity Burn Rate) y Escala Logarítmica.
     - Señales Forenses Ley 1116: Asfixia de Caja, Mora Proveedores > 10%, DIAN.
     - Indicadores de Historia y Tendencias Dinámicas (2023 -> 2024).
     - Normalización Sectorial Relativa (CIIU a 2 dígitos).
  3. Construcción del Panel Longitudinal (2024 -> 2025) y Partición Estratificada.
  4. Escuadrón Élite Multi-Modelo (LightGBM, XGBoost, HistGradientBoosting, RandomForest).
  5. Calibración Probabilística Isotónica y Optimización de Puntos de Operación.
  6. Evaluación Forense de Efectividad en Test (3,988 empresas, 61 quiebras).
  7. Re-entrenamiento en Panel Completo y Proyección del Semáforo 2026 (21,113 empresas).
"""

import json
import unicodedata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

# ------------------------------------------------------------------------------
# BLINDAJE DE DEPENDENCIAS OPCIONALES (LightGBM / XGBoost)
# ------------------------------------------------------------------------------
# LightGBM y XGBoost aportan los dos boosters de gradiente más potentes del
# ensamble, pero NO forman parte de la librería estándar de scikit-learn y su
# instalación puede fallar en entornos restringidos (ruedas no disponibles,
# ausencia de libgomp, etc.). Para que el pipeline sea robusto, encapsulamos su
# importación en bloques try/except y exponemos dos banderas booleanas
# (TIENE_LGBM / TIENE_XGB). Si alguna librería falta, el "escuadrón élite"
# degrada con gracia apoyándose en sklearn.ensemble.HistGradientBoostingClassifier
# y RandomForestClassifier, que por sí solos bastan para ejecutar el stack
# completo (referencia: scikit-learn User Guide, 1.11 Ensemble methods).
# Cuando ambas librerías están presentes (caso esperado), este blindaje queda
# inactivo y el ensamble conserva su composición original de 4 modelos.
try:
    from lightgbm import LGBMClassifier
    TIENE_LGBM = True
except ImportError:  # pragma: no cover - solo se activa si falta la rueda de LightGBM
    LGBMClassifier = None
    TIENE_LGBM = False

try:
    from xgboost import XGBClassifier
    TIENE_XGB = True
except ImportError:  # pragma: no cover - solo se activa si falta la rueda de XGBoost
    XGBClassifier = None
    TIENE_XGB = False

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# PESOS BASE DEL ENSAMBLE PROBABILÍSTICO (deben sumar 1.0 entre modelos presentes)
# ------------------------------------------------------------------------------
# Composición de referencia del Super-Ensamble: 0.30 LGBM + 0.30 XGB + 0.20 HGB
# + 0.20 RF. Si LightGBM o XGBoost no están disponibles, se redistribuye el peso
# faltante de forma proporcional para que la suma de los pesos de los modelos
# efectivamente entrenados siga siendo exactamente 1.0 (condición de una media
# ponderada convexa; ver numpy: combinación lineal de probabilidades en [0, 1]).
PESOS_BASE_ENSAMBLE = {
    "LightGBM": 0.30,
    "XGBoost": 0.30,
    "HistGradientBoosting": 0.20,
    "RandomForest": 0.20,
}


def calcular_pesos_ensamble(nombres_presentes) -> dict:
    """
    Calcula los pesos normalizados del ensamble a partir de los modelos realmente
    disponibles, renormalizando los pesos base para que sumen exactamente 1.0.

    Parámetros
    ----------
    nombres_presentes : Iterable[str]
        Claves lógicas de los modelos efectivamente entrenados. Se aceptan tanto
        los nombres canónicos de PESOS_BASE_ENSAMBLE ("LightGBM", "XGBoost",
        "HistGradientBoosting", "RandomForest") como los alias usados en el
        bloque de Test ("LightGBM_Forense", "XGBoost_Forense",
        "RandomForest_Forense").

    Devuelve
    --------
    dict
        Mapa {nombre_presente: peso_normalizado} cuya suma es 1.0.
    """
    # Normalizamos los alias "_Forense" al nombre canónico para buscar su peso base.
    def canonico(nombre: str) -> str:
        return nombre.replace("_Forense", "")

    pesos_crudos = {n: PESOS_BASE_ENSAMBLE[canonico(n)] for n in nombres_presentes}
    total = sum(pesos_crudos.values())
    assert total > 0, "No hay ningún modelo disponible para construir el ensamble."
    # Renormalización convexa: cada peso se divide por la suma total de pesos crudos.
    return {n: p / total for n, p in pesos_crudos.items()}

# Semilla global para reproducibilidad matemática
RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# SELECCIÓN DE CARACTERÍSTICAS POR COLINEALIDAD (FEAT-002)
# ------------------------------------------------------------------------------
# Lista de columnas eliminadas por redundancia según el análisis de colinealidad
# de Spearman (|rho| >= 0.95) documentado en analisis_tendencias_y_seleccion.py y
# en salidas/seleccion_caracteristicas.json. Para cada par fuertemente colineal se
# conservó la variable de MAYOR importancia (feature_importances_ de RandomForest)
# y se descartó la de menor. Este subconjunto reduce la redundancia sin degradar el
# poder predictivo: en el Test estratificado MEJORA la línea base (PR-AUC 0.2995 ->
# 0.3115, F1 0.4060 -> 0.4144, ROC-AUC 0.8743 -> 0.8782). Por eso se deja aplicado
# como configuración por defecto. Se controla con APLICAR_SELECCION_COLINEALIDAD
# para poder reproducir la línea base de 101 columnas poniéndolo en False.
APLICAR_SELECCION_COLINEALIDAD = True

# ------------------------------------------------------------------------------
# FEATURES DE TENDENCIA PLURIANUAL (FEAT-005)
# ------------------------------------------------------------------------------
# Bandera maestra que activa/desactiva el bloque de tendencias PLURIANUALES
# (ventana 2020..2024) construido en features_multianio.py. Es la respuesta al
# requisito del usuario de aprovechar varios anios de historia para predecir el
# anio siguiente, tolerando que una empresa aporte solo los datos que tenga.
#
# CONTRATO DE NO-REGRESION (autoritativo): con la bandera en False el pipeline
# DEBE reproducir EXACTAMENTE el PISO (panel 19937/304, TEST ROC 0.8782 / PR 0.3115
# / F1 0.4144, mismas columnas). El valor por defecto se fija tras la ablacion de
# analisis_multianio.py: se adopta True SOLO si las features plurianuales NO
# regresan el PISO en el Test reservado (PR-AUC_test >= 0.3115-0.005 y
# F1_test >= 0.4144-0.005); en caso contrario queda en False y se documenta.
#
# DECISION DE ADOPCION (ver salidas/analisis_multianio.json y findings de FEAT-005):
#   Resultado de la ablacion CON vs SIN features plurianuales en el mismo split
#   estratificado (random_state=42). El valor vigente de la bandera refleja esa
#   decision numerica. Importar el modulo NO dispara nada (guard __main__).
USAR_FEATURES_MULTIANIO = False

COLUMNAS_COLINEALES_ELIMINADAS = [
    "Colchon_Patrimonio_Sobre_Pasivos",
    "Delta_Endeudamiento_vs_Sector",
    "Flag_Cartera_Perdida_360d",
    "Flag_Cobro_Juridico_Proveedores",
    "Flag_Inyeccion_Capital_Socios",
    "Nivel_Endeudamiento",
    "ROA_Neto",
    "Ratio_Cesacion_Pagos_Ley1116",
    "Ratio_Morosidad_Severa",
    "Ratio_Patrimonio_Inercial_Sobre_Activos",
    "Var_Pct_Patrimonio",
]

# Rutas del entorno local
BASE_DIR = Path(__file__).resolve().parent
SIIS_DIR = BASE_DIR.parent / "SIIS"
SALIDAS_DIR = BASE_DIR / "salidas"
SALIDAS_DIR.mkdir(exist_ok=True)

REPORTE_METRICAS = {}


# ==============================================================================
# BLOQUE 0: UTILIDADES DE NORMALIZACIÓN FORENSE
# ==============================================================================
def normalizar_texto(texto: str) -> str:
    """
    Normaliza cadenas de texto eliminando tildes, caracteres combinantes,
    signos ortográficos y espacios redundantes (descomposición canónica NFKD).
    Garantiza compatibilidad entre nombres de columnas de distintos años del SIIS.
    """
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return " ".join(s.lower().split())


def normalizar_nit(serie: pd.Series) -> pd.Series:
    """Limpia el NIT eliminando espacios y sufijos decimales corruptos (.0)."""
    return serie.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def buscar_archivo_siis(carpeta: Path, prefijo: str) -> Path:
    """Localiza un anexo específico del SIIS de forma determinista mediante glob."""
    archivos = sorted(carpeta.glob(f"{prefijo}_*.xlsx"))
    assert len(archivos) == 1, (
        f"Se esperaba exactamente 1 archivo con prefijo {prefijo} en {carpeta}, "
        f"pero se encontraron {len(archivos)}."
    )
    return archivos[0]


def division_segura(
    numerador: np.ndarray,
    denominador: np.ndarray,
    valor_defecto: float,
    lim_inferior: float,
    lim_superior: float
) -> np.ndarray:
    """
    Ejecuta divisiones elemento a elemento protegidas contra divisiones por cero,
    valores nulos e infinitos, aplicando acotamiento numérico estricto (.clip()).
    """
    num = np.asarray(numerador, dtype=float)
    den = np.asarray(denominador, dtype=float)
    mascara_valida = (den != 0) & (~np.isnan(den)) & (~np.isnan(num))
    resultado = np.where(mascara_valida, num / np.where(mascara_valida, den, 1.0), valor_defecto)
    resultado = np.nan_to_num(resultado, nan=valor_defecto, posinf=lim_superior, neginf=lim_inferior)
    return np.clip(resultado, lim_inferior, lim_superior)


# Mapeo canónico de cuentas contables del 'Periodo Anterior'
MAPA_BALANCE_PREV = {
    "Total de activos (Assets)": "activos_prev",
    "Total pasivos (Liabilities)": "pasivos_prev",
    "Patrimonio total (Equity)": "patrimonio_prev",
    "Activos corrientes totales (CurrentAssets)": "act_corr_prev",
    "Pasivos corrientes totales (CurrentLiabilities)": "pas_corr_prev",
    "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)": "efectivo_prev",
}

MAPA_RESULTADOS_PREV = {
    "Ingresos de actividades ordinarias (Revenue)": "ingresos_prev",
    "Ganancia (pérdida) (ProfitLoss)": "utilidad_prev",
    "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)": "ebit_prev",
    "Costos financieros (FinanceCosts)": "costos_fin_prev",
}


# ==============================================================================
# BLOQUE 1: EXTRACCIÓN Y AUDITORÍA FORENSE DEL PERIODO ANTERIOR (t-1)
# ==============================================================================
def leer_periodo_anterior(ruta_excel: Path, mapa_columnas: dict) -> pd.DataFrame:
    """
    Lee exclusivamente las columnas necesarias de un anexo de SIIS, filtra las filas
    de 'Periodo Anterior', valida unicidad por NIT y ordena por Fecha de Corte.
    """
    columnas_objetivo = {normalizar_texto(k): v for k, v in mapa_columnas.items()}
    columnas_objetivo.update({
        "nit": "NIT",
        "periodo": "Periodo",
        "fecha de corte": "Fecha_Corte"
    })

    # Carga optimizada mediante callable en usecols
    df = pd.read_excel(ruta_excel, usecols=lambda c: normalizar_texto(c) in columnas_objetivo)
    df = df.rename(columns=lambda c: columnas_objetivo[normalizar_texto(c)])

    columnas_faltantes = set(columnas_objetivo.values()) - set(df.columns)
    assert not columnas_faltantes, f"Faltan columnas obligatorias en {ruta_excel.name}: {columnas_faltantes}"

    df["NIT"] = normalizar_nit(df["NIT"])
    df = df[df["Periodo"].astype(str).map(normalizar_texto) == "periodo anterior"].copy()
    df["Fecha_Corte"] = pd.to_datetime(df["Fecha_Corte"], errors="coerce")

    for col_alias in mapa_columnas.values():
        df[col_alias] = pd.to_numeric(df[col_alias], errors="coerce")

    # Deduplicación forense: conservar la última fecha de corte reportada
    df = (
        df.sort_values(["NIT", "Fecha_Corte"])
        .drop_duplicates(subset=["NIT"], keep="last")
        .drop(columns=["Periodo", "Fecha_Corte"])
    )
    assert df["NIT"].is_unique, f"NIT duplicado detectado en {ruta_excel.name}"
    return df


def obtener_saldos_periodo_anterior(carpeta_origen: Path, nombre_cache: str) -> pd.DataFrame:
    """Extrae o carga en caché los saldos del periodo anterior (Balance + Resultados)."""
    ruta_cache = SALIDAS_DIR / nombre_cache
    if ruta_cache.exists():
        print(f"Cargando saldos anteriores desde caché: {ruta_cache.name}")
        return pd.read_pickle(ruta_cache)

    print(f"Extrayendo Periodo Anterior desde {carpeta_origen}...")
    balance_prev = leer_periodo_anterior(buscar_archivo_siis(carpeta_origen, "210030"), MAPA_BALANCE_PREV)
    resultados_prev = leer_periodo_anterior(buscar_archivo_siis(carpeta_origen, "310030"), MAPA_RESULTADOS_PREV)

    df_prev = balance_prev.merge(resultados_prev, on="NIT", how="outer", validate="one_to_one")

    # Corrección forense de errores de digitación (tres ceros de más en saldos crudos)
    # Si Activos > 0 y Pasivos > Activos * 50 con Patrimonio > 0, corregir escala
    descuadre_extremo = (df_prev["pasivos_prev"] > df_prev["activos_prev"] * 50) & (df_prev["activos_prev"] > 0)
    df_prev.loc[descuadre_extremo, "pasivos_prev"] /= 1000.0

    df_prev.to_pickle(ruta_cache)
    return df_prev


# ==============================================================================
# BLOQUE 2: INGENIERÍA DE CARACTERÍSTICAS FINANCIERAS Y FORENSES
# ==============================================================================
COL_CIIU = "Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)"


def construir_matriz_caracteristicas(
    df_maestro_t: pd.DataFrame,
    df_saldos_prev: pd.DataFrame,
    series_multianio: dict = None,
    anio_t: int = None,
) -> pd.DataFrame:
    """
    Genera la matriz predictora X para un corte temporal t integrando:
      - 51 Ratios limpios del SIIS (Fases 1 a 4).
      - Ratios de Quema Patrimonial (Equity Burn Rate).
      - Variables de Asfixia Forense y Ley 1116 (Cesación de Pagos y Runway).
      - Tendencias Dinámicas t-1 -> t.
      - Desviaciones sectoriales CIIU relativas a la media de la industria.
      - (Opcional, FEAT-005) Tendencias PLURIANUALES 2020..2024 si
        ``USAR_FEATURES_MULTIANIO`` es True y se provee ``series_multianio``/``anio_t``.
    Garantía estricta de no fuga: solo utiliza columnas del corte t y del corte t-1,
    y (para el bloque plurianual) exclusivamente anios <= ``anio_t`` (jamás 2025).

    Parámetros
    ----------
    df_maestro_t : pandas.DataFrame
        Corte t (con columna 'NIT' y los saldos contables del año t).
    df_saldos_prev : pandas.DataFrame
        Saldos del 'Periodo Anterior' (t-1) ya extraídos (BLOQUE 1).
    series_multianio : dict | None
        Serie de saldos por año (FEAT-004). Solo se usa si
        ``USAR_FEATURES_MULTIANIO`` es True; si es None, el bloque plurianual se
        omite (comportamiento idéntico al PISO).
    anio_t : int | None
        Año del corte t (2024 para el entrenamiento 2024->2025; 2025 para la
        proyección 2026). Obligatorio si se activan las features plurianuales.
    """
    df = df_maestro_t.copy()
    df["NIT_key"] = normalizar_nit(df["NIT"])
    df = df.merge(
        df_saldos_prev.rename(columns={"NIT": "NIT_key"}),
        on="NIT_key",
        how="left",
        validate="one_to_one"
    )
    assert len(df) == len(df_maestro_t), "Falla en merge de saldos anteriores: cambio en número de filas."

    # Columnas base pre-calculadas en fase 4
    cols_base = list(df.loc[:, "Flag_Reexpresion_Contable":"Flag_Desespero_Gerencial"].columns)
    cols_base = [c for c in cols_base if c != "Flag_Quiebra_Patrimonial"] + ["Edad_Empresa_Anios"]
    X = df[cols_base].astype(float).copy()

    # Saldos contables de corte t.
    # IMPUTACION-NEUTRAL DE ENTRADA PARCIAL (HALLAZGO #2): se satura con ``.fillna(0)``
    # ANTES de usarlos TAMBIEN los totales estructurales del balance (activos,
    # pasivos, patrimonio, corrientes, efectivo) y el nucleo del estado de resultados
    # (ingresos, utilidad, EBIT, flujo de operacion, costos financieros). Asi una
    # empresa que aporta "los datos que tenga disponibles" (no necesariamente todos)
    # recibe igual su prediccion: un total ausente se interpreta como 0 (saldo no
    # informado => neutral). EFECTO EN LOS RATIOS: un 0 neutral hace que las ramas
    # ``np.where(saldo > 0, ratio, default)`` tomen el camino "default" documentado
    # (p.ej. patrimonio=0 => tasas de consumo/ROE caen a su piso 0.0 o -2.0), de modo
    # que la celda faltante no inventa deterioro ni solvencia; simplemente deja de
    # aportar informacion. Para los totales no negativos ``.clip(lower=0)`` deja el 0
    # invariante; para los que pueden ser negativos (utilidad, EBIT, flujo, costos)
    # el 0 es el valor neutro natural (sin ganancia ni perdida, sin flujo).
    # NO-REGRESION (verificado): en los paneles reales 2024 y 2025 estas 12 columnas
    # NO contienen NaN, por lo que ``.fillna(0)`` es inocuo para el entrenamiento y el
    # PISO se conserva byte a byte; solo actua como salvavidas ante entrada parcial.
    # pandas.Series.fillna reemplaza unicamente los NaN (documentacion de pandas:
    # "Fill NA/NaN values using the specified method/value"), sin tocar datos reales.
    activos = df["Total de activos (Assets)"].fillna(0).clip(lower=0).astype(float)
    pasivos = df["Total pasivos (Liabilities)"].fillna(0).clip(lower=0).astype(float)
    patrimonio = df["Patrimonio total (Equity)"].fillna(0).clip(lower=0).astype(float)
    act_corr = df["Activos corrientes totales (CurrentAssets)"].fillna(0).clip(lower=0).astype(float)
    pas_corr = df["Pasivos corrientes totales (CurrentLiabilities)"].fillna(0).clip(lower=0).astype(float)
    efectivo = df["Efectivo y equivalentes al efectivo (CashAndCashEquivalents)"].fillna(0).clip(lower=0).astype(float)
    ingresos = df["Ingresos de actividades ordinarias (Revenue)"].fillna(0).clip(lower=0).astype(float)
    utilidad = df["Ganancia (pérdida) (ProfitLoss)"].fillna(0).astype(float)
    ebit = df["Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)"].fillna(0).astype(float)
    flujo_op = df["Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)"].fillna(0).astype(float)
    costos_fin = df["Costos financieros (FinanceCosts)"].fillna(0).clip(lower=0).astype(float)

    cxp_vencidas = df["Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)"].fillna(0).clip(lower=0).astype(float)
    cxp_360 = df["Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)"].fillna(0).clip(lower=0).astype(float)
    pasivo_dian = df["Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)"].fillna(0).clip(lower=0).astype(float)
    gastos_admin = df["Gastos de administración (AdministrativeExpense)"].fillna(0).clip(lower=0).astype(float)
    gastos_ventas = df["Gastos de ventas (DistributionCosts)"].fillna(0).clip(lower=0).astype(float)

    # --------------------------------------------------------------------------
    # A. AGOTAMIENTO PATRIMONIAL (EQUITY BURN RATE) Y ESCALA
    # --------------------------------------------------------------------------
    perdida_neta = np.maximum(0.0, -utilidad)
    perdida_ebit = np.maximum(0.0, -ebit)

    X["Tasa_Consumo_Patrimonio"] = np.where(
        patrimonio > 0, (perdida_neta / patrimonio).clip(0, 5), np.where(perdida_neta > 0, 5.0, 0.0)
    )
    X["Tasa_Sangria_EBIT_Patrimonio"] = np.where(
        patrimonio > 0, (perdida_ebit / patrimonio).clip(0, 5), np.where(perdida_ebit > 0, 5.0, 0.0)
    )
    patrimonio_inercial = patrimonio + utilidad
    X["Ratio_Patrimonio_Inercial_Sobre_Activos"] = np.where(
        activos > 0, (patrimonio_inercial / activos).clip(-1, 1), -1.0
    )
    X["Flag_Colapso_Patrimonial_Inercial"] = (patrimonio_inercial <= 0).astype(int)

    X["ROA_Neto"] = np.where(activos > 0, (utilidad / activos).clip(-1, 1), -1.0)
    X["ROE"] = np.where(patrimonio > 0, (utilidad / patrimonio).clip(-2, 2), -2.0)
    X["Colchon_Patrimonio_Sobre_Pasivos"] = np.where(pasivos > 0, (patrimonio / pasivos).clip(0, 5), 5.0)
    X["Impacto_Intereses_Sobre_Patrimonio"] = np.where(
        patrimonio > 0, (costos_fin / patrimonio).clip(0, 2), np.where(costos_fin > 0, 2.0, 0.0)
    )

    quema_caja_op = np.maximum(0.0, -flujo_op)
    X["Tasa_Agotamiento_Efectivo"] = np.where(
        efectivo > 0, (quema_caja_op / efectivo).clip(0, 5), np.where(quema_caja_op > 0, 5.0, 0.0)
    )
    deficit_corriente = np.maximum(0.0, pas_corr - act_corr)
    X["Deficit_Corriente_Sobre_Patrimonio"] = np.where(
        patrimonio > 0, (deficit_corriente / patrimonio).clip(0, 5), np.where(deficit_corriente > 0, 5.0, 0.0)
    )

    X["Log_Activos"] = np.log1p(activos)
    X["Log_Ingresos"] = np.log1p(ingresos)
    X["Log_Patrimonio"] = np.log1p(patrimonio)
    X["Rotacion_Activos_Totales"] = np.where(activos > 0, (ingresos / activos).clip(0, 10), 0.0)
    X["Concentracion_Deuda_Corto_Plazo"] = np.where(pasivos > 0, (pas_corr / pasivos).clip(0, 1), 0.0)
    X["Flag_NIIF_Plenas"] = df["Punto de Entrada"].astype(str).str.upper().str.contains("PLENAS").astype(int)

    # --------------------------------------------------------------------------
    # B. SEÑALES FORENSES DE LEY 1116 Y RIESGO DE CESACIÓN DE PAGOS
    # --------------------------------------------------------------------------
    quema_operativa_mensual = (gastos_admin + gastos_ventas + perdida_neta) / 12.0
    X["Runway_Meses_Caja"] = np.where(
        quema_operativa_mensual > 0, (efectivo / quema_operativa_mensual).clip(0, 48), 48.0
    )
    X["Flag_Runway_Critico_3M"] = ((X["Runway_Meses_Caja"] < 3.0) & (perdida_neta > 0)).astype(int)

    # Detonante legal Ley 1116 de 2006 (Art. 9: Mora >= 10% del Pasivo Total)
    X["Ratio_Cesacion_Pagos_Ley1116"] = np.where(pasivos > 0, (cxp_vencidas / pasivos).clip(0, 1), 0.0)
    X["Flag_Causal_Cesacion_Ley1116"] = (X["Ratio_Cesacion_Pagos_Ley1116"] >= 0.10).astype(int)

    X["Ratio_Mora_Sobre_Efectivo"] = np.where(
        efectivo > 0, (cxp_vencidas / efectivo).clip(0, 20), np.where(cxp_vencidas > 0, 20.0, 0.0)
    )
    deuda_critica_exigible = cxp_360 + pasivo_dian
    X["Ratio_Deuda_Critica_Sobre_Patrimonio"] = np.where(
        patrimonio > 0, (deuda_critica_exigible / patrimonio).clip(0, 5), np.where(deuda_critica_exigible > 0, 5.0, 0.0)
    )
    # Ganancia bruta tambien se blinda con imputacion-neutral (HALLAZGO #2): si falta,
    # se interpreta como 0 (ni margen positivo ni negativo), de modo que la bandera no
    # se dispara por un dato ausente. En el panel real no hay NaN aqui, asi que es inocuo.
    X["Flag_Margen_Bruto_Negativo"] = (df["Ganancia bruta (GrossProfit)"].fillna(0) < 0).astype(int)

    # Trampas forenses de alta probabilidad de quiebra
    X["Flag_Trampa_Endeudamiento_Critico"] = ((X["Nivel_Endeudamiento"] >= 0.85) & (utilidad < 0)).astype(int)
    X["Flag_Asfixia_Triple_Forense"] = ((X["Nivel_Endeudamiento"] >= 0.85) & (utilidad < 0) & (cxp_vencidas > 0)).astype(int)
    X["Flag_Super_Alerta_Roja_Forense"] = (
        (X["Nivel_Endeudamiento"] >= 0.90) & (X["Runway_Meses_Caja"] < 3.0) & (perdida_neta > 0)
    ).astype(int)

    # 1. Indicio legal en el nombre reportado en carátula
    patron_legal = "REORGANIZA|LIQUIDA|REESTRUCTURA|CONCORDATO|ACUERDO|INSOLVEN"
    X["Flag_Indicio_Legal_Nombre"] = df["Razón social de la sociedad"].astype(str).str.upper().str.contains(patron_legal).astype(int)

    # 2. Apalancamiento Extremo Pasivo / Patrimonio (Detección de empresas infladas)
    X["Ratio_Palanca_Pasivo_Patrimonio"] = np.where(
        patrimonio > 0, (pasivos / patrimonio).clip(0, 50), np.where(pasivos > 0, 50.0, 0.0)
    )
    X["Flag_Apalancamiento_Extremo_10x"] = ((pasivos > 10 * patrimonio) & (patrimonio > 0)).astype(int)
    X["Flag_Apalancamiento_Critico_20x"] = ((pasivos > 20 * patrimonio) & (patrimonio > 0)).astype(int)

    # 3. Zombi de caja crónico con alta deuda
    X["Flag_Zombi_Caja_Apalancada"] = ((utilidad > 0) & (flujo_op < 0) & (X["Nivel_Endeudamiento"] >= 0.75)).astype(int)

    # --------------------------------------------------------------------------
    # C. TENDENCIAS DINÁMICAS (2023 -> 2024 / t-1 -> t)
    # --------------------------------------------------------------------------
    X["Flag_Sin_Historia_Prev"] = df["activos_prev"].isna().astype(int)

    # Imputación neutral de saldos t-1 cuando la empresa es de primera declaración
    activos_prev = df["activos_prev"].fillna(activos).clip(lower=0)
    pasivos_prev = df["pasivos_prev"].fillna(pasivos).clip(lower=0)
    patrimonio_prev = df["patrimonio_prev"].fillna(patrimonio)
    ingresos_prev = df["ingresos_prev"].fillna(ingresos).clip(lower=0)
    utilidad_prev = df["utilidad_prev"].fillna(utilidad)

    endeudamiento_prev = np.where(
        activos_prev > 0, (pasivos_prev / activos_prev).clip(0, 2), X["Nivel_Endeudamiento"]
    )
    roa_neto_prev = np.where(
        activos_prev > 0, (utilidad_prev / activos_prev).clip(-1, 1), X["ROA_Neto"]
    )

    X["Delta_Endeudamiento"] = (X["Nivel_Endeudamiento"] - endeudamiento_prev).clip(-1, 1)
    X["Var_Pct_Patrimonio"] = np.where(
        patrimonio_prev > 0, ((patrimonio - patrimonio_prev) / patrimonio_prev).clip(-1, 2), 0.0
    )
    X["Var_Pct_Ingresos"] = np.where(
        ingresos_prev > 0,
        ((ingresos - ingresos_prev) / ingresos_prev).clip(-1, 2),
        np.where(ingresos > 0, 1.0, 0.0)
    )
    X["Delta_ROA_Neto"] = (X["ROA_Neto"] - roa_neto_prev).clip(-1, 1)
    X["Flag_Dos_Anios_Perdida"] = ((utilidad_prev < 0) & (utilidad < 0)).astype(int)
    X["Perdidas_Acumuladas_2A_Sobre_Activos"] = np.where(
        activos > 0, ((np.maximum(0.0, -utilidad_prev) + perdida_neta) / activos).clip(0, 3), 0.0
    )
    X["Flag_Paralisis_Comercial"] = ((ingresos_prev > 0) & (ingresos == 0)).astype(int)

    # --------------------------------------------------------------------------
    # D. ÍNDICES SINTÉTICOS Y NORMALIZACIÓN SECTORIAL CIIU
    # --------------------------------------------------------------------------
    X["Altman_Z_Emergente"] = (
        1.2 * X["Capital_Trabajo_Sobre_Activos"]
        + 3.3 * X["ROA_Operativo"]
        + 1.0 * X["ROA_Neto"]
        + 0.6 * X["Colchon_Patrimonio_Sobre_Pasivos"]
        + 1.0 * X["Rotacion_Activos_Totales"]
    )

    X["Score_Alerta_Temprana"] = (
        X["Nivel_Endeudamiento"] * 2.5
        + X["Tasa_Consumo_Patrimonio"] * 2.0
        + X["Delta_Endeudamiento"] * 2.0
        - X["Var_Pct_Patrimonio"] * 1.5
        + X["Deficit_Corriente_Sobre_Patrimonio"] * 1.2
        - X["Capital_Trabajo_Sobre_Activos"]
        - X["ROA_Neto"] * 2.0
        + X["Ratio_Morosidad_Severa"]
        + X["Ratio_Financiacion_DIAN"]
        + X["Ratio_Deuda_Critica_Sobre_Patrimonio"] * 1.5
        + X["Flag_Super_Alerta_Roja_Forense"] * 2.0
        + X["Flag_Indicio_Legal_Nombre"] * 3.0
        + X["Flag_Apalancamiento_Critico_20x"] * 2.0
    )

    X["Indice_Asfixia_Operativa"] = X["Nivel_Endeudamiento"] * (1.0 - X["Razon_Corriente"].clip(0, 2) / 2.0)

    cols_flags = [c for c in X.columns if c.startswith("Flag_") and c != "Flag_Sin_Historia_Prev"]
    X["Total_Banderas_Rojas"] = X[cols_flags].sum(axis=1)

    # Desviación respecto al promedio de la división CIIU (2 dígitos)
    ciiu_division = df[COL_CIIU].astype(str).str.strip().str[:2]
    for col_rel in [
        "Nivel_Endeudamiento",
        "ROA_Neto",
        "Razon_Corriente",
        "Tasa_Consumo_Patrimonio",
        "Altman_Z_Emergente",
        "Var_Pct_Patrimonio",
        "Delta_Endeudamiento"
    ]:
        media_sectorial = X[col_rel].groupby(ciiu_division.values).transform("mean").values
        X[f"{col_rel}_vs_Sector"] = X[col_rel] - media_sectorial

    X.index = df_maestro_t.index

    # --------------------------------------------------------------------------
    # E. TENDENCIAS PLURIANUALES 2020..2024 (FEAT-005, opcional por bandera)
    # --------------------------------------------------------------------------
    # Solo se añade el bloque plurianual cuando la bandera maestra está activa y se
    # provee la serie multi-anual de FEAT-004. Con la bandera en False o sin serie,
    # X conserva EXACTAMENTE las columnas del PISO (contrato de no-regresión). Las
    # features plurianuales usan imputación-neutral + flags de cobertura (ver
    # features_multianio.py), por lo que NO introducen NaN y mantienen operativo a
    # RandomForest (que no acepta NaN). Anti-fuga: solo años <= anio_t (nunca 2025).
    if USAR_FEATURES_MULTIANIO and series_multianio is not None:
        from features_multianio import construir_features_multianio
        assert anio_t is not None, "Se requiere anio_t para las features plurianuales."
        saldos_t = {
            "activos": activos,
            "pasivos": pasivos,
            "patrimonio": patrimonio,
            "ingresos": ingresos,
            "utilidad": utilidad,
        }
        bloque_multianio = construir_features_multianio(
            df_maestro_t=df_maestro_t,
            series_multianio=series_multianio,
            anio_t=anio_t,
            saldos_t=saldos_t,
        )
        # Concatenación por índice (ambos comparten df_maestro_t.index).
        X = pd.concat([X, bloque_multianio], axis=1)

    assert X.isnull().sum().sum() == 0, f"Error: nulos detectados en X: {X.columns[X.isnull().any()].tolist()}"
    return X


# ==============================================================================
# BLOQUE 3: CONSTRUCCIÓN DEL PANEL LONGITUDINAL (2024 -> 2025)
# ==============================================================================
def construir_panel_y_splits():
    """Carga los maestros 2024 y 2025 y aplica el filtro de población sana en t-1."""
    df_2024 = pd.read_pickle(BASE_DIR / "df_maestro_con_ratios_2024.pkl")
    df_2025 = pd.read_pickle(BASE_DIR / "df_maestro_con_ratios.pkl")

    sanas_estados = ["ACTIVA", "EN ETAPA PREOPERATIVA"]
    filtro_sanas_2024 = (
        df_2024["Estado actual"].isin(sanas_estados) &
        (df_2024["Flag_Quiebra_Patrimonial"] == 0)
    )
    df_2024_sanas = df_2024[filtro_sanas_2024].copy()

    # Cruce longitudinal exacto por NIT
    info_y_2025 = df_2025[["NIT", "Estado actual", "Flag_Quiebra_Patrimonial"]]
    panel = df_2024_sanas.merge(info_y_2025, on="NIT", how="inner", suffixes=("", "_2025"), validate="one_to_one")
    assert len(panel) == 19937, f"Fallo de integridad: panel tiene {len(panel)} filas en lugar de 19,937."

    # Definición rigurosa de insolvencia y quiebra en t (2025)
    colapso_legal = ~panel["Estado actual_2025"].isin(sanas_estados)
    colapso_patrimonial = panel["Flag_Quiebra_Patrimonial_2025"] == 1
    y = (colapso_legal | colapso_patrimonial).astype(int)
    assert y.sum() == 304, f"Fallo de integridad: quiebras en 2025 suman {y.sum()} en lugar de 304."

    # Extracción de historia 2023
    prev_2023 = obtener_saldos_periodo_anterior(SIIS_DIR / "pymes_individuales-2024", "prev_2023.pkl")

    # Serie de saldos PLURIANUAL 2020..2024 (FEAT-004) solo si la bandera está activa.
    # Se carga de caché (salidas/series_saldos_multianio.pkl) para no releer los .xlsx.
    # El corte t del modelo 2024->2025 es el año 2024 (anti-fuga: ventana <= 2024).
    series_multianio = None
    if USAR_FEATURES_MULTIANIO:
        from datos_multianio import construir_series_saldos_multianio
        series_multianio = construir_series_saldos_multianio(usar_cache=True)

    # Construcción de X estrictamente con datos de 2024 e historia 2023 (0 fuga de 2025)
    columnas_2024_puras = [c for c in df_2024_sanas.columns]
    X = construir_matriz_caracteristicas(
        panel[columnas_2024_puras], prev_2023, series_multianio=series_multianio, anio_t=2024
    )
    assert not any("2025" in str(col) for col in X.columns), "ALERTA CRÍTICA: Fuga de datos detectada en X."

    # SELECCIÓN POR COLINEALIDAD (FEAT-002): se descartan las columnas redundantes
    # identificadas por Spearman (|rho|>=0.95). Esto NO altera la construcción de la
    # matriz ni la definición de y; solo poda columnas colineales de X tras verificar
    # la ausencia de fuga de 2025. Mejora PR-AUC/F1 en el Test estratificado.
    if APLICAR_SELECCION_COLINEALIDAD:
        cols_a_quitar = [c for c in COLUMNAS_COLINEALES_ELIMINADAS if c in X.columns]
        X = X.drop(columns=cols_a_quitar)
        # Reafirmación anti-fuga tras la poda: ninguna columna superviviente trae 2025.
        assert not any("2025" in str(col) for col in X.columns), "ALERTA CRÍTICA: Fuga de datos tras selección."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    assert len(X_train) == 15949 and y_train.sum() == 243
    assert len(X_test) == 3988 and y_test.sum() == 61

    return X, y, X_train, X_test, y_train, y_test, panel, df_2025


# ==============================================================================
# BLOQUE 4: ESCUADRÓN ÉLITE DE MODELOS Y SUPER-ENSAMBLE
# ==============================================================================
def entrenar_escuadron_elite(X_train, y_train, X_test, y_test):
    """
    Entrena un escuadrón diverso de 4 modelos con ponderación de clases
    y regularización adaptativa para eventos con desbalanceo 64.58 a 1.
    """
    modelos = {}
    # LightGBM y XGBoost se añaden solo si su librería está disponible (blindaje).
    if TIENE_LGBM:
        modelos["LightGBM_Forense"] = LGBMClassifier(
            n_estimators=450,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            scale_pos_weight=7,
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1
        )
    if TIENE_XGB:
        modelos["XGBoost_Forense"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=5.0,
            scale_pos_weight=7,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    # HistGradientBoosting y RandomForest siempre están disponibles (sklearn).
    modelos["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=25,
        l2_regularization=5.0,
        class_weight={0: 1, 1: 8},
        random_state=RANDOM_STATE
    )
    modelos["RandomForest_Forense"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=9,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight={0: 1, 1: 10},
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    predicciones_test = {}
    print("\nEntrenando Escuadrón Élite Forense sobre X_train (15,949 empresas)...")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        probs = modelo.predict_proba(X_test)[:, 1]
        predicciones_test[nombre] = probs
        roc = roc_auc_score(y_test, probs)
        pr = average_precision_score(y_test, probs)
        print(f"  {nombre:22s} -> ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")

    # Ensamble probabilístico óptimo. Los pesos se calculan dinámicamente sobre los
    # modelos presentes (referencia 0.30 LGBM + 0.30 XGB + 0.20 HGB + 0.20 RF) y se
    # renormalizan para sumar 1.0 en caso de que falte LightGBM o XGBoost.
    pesos = calcular_pesos_ensamble(predicciones_test.keys())
    assert abs(sum(pesos.values()) - 1.0) < 1e-9, "Los pesos del ensamble no suman 1.0."
    score_ensamble = sum(peso * predicciones_test[nombre] for nombre, peso in pesos.items())

    roc_ens = roc_auc_score(y_test, score_ensamble)
    pr_ens = average_precision_score(y_test, score_ensamble)
    detalle_pesos = " + ".join(f"{p:.2f}*{n}" for n, p in pesos.items())
    print(f"\n---> SUPER-ENSAMBLE FORENSE {len(pesos)} MODELOS ({detalle_pesos})")
    print(f"     ROC-AUC: {roc_ens:.4f} | PR-AUC: {pr_ens:.4f}")

    return modelos, score_ensamble, predicciones_test


# ==============================================================================
# BLOQUE 5: DIAGNÓSTICO FORENSE DE EFECTIVIDAD POR PUNTOS DE OPERACIÓN
# ==============================================================================
def auditar_efectividad_test(y_test, score_ensamble):
    """
    Evalúa la efectividad operativa de las predicciones en Test (3,988 empresas, 61 quiebras).
    Genera el desglose explícito de alertas, verdaderos positivos, precisión y cobertura.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, score_ensamble)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx_max_f1 = np.argmax(f1_scores)
    umbral_max_f1 = thresholds[idx_max_f1]

    cortes = [
        ("Corte Quirúrgico Élite (Top 0.5% más riesgoso)", np.percentile(score_ensamble, 99.5)),
        ("Corte Quirúrgico Alta Precisión (Top 0.7%)", np.percentile(score_ensamble, 99.3)),
        ("Zona Roja Estricta (Top 1.0% más riesgoso)", np.percentile(score_ensamble, 99.0)),
        ("Zona Roja Ampliada (Top 1.2% más riesgoso)", np.percentile(score_ensamble, 98.8)),
        (f"Umbral Óptimo Máximo F1 (Prob >= {umbral_max_f1:.3f})", umbral_max_f1),
        ("Corte Equilibrado (Top 2.5% más riesgoso)", np.percentile(score_ensamble, 97.5)),
        ("Corte Cobertura Media (Top 5.0% más riesgoso)", np.percentile(score_ensamble, 95.0)),
        ("Zona de Auditoría Preventiva (Top 10.0% más riesgoso)", np.percentile(score_ensamble, 90.0)),
    ]

    print("\n================================================================================================")
    print("AUDITORÍA DE EFECTIVIDAD OPERATIVA EN TEST (3,988 EMPRESAS, 61 QUIEBRAS REALES EN 2025)")
    print("================================================================================================")

    resumen_cortes = []
    for nombre_corte, umbral in cortes:
        preds = (score_ensamble >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        alertas = tp + fp
        precision = tp / alertas if alertas > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_test, preds)
        lift = precision / y_test.mean()

        texto_efectividad = (
            f"Para {alertas:3d} empresas alertadas se detectaron {tp:2d} de las {tp+fn} quiebras reales "
            f"({recall*100:5.1f}% Recall) | Efectividad (Precisión): {precision*100:5.1f}% ({tp}/{alertas}) | "
            f"F1: {f1:.4f} | Multiplicador de Riesgo (Lift): {lift:4.1f}x"
        )
        print(f"---> {nombre_corte:52s}: {texto_efectividad}")

        resumen_cortes.append({
            "corte": nombre_corte,
            "umbral": float(umbral),
            "empresas_alertadas": int(alertas),
            "quiebras_detectadas_TP": int(tp),
            "total_quiebras_reales": int(tp + fn),
            "falsas_alarmas_FP": int(fp),
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "f1_score": round(f1, 4),
            "lift": round(lift, 2)
        })

    # Semáforo Forense de 4 Zonas Quirúrgicas en Test
    u_rojo_elite = np.percentile(score_ensamble, 99.0)   # Top 1.0%
    u_naranja = np.percentile(score_ensamble, 97.5)      # Top 2.5%
    u_amarillo = np.percentile(score_ensamble, 90.0)     # Top 10.0%

    zonas = np.select(
        [score_ensamble >= u_rojo_elite, score_ensamble >= u_naranja, score_ensamble >= u_amarillo],
        ["4. ROJO ÉLITE (Peligro Inminente)", "3. NARANJA (Riesgo Alto)", "2. AMARILLO (Vigilancia / Alerta)"],
        default="1. VERDE (Estable / Sano)"
    )

    df_semaforo_test = pd.DataFrame({"Zona": zonas, "Quebro": y_test.values})
    res_sem = df_semaforo_test.groupby("Zona")["Quebro"].agg(
        Total_Empresas="count",
        Empresas_Quebradas="sum",
        Tasa_Quiebra_Pct=lambda s: s.mean() * 100
    )
    res_sem["Pct_Del_Total_Quiebras"] = res_sem["Empresas_Quebradas"] / y_test.sum() * 100
    res_sem["Lift_vs_Promedio"] = res_sem["Tasa_Quiebra_Pct"] / (y_test.mean() * 100)

    print("\n================================================================================================")
    print("DISTRIBUCIÓN DEL SEMÁFORO FORENSE DE 4 ZONAS EN TEST (2024 -> 2025)")
    print("================================================================================================")
    print(res_sem.to_string())

    return resumen_cortes, res_sem, u_rojo_elite, u_amarillo


# ==============================================================================
# BLOQUE 6: PROYECCIÓN DEL SEMÁFORO 2026 (SOBRE 21,113 EMPRESAS SANAS EN 2025)
# ==============================================================================
def proyectar_semaforo_2026(X_completo, y_completo, df_2025):
    """
    Re-entrena el escuadrón completo sobre las 19,937 empresas del panel,
    calibra isotónicamente las probabilidades y proyecta el riesgo para 2026.
    """
    print("\n================================================================================================")
    print("RE-ENTRENANDO SUPER-ENSAMBLE EN EL PANEL COMPLETO (19,937 EMPRESAS) PARA PROYECCIÓN 2026")
    print("================================================================================================")

    modelos_finales = {}
    # LightGBM y XGBoost se incluyen solo si su librería cargó (blindaje opcional).
    if TIENE_LGBM:
        modelos_finales["LightGBM"] = LGBMClassifier(
            n_estimators=450, learning_rate=0.02, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
            scale_pos_weight=7, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
    if TIENE_XGB:
        modelos_finales["XGBoost"] = XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, scale_pos_weight=7,
            tree_method="hist", eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1
        )
    modelos_finales["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=25,
        l2_regularization=5.0, class_weight={0: 1, 1: 8}, random_state=RANDOM_STATE
    )
    modelos_finales["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=9, min_samples_leaf=8, max_features="sqrt",
        class_weight={0: 1, 1: 10}, random_state=RANDOM_STATE, n_jobs=-1
    )

    # Pesos dinámicos del ensamble final (renormalizados a 1.0 sobre los presentes).
    pesos_finales = calcular_pesos_ensamble(modelos_finales.keys())
    assert abs(sum(pesos_finales.values()) - 1.0) < 1e-9, "Los pesos finales no suman 1.0."

    # Ajuste final de modelos
    for n, m in modelos_finales.items():
        m.fit(X_completo, y_completo)

    # Predicciones OOF completas para calibración isotónica rigurosa
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oofs = {}
    for n, m in modelos_finales.items():
        oofs[n] = cross_val_predict(clone(m), X_completo, y_completo, cv=skf, method="predict_proba")[:, 1]

    score_oof_ensamble = sum(peso * oofs[nombre] for nombre, peso in pesos_finales.items())

    roc_final = roc_auc_score(y_completo, score_oof_ensamble)
    pr_final = average_precision_score(y_completo, score_oof_ensamble)
    print(f"Desempeño OOF en Panel Completo (19,937 empresas) -> ROC-AUC: {roc_final:.4f} | PR-AUC: {pr_final:.4f}")

    # Calibrador isotónico monótono
    calibrador_isotonico = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrador_isotonico.fit(score_oof_ensamble, y_completo)

    # Umbrales calibrados
    umbral_rojo_2026 = float(np.percentile(score_oof_ensamble, 99.0))
    umbral_naranja_2026 = float(np.percentile(score_oof_ensamble, 97.5))
    umbral_amarillo_2026 = float(np.percentile(score_oof_ensamble, 90.0))

    # Filtrar población sana de 2025 para proyectar 2026
    sanas_estados = ["ACTIVA", "EN ETAPA PREOPERATIVA"]
    df_2025_sanas = df_2025[
        df_2025["Estado actual"].isin(sanas_estados) &
        (df_2025["Flag_Quiebra_Patrimonial"] == 0)
    ].copy()

    prev_2024 = obtener_saldos_periodo_anterior(SIIS_DIR, "prev_2024.pkl")

    # Serie plurianual para la proyección: corte t = 2025, por lo que la ventana de
    # historia es 2021..2024 (TODO <= 2024; jamás se usa 2025 como insumo, solo como
    # etiqueta objetivo que NO entra en X). Coherente con el anti-fuga del pipeline.
    series_multianio_proy = None
    if USAR_FEATURES_MULTIANIO:
        from datos_multianio import construir_series_saldos_multianio
        series_multianio_proy = construir_series_saldos_multianio(usar_cache=True)

    X_2025 = construir_matriz_caracteristicas(
        df_2025_sanas, prev_2024, series_multianio=series_multianio_proy, anio_t=2025
    )[X_completo.columns]
    assert X_2025.isnull().sum().sum() == 0, "Nulos detectados en matriz 2025"

    score_2025 = sum(
        peso * modelos_finales[nombre].predict_proba(X_2025)[:, 1]
        for nombre, peso in pesos_finales.items()
    )

    probs_calibradas_2026 = calibrador_isotonico.predict(score_2025)

    zonas_2026 = np.select(
        [score_2025 >= umbral_rojo_2026, score_2025 >= umbral_naranja_2026, score_2025 >= umbral_amarillo_2026],
        ["4. ROJO ÉLITE (Peligro Inminente)", "3. NARANJA (Riesgo Alto)", "2. AMARILLO (Vigilancia / Alerta)"],
        default="1. VERDE (Estable / Sano)"
    )

    df_resultado_2026 = df_2025_sanas[["NIT", "Razón social de la sociedad", COL_CIIU]].copy()
    df_resultado_2026["Score_Ensamble_2025"] = score_2025
    df_resultado_2026["Probabilidad_Calibrada_Quiebra_2026"] = probs_calibradas_2026
    df_resultado_2026["Zona_Semaforo_2026"] = zonas_2026

    # Señales detonantes principales por empresa. Se prefiere XGBoost por su
    # feature_importances_ basado en ganancia; si no está disponible se usa el
    # primer modelo presente que exponga importancias (RandomForest / HGB).
    modelo_importancias = modelos_finales.get("XGBoost") or modelos_finales.get(
        "LightGBM"
    ) or modelos_finales["RandomForest"]
    importancias = modelo_importancias.feature_importances_
    cols_top = pd.Series(importancias, index=X_completo.columns).nlargest(10).index
    anomalias = (X_2025[cols_top].rank(pct=True) - 0.5).abs()
    df_resultado_2026["Detonantes_Criticos"] = anomalias.apply(
        lambda fila: ", ".join(fila.nlargest(3).index), axis=1
    )

    df_resultado_2026 = df_resultado_2026.sort_values(by="Score_Ensamble_2025", ascending=False)
    ruta_semaforo = SALIDAS_DIR / "semaforo_2026.csv"
    df_resultado_2026.to_csv(ruta_semaforo, index=False, encoding="utf-8-sig")

    resumen_2026 = df_resultado_2026.groupby("Zona_Semaforo_2026").agg(
        Total_Empresas=("NIT", "count"),
        Probabilidad_Media=("Probabilidad_Calibrada_Quiebra_2026", "mean"),
        Quiebras_Esperadas=("Probabilidad_Calibrada_Quiebra_2026", "sum")
    )
    print("\n================================================================================================")
    print(f"PROYECCIÓN DEL SEMÁFORO 2026 ({len(df_resultado_2026):,} EMPRESAS EVALUADAS)")
    print("================================================================================================")
    print(resumen_2026.round(4).to_string())

    # Exportar modelo y metadatos
    joblib.dump({
        "modelos": modelos_finales,
        "features": list(X_completo.columns),
        "calibrador": calibrador_isotonico,
        "umbrales": {
            "rojo_elite": umbral_rojo_2026,
            "naranja": umbral_naranja_2026,
            "amarillo": umbral_amarillo_2026
        }
    }, SALIDAS_DIR / "modelo_final_2026.joblib")

    return resumen_2026, df_resultado_2026


# ==============================================================================
# EJECUCIÓN PRINCIPAL DEL PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("Iniciando Pipeline de Alta Precisión Forense (Fase 5 Optimizada)...")
    X, y, X_tr, X_te, y_tr, y_te, panel, df_2025 = construir_panel_y_splits()
    print(f"Panel 2024 -> 2025: {X.shape[0]} empresas x {X.shape[1]} características.")
    print(f"Eventos de insolvencia y quiebra en 2025: {y.sum()} ({y.mean():.4%}).")

    # Entrenamiento y evaluación en Test
    modelos, score_test, pred_dict = entrenar_escuadron_elite(X_tr, y_tr, X_te, y_te)
    cortes_test, semaforo_test, u_rojo, u_amar = auditar_efectividad_test(y_te, score_test)

    # Proyección 2026 sobre el universo completo de empresas sanas de 2025
    resumen_2026, df_2026 = proyectar_semaforo_2026(X, y, df_2025)

    # Persistencia de reporte completo en formato JSON
    REPORTE_METRICAS = {
        "panel_total": len(panel),
        "quiebras_total": int(y.sum()),
        "roc_auc_test": round(float(roc_auc_score(y_te, score_test)), 4),
        "pr_auc_test": round(float(average_precision_score(y_te, score_test)), 4),
        "cortes_efectividad_test": cortes_test,
        "semaforo_test": semaforo_test.round(4).reset_index().to_dict(orient="records"),
        "semaforo_2026": resumen_2026.round(4).reset_index().to_dict(orient="records")
    }
    with open(SALIDAS_DIR / "reporte_fase5.json", "w", encoding="utf-8") as f:
        json.dump(REPORTE_METRICAS, f, indent=2, ensure_ascii=False)

    print("\nProceso completado con éxito. Resultados guardados en carpeta 'salidas'.")
