# -*- coding: utf-8 -*-
"""
RECONSTRUCCION DE MAESTROS HISTORICOS CON RATIOS FASE-4 (FEAT-001)
==============================================================================
Objetivo de negocio
--------------------
La ZONA ROJA del modelo de insolvencia esta limitada por la escasez de positivos:
el unico panel disponible (2024 -> 2025) aporta solo 304 quiebras. Para afinar los
tiers de alta probabilidad (ROJO ELITE / NARANJA) necesitamos MAS empresas y mas
positivos, apilando transiciones longitudinales adicionales (2021 -> 2022,
2022 -> 2023, 2023 -> 2024). Para armar esas transiciones hace falta, por cada ano
t, un MAESTRO completo con las 51 ratios Fase-4 y las cuentas crudas con el MISMO
esquema de 94 columnas que ``df_maestro_con_ratios_2024.pkl``, de modo que la
funcion ``construir_matriz_caracteristicas`` del pipeline activo corra sin cambios.

Esos maestros solo existen hoy para 2024 y 2025. Este modulo los RECONSTRUYE para
2021, 2022 y 2023 replicando la logica de las Fases 1 a 4 del repo
(``py/script_uno_2024.py`` + ``py/script_dos_limpieza_2024.py`` +
``py/script_tres_ratios_2024.py``), apuntando a cada carpeta
``SIIS/pymes_individuales-<anio>``.

Alcance de FEAT-001
-------------------
Esta feature SOLO construye y cachea datos; NO cambia el entrenamiento. Entrega:
  * ``construir_maestro_ratios_anio(anio, usar_cache)`` -> maestro de 94 columnas.
  * ``construir_estado_actual_por_anio(usar_cache)`` -> dict {anio: DataFrame[NIT,
    estado_actual_norm]} desde las caratulas 10000 de 2021..2024.
  * Diagnostico de cuantos positivos y_estricto aportaria cada transicion
    (2021->2022, 2022->2023, 2023->2024) frente a los 304 del panel 2024->2025,
    persistido en ``salidas/diagnostico_transiciones.json``.

Prohibiciones (autoritativas del usuario)
-----------------------------------------
  * ``py/`` y ``SIIS/`` son SOLO LECTURA: aqui solo se LEEN los .xlsx de SIIS y no
    se modifica ningun script de ``py/``. La logica de Fase 1-4 se COPIA/ADAPTA aqui.
  * Todo el codigo y artefactos viven en ``fase5_optimizacion/``.
  * Codigo, comentarios y analisis EN ESPANOL, sin guiones largos.
  * Validaciones por assert; sin fuga de datos (cada maestro usa solo datos del
    propio ano t; la parte legal de la etiqueta y usa el estado de t+1, nunca t).

Reutilizacion de codigo
-----------------------
Se importan utilidades ya auditadas del pipeline activo
(``normalizar_texto`` NFKD, ``normalizar_nit``, ``buscar_archivo_siis``,
``SIIS_DIR``, ``SALIDAS_DIR``). El import es seguro: ``script_cinco_optimizado.py``
protege su ejecucion con ``if __name__ == '__main__'``, de modo que importarlo NO
dispara el entrenamiento.
"""

import json

import numpy as np
import pandas as pd

from script_cinco_optimizado import (
    SALIDAS_DIR,
    SIIS_DIR,
    buscar_archivo_siis,
    normalizar_nit,
    normalizar_texto,
)

# ------------------------------------------------------------------------------
# PARAMETROS GLOBALES
# ------------------------------------------------------------------------------
# Anos para los que se reconstruye el maestro completo con ratios (no 2024/2025,
# que ya existen como pkl, ni 2020, que solo es observable como 'periodo anterior').
ANIOS_HISTORICOS = (2021, 2022, 2023)

# Anos cuya caratula 10000 aporta el 'Estado actual' (snapshot anual por empresa).
# Se incluye 2024 porque la etiqueta de la transicion 2023->2024 usa el estado 2024.
ANIOS_ESTADO = (2021, 2022, 2023, 2024)

# Estados que se consideran "sanos" (empresa operativa, no en proceso concursal).
# Identica definicion a la del pipeline (construir_panel_y_splits): una empresa es
# candidata a positivo si estaba sana en t y deja de estarlo en t+1. Se normalizan
# con normalizar_texto para ser inmunes a tildes/mayusculas (p.ej. 'ACTIVA' y
# 'EN ETAPA PREOPERATIVA').
ESTADOS_SANOS_NORM = frozenset({
    normalizar_texto("ACTIVA"),
    normalizar_texto("EN ETAPA PREOPERATIVA"),
})

NOMBRE_CACHE_ESTADO = "estado_actual_por_anio.pkl"
NOMBRE_DIAGNOSTICO_TRANSICIONES = "diagnostico_transiciones.json"
NOMBRE_MAESTRO_REFERENCIA = "df_maestro_con_ratios_2024.pkl"


def _ruta_carpeta_anio(anio: int):
    """Devuelve la ruta a la carpeta ``SIIS/pymes_individuales-<anio>`` y valida que exista."""
    carpeta = SIIS_DIR / f"pymes_individuales-{anio}"
    assert carpeta.is_dir(), f"No existe la carpeta esperada: {carpeta}"
    return carpeta


# ==============================================================================
# UTILIDADES DE LECTURA DE ANEXOS (replican el 'ESCUDO PRE-MERGE' de la Fase 1)
# ==============================================================================
def _limpiar_nit_inplace(df: pd.DataFrame) -> None:
    """
    Normaliza la columna NIT en el sitio replicando ``limpiar_nit`` de la Fase 1
    pero reutilizando ``normalizar_nit`` (quita espacios y el sufijo decimal '.0').
    """
    df["NIT"] = normalizar_nit(df["NIT"])


def _verificar_duplicados(df: pd.DataFrame, nombre: str) -> None:
    """Replica ``verificar_duplicados`` de la Fase 1: aborta si hay NITs clonados."""
    duplicados = df[df["NIT"].duplicated(keep=False)]
    assert duplicados.empty, (
        f"ERROR: [{nombre}] tiene NITs duplicados tras la deduplicacion forense: "
        f"{duplicados['NIT'].value_counts().to_dict()}"
    )


def _verificar_sin_nit(df: pd.DataFrame, nombre: str) -> None:
    """Replica ``verificar_sinNIT`` de la Fase 1: aborta si hay filas sin NIT."""
    assert df["NIT"].isnull().sum() == 0, f"ERROR: [{nombre}] trae filas sin NIT."


def _leer_anexo(
    carpeta,
    prefijo: str,
    columnas: list,
    nombre: str,
    filtro_periodo_actual: bool = False,
    filtro_concepto: str = None,
    columna_concepto: str = "Concepto",
) -> pd.DataFrame:
    """
    Lee un anexo del SIIS replicando el 'ESCUDO PRE-MERGE' de ``script_uno_2024.py``:
    carga solo las columnas necesarias, limpia el NIT, filtra por 'Periodo Actual'
    y/o por un 'Concepto' concreto cuando aplica, se queda con la Fecha de Corte mas
    reciente por NIT y valida unicidad.

    Parametros
    ----------
    carpeta : pathlib.Path
        Carpeta ``SIIS/pymes_individuales-<anio>``.
    prefijo : str
        Prefijo numerico del anexo (p.ej. '210030'); se resuelve con
        ``buscar_archivo_siis`` (glob determinista, un unico archivo).
    columnas : list[str]
        Columnas a materializar. Debe incluir 'NIT' y 'Fecha de Corte' (y 'Periodo'
        o la columna de concepto cuando se filtra por ellas).
    nombre : str
        Etiqueta para los mensajes de assert.
    filtro_periodo_actual : bool
        Si True, conserva solo las filas con 'Periodo' == 'Periodo Actual' (anexos
        210030/310030/410030 que traen periodo actual + anterior en el mismo archivo).
    filtro_concepto : str | None
        Si se indica, conserva solo las filas cuya columna de concepto es igual a este
        valor (p.ej. 'Emisión de patrimonio' en 610010, o el total corriente en
        801000/801600). La comparacion es exacta sobre el texto original (identica a
        la Fase 1, que compara contra el literal con tildes).
    columna_concepto : str
        Nombre de la columna sobre la que aplica ``filtro_concepto`` (por defecto
        'Concepto').

    Devuelve
    --------
    pandas.DataFrame
        Una fila por NIT, con las columnas de datos pedidas (se eliminan 'Periodo',
        la de concepto y 'Fecha de Corte', usadas solo como filtros/orden).

    Notas tecnicas (pandas)
    -----------------------
    * pandas.read_excel con ``usecols`` como lista materializa solo esas columnas,
      acelerando la lectura de estos .xlsx grandes.
    * drop_duplicates(keep='last') tras sort_values por 'Fecha de Corte' conserva el
      reporte mas reciente por NIT (deduplicacion forense, identica a la Fase 1).
    """
    ruta = buscar_archivo_siis(carpeta, prefijo)
    df = pd.read_excel(ruta, usecols=columnas)
    _limpiar_nit_inplace(df)

    if filtro_periodo_actual:
        # El filtro de la Fase 1 compara contra el literal 'Periodo Actual'. Se usa
        # normalizar_texto para ser inmune a diferencias de tildes/mayusculas entre
        # anos, pero el valor objetivo es el mismo 'periodo actual'.
        df = df[df["Periodo"].astype(str).map(normalizar_texto) == "periodo actual"]

    if filtro_concepto is not None:
        df = df[df[columna_concepto] == filtro_concepto]

    df = df.sort_values(by=["NIT", "Fecha de Corte"]).drop_duplicates(subset=["NIT"], keep="last")

    columnas_a_borrar = [c for c in ["Periodo", columna_concepto, "Fecha de Corte"] if c in df.columns]
    df = df.drop(columns=columnas_a_borrar)

    _verificar_duplicados(df, nombre)
    _verificar_sin_nit(df, nombre)
    return df


# ==============================================================================
# FASE 1 (ADAPTADA): CONSTRUCCION DEL MAESTRO CRUDO DE UN ANO
# ==============================================================================
def _construir_maestro_crudo_anio(anio: int) -> pd.DataFrame:
    """
    Replica ``script_uno_2024.py`` para un ano arbitrario: lee la caratula 10000 y
    los anexos 210030, 310030, 410030, 520000, 610010, 801000, 801600, 801800b, y
    los mergea por NIT con LEFT join sobre la caratula (que fija la poblacion de
    empresas del ano). Mantiene los 'ESCUDOS POST-MERGE' por assert.
    """
    carpeta = _ruta_carpeta_anio(anio)

    # --- ARCHIVO 1: 10000_Caratula (fija la poblacion de empresas del ano) ---
    columnas_caratula = [
        "Punto de Entrada",
        "NIT",
        "Razón social de la sociedad",
        "Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)",
        "Fecha de constitución (Aaaa-Mm-Dd)",
        "Estado actual",
        "Fecha de Corte",
        "Concepto del Revisor fiscal en su informe",
        "Estos estados financieros presentan información reexpresada?",
        "El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior",
    ]
    df_caratula = pd.read_excel(buscar_archivo_siis(carpeta, "10000"), usecols=columnas_caratula)
    _limpiar_nit_inplace(df_caratula)
    df_caratula = df_caratula.sort_values(by=["NIT", "Fecha de Corte"]).drop_duplicates(subset=["NIT"], keep="last")
    _verificar_duplicados(df_caratula, f"Caratula {anio}")
    _verificar_sin_nit(df_caratula, f"Caratula {anio}")

    df_maestro = df_caratula

    def _merge(df_base, df_anexo, nombre):
        """Merge LEFT por NIT con escudo post-merge (no debe alterar el nº de filas)."""
        fusion = pd.merge(df_base, df_anexo, on="NIT", how="left")
        assert len(fusion) == len(df_caratula), f"ERROR: el cruce de {nombre} altero las filas maestras."
        return fusion

    # --- ARCHIVO 2: 210030_Estado de situacion financiera (BALANCE) ---
    columnas_balance = [
        "NIT",
        "Total de activos (Assets)",
        "Total pasivos (Liabilities)",
        "Patrimonio total (Equity)",
        "Activos corrientes totales (CurrentAssets)",
        "Pasivos corrientes totales (CurrentLiabilities)",
        "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)",
        "Inventarios corrientes (Inventories)",
        "Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes (TradeAndOtherCurrentReceivables)",
        "Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)",
        "Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)",
        "Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)",
        "Periodo",
        "Fecha de Corte",
    ]
    df_balance = _leer_anexo(carpeta, "210030", columnas_balance, f"Balance {anio}", filtro_periodo_actual=True)
    df_maestro = _merge(df_maestro, df_balance, "Balance (210030)")

    # --- ARCHIVO 3: 310030_Estado de resultados por funcion de gasto ---
    columnas_resultados = [
        "NIT",
        "Ingresos de actividades ordinarias (Revenue)",
        "Ganancia (pérdida) (ProfitLoss)",
        "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)",
        "Costo de ventas (CostOfSales)",
        "Costos financieros (FinanceCosts)",
        "Otros ingresos (OtherIncome)",
        "Ganancia bruta (GrossProfit)",
        "Gastos de administración (AdministrativeExpense)",
        "Gastos de ventas (DistributionCosts)",
        "Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)",
        "Periodo",
        "Fecha de Corte",
    ]
    df_resultados = _leer_anexo(carpeta, "310030", columnas_resultados, f"Resultados {anio}", filtro_periodo_actual=True)
    df_maestro = _merge(df_maestro, df_resultados, "Resultados (310030)")

    # --- ARCHIVO 4: 410030_Estado del resultado integral (ORI) ---
    columnas_ori = [
        "NIT",
        "Resultado integral total (ComprehensiveIncome)",
        "Periodo",
        "Fecha de Corte",
    ]
    df_ori = _leer_anexo(carpeta, "410030", columnas_ori, f"ORI {anio}", filtro_periodo_actual=True)
    df_maestro = _merge(df_maestro, df_ori, "ORI (410030)")

    # --- ARCHIVO 5: 520000_Estado de flujos de efectivo (metodo indirecto) ---
    # NOTA: 520000 no trae columna 'Periodo' (es flujo del ejercicio); solo se
    # deduplica por Fecha de Corte, identico a la Fase 1.
    columnas_flujos = [
        "NIT",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)",
        "Fecha de Corte",
    ]
    df_flujos = _leer_anexo(carpeta, "520000", columnas_flujos, f"Flujos {anio}")
    df_maestro = _merge(df_maestro, df_flujos, "Flujos (520000)")

    # --- ARCHIVO 6: 610010_Estado de cambios en el patrimonio (capital emitido) ---
    # Trampa Supersociedades: se filtra el concepto 'Emisión de patrimonio' (saldo
    # final del ano), identico a la Fase 1.
    columnas_patrimonio = [
        "NIT",
        "Concepto",
        "Capital emitido [miembro] (CapitalEmitidoMiembro)",
        "Fecha de Corte",
    ]
    df_patrimonio = _leer_anexo(
        carpeta, "610010", columnas_patrimonio, f"Patrimonio {anio}",
        filtro_concepto="Emisión de patrimonio",
    )
    df_maestro = _merge(df_maestro, df_patrimonio, "Patrimonio (610010)")

    # --- ARCHIVO 8: 801000_Notas - Cuentas comerciales por cobrar (cartera) ---
    columnas_cxc = [
        "NIT",
        "Concepto",
        "Total saldo del ejercicio [miembro] (CxCTotalMiembro)",
        "Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)",
        "Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)",
        "Fecha de Corte",
    ]
    df_cxc = _leer_anexo(
        carpeta, "801000", columnas_cxc, f"CxC {anio}",
        filtro_concepto="Total cuentas por cobrar y otras cuentas por cobrar corrientes",
    )
    df_maestro = _merge(df_maestro, df_cxc, "CxC (801000)")

    # --- ARCHIVO 14: 801600_Notas - Cuentas comerciales por pagar (morosidad) ---
    columnas_cxp = [
        "NIT",
        "Concepto",
        "Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)",
        "Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)",
        "Fecha de Corte",
    ]
    df_cxp = _leer_anexo(
        carpeta, "801600", columnas_cxp, f"CxP {anio}",
        filtro_concepto="Total cuentas comerciales por pagar, otras cuentas por pagar y otros pasivos corrientes",
    )
    df_maestro = _merge(df_maestro, df_cxp, "CxP (801600)")

    # --- ARCHIVO 17: 801800b_Notas - Otras ganancias/perdidas y nomina (desespero) ---
    columnas_desespero = [
        "NIT",
        "Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)",
        "Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)",
        "Gastos por beneficios de terminación (TerminationBenefitsExpense)",
        "Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)",
        "Fecha de Corte",
    ]
    df_desespero = _leer_anexo(carpeta, "801800b", columnas_desespero, f"Desespero {anio}")
    df_maestro = _merge(df_maestro, df_desespero, "Desespero (801800b)")

    return df_maestro.reset_index(drop=True)


# ==============================================================================
# FASE 3 (ADAPTADA): LIMPIEZA E IMPUTACION DEL MAESTRO CRUDO
# ==============================================================================
def _limpiar_maestro(df_maestro: pd.DataFrame) -> pd.DataFrame:
    """
    Replica ``script_dos_limpieza_2024.py``: repara fechas, crea
    'Edad_Empresa_Anios', imputa categoricas, descarta empresas con balance o
    resultados estructurales vacios, imputa con 0 las subcuentas donde NaN == $0 y
    corrige errores de escala ('tres ceros de mas'). Deja el maestro sin nulos.
    """
    df_maestro = df_maestro.copy()

    # --- PASO 1: caratula (fechas, edad, categoricas) ---
    fechas_texto = df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"].astype(str).str.strip()
    correcciones_fechas = {
        r"^119": "199",
        r"^020": "200",
        r"^021": "201",
        r"^0189": "1989",
        r"^0199": "1999",
        r"^0997": "1997",
        r"^0015": "2015",
        r"^0986": "1986",
    }
    for patron, reemplazo in correcciones_fechas.items():
        fechas_texto = fechas_texto.str.replace(patron, reemplazo, regex=True)

    df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"] = pd.to_datetime(fechas_texto, errors="coerce")
    df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"] = df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"].where(
        (df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"].dt.year >= 1900)
        & (df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"].dt.year <= 2025)
    )
    df_maestro["Fecha de Corte"] = pd.to_datetime(df_maestro["Fecha de Corte"], errors="coerce")

    df_maestro["Edad_Empresa_Anios"] = (
        df_maestro["Fecha de Corte"] - df_maestro["Fecha de constitución (Aaaa-Mm-Dd)"]
    ).dt.days / 365.25
    mediana_edad = df_maestro["Edad_Empresa_Anios"].median()
    df_maestro["Edad_Empresa_Anios"] = df_maestro["Edad_Empresa_Anios"].fillna(mediana_edad)

    df_maestro = df_maestro.drop(columns=["Fecha de constitución (Aaaa-Mm-Dd)", "Fecha de Corte"])

    df_maestro["Concepto del Revisor fiscal en su informe"] = df_maestro[
        "Concepto del Revisor fiscal en su informe"
    ].fillna("00. NO APLICA / SIN OBLIGACIÓN")
    df_maestro["Estos estados financieros presentan información reexpresada?"] = df_maestro[
        "Estos estados financieros presentan información reexpresada?"
    ].fillna("NO")
    df_maestro["Estado actual"] = df_maestro["Estado actual"].fillna("ACTIVA")
    df_maestro["Razón social de la sociedad"] = df_maestro["Razón social de la sociedad"].fillna("SIN RAZÓN SOCIAL")
    # Imputacion de CIIU ausente (hallazgo historico: 2023 trae 1 empresa sin CIIU,
    # mientras que el maestro 2024 no tenia nulos aqui). Se usa un codigo neutro que
    # agrupa a estas empresas en su propio "sector" sin romper las desviaciones
    # sectoriales CIIU de construir_matriz_caracteristicas (que agrega por esta
    # columna). Mantiene la garantia de maestro sin nulos de la Fase 3.
    df_maestro["Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)"] = (
        df_maestro["Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)"].fillna("SIN CIIU")
    )
    df_maestro["El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior"] = (
        df_maestro["El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior"].fillna("NO")
    )

    # --- PASO 2: balance (descartar estructurales vacios + imputar subcuentas 0) ---
    columnas_estructurales_obligatorias = [
        "Total de activos (Assets)",
        "Total pasivos (Liabilities)",
        "Patrimonio total (Equity)",
        "Activos corrientes totales (CurrentAssets)",
        "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)",
        "Ingresos de actividades ordinarias (Revenue)",
        "Ganancia (pérdida) (ProfitLoss)",
        "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)",
        "Ganancia bruta (GrossProfit)",
    ]
    df_maestro = df_maestro.dropna(subset=columnas_estructurales_obligatorias).reset_index(drop=True)

    columnas_imputar_cero_balance = [
        "Pasivos corrientes totales (CurrentLiabilities)",
        "Inventarios corrientes (Inventories)",
        "Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes (TradeAndOtherCurrentReceivables)",
        "Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)",
        "Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)",
        "Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)",
    ]
    df_maestro[columnas_imputar_cero_balance] = df_maestro[columnas_imputar_cero_balance].fillna(0)

    # --- PASO 3: resultados (imputar costos/gastos/impuestos con 0) ---
    columnas_imputar_cero_resultados = [
        "Costo de ventas (CostOfSales)",
        "Costos financieros (FinanceCosts)",
        "Gastos de administración (AdministrativeExpense)",
        "Gastos de ventas (DistributionCosts)",
        "Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)",
    ]
    if "Otras ganancias (pérdidas) (OtherGainsLosses)" in df_maestro.columns:
        columnas_imputar_cero_resultados.append("Otras ganancias (pérdidas) (OtherGainsLosses)")
    if "Otros ingresos (OtherIncome)" in df_maestro.columns:
        columnas_imputar_cero_resultados.append("Otros ingresos (OtherIncome)")
    df_maestro[columnas_imputar_cero_resultados] = df_maestro[columnas_imputar_cero_resultados].fillna(0)

    # --- PASO 4: ORI (imputar con la utilidad neta cuando falta) ---
    df_maestro["Resultado integral total (ComprehensiveIncome)"] = df_maestro[
        "Resultado integral total (ComprehensiveIncome)"
    ].fillna(df_maestro["Ganancia (pérdida) (ProfitLoss)"])

    # --- PASO 5: flujos de efectivo (imputar 0 sin .abs(), los negativos son reales) ---
    columnas_flujos = [
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)",
    ]
    df_maestro[columnas_flujos] = df_maestro[columnas_flujos].fillna(0)

    # --- PASO 6: patrimonio (imputar 0 y recortar negativos: solo inyeccion fresca) ---
    col_capital = "Capital emitido [miembro] (CapitalEmitidoMiembro)"
    df_maestro[col_capital] = df_maestro[col_capital].fillna(0).clip(lower=0)

    # --- PASO 7: cartera CxC (imputar 0) ---
    columnas_cxc = [
        "Total saldo del ejercicio [miembro] (CxCTotalMiembro)",
        "Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)",
        "Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)",
    ]
    df_maestro[columnas_cxc] = df_maestro[columnas_cxc].fillna(0)

    # --- PASO 8: proveedores CxP (imputar 0, corregir escala y topes logicos) ---
    col_cxp_vencidas = "Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)"
    col_cxp_mas_360 = "Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)"
    col_pasivo_corriente = "Pasivos corrientes totales (CurrentLiabilities)"
    df_maestro[[col_cxp_vencidas, col_cxp_mas_360]] = df_maestro[[col_cxp_vencidas, col_cxp_mas_360]].fillna(0)

    mascara_tres_ceros_cxp = (
        (df_maestro[col_pasivo_corriente] > 0)
        & (df_maestro[col_cxp_vencidas] > df_maestro[col_pasivo_corriente] * 50)
    )
    df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_vencidas] = df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_vencidas] / 1000.0
    df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_mas_360] = df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_mas_360] / 1000.0
    df_maestro[col_cxp_vencidas] = np.minimum(df_maestro[col_cxp_vencidas], df_maestro[col_pasivo_corriente])
    df_maestro[col_cxp_mas_360] = np.minimum(df_maestro[col_cxp_mas_360], df_maestro[col_cxp_vencidas])

    # --- PASO 9: desespero gerencial y nomina (imputar 0, signos, escala, coherencia) ---
    col_gan_activos = "Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)"
    col_per_activos = "Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)"
    col_despidos = "Gastos por beneficios de terminación (TerminationBenefitsExpense)"
    col_total_empleados = "Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)"
    col_activos_totales = "Total de activos (Assets)"

    columnas_desespero = [col_gan_activos, col_per_activos, col_despidos, col_total_empleados]
    df_maestro[columnas_desespero] = df_maestro[columnas_desespero].fillna(0)

    mascara_gan_negativa = df_maestro[col_gan_activos] < 0
    df_maestro.loc[mascara_gan_negativa, col_per_activos] = (
        df_maestro.loc[mascara_gan_negativa, col_per_activos].abs()
        + df_maestro.loc[mascara_gan_negativa, col_gan_activos].abs()
    )
    df_maestro[col_gan_activos] = df_maestro[col_gan_activos].clip(lower=0)
    df_maestro[col_per_activos] = df_maestro[col_per_activos].abs()
    df_maestro[col_total_empleados] = df_maestro[col_total_empleados].abs()
    df_maestro[col_despidos] = df_maestro[col_despidos].abs()

    mascara_tres_ceros_activos = (
        (df_maestro[col_activos_totales] > 0)
        & (df_maestro[col_gan_activos] > df_maestro[col_activos_totales] * 10)
    )
    df_maestro.loc[mascara_tres_ceros_activos, col_gan_activos] /= 1000.0
    df_maestro[col_gan_activos] = np.minimum(df_maestro[col_gan_activos], df_maestro[col_activos_totales])
    df_maestro[col_per_activos] = np.minimum(df_maestro[col_per_activos], df_maestro[col_activos_totales])

    gastos_totales = (
        df_maestro["Costo de ventas (CostOfSales)"]
        + df_maestro["Gastos de administración (AdministrativeExpense)"]
        + df_maestro["Gastos de ventas (DistributionCosts)"]
    )
    mascara_tres_ceros_nomina = (
        (gastos_totales > 0) & (df_maestro[col_total_empleados] > gastos_totales * 10)
    )
    df_maestro.loc[mascara_tres_ceros_nomina, col_total_empleados] /= 1000.0
    df_maestro.loc[mascara_tres_ceros_nomina, col_despidos] /= 1000.0
    df_maestro[col_total_empleados] = np.maximum(df_maestro[col_total_empleados], df_maestro[col_despidos])

    # --- AUDITORIA FINAL: sin nulos (identica al assert de la Fase 3) ---
    total_nulos = int(df_maestro.isnull().sum().sum())
    nulos_por_col = df_maestro.isnull().sum()
    nulos_por_col = nulos_por_col[nulos_por_col > 0].to_dict()
    assert total_nulos == 0, (
        f"ERROR: quedan {total_nulos} valores nulos tras la limpieza. Columnas: {nulos_por_col}"
    )
    return df_maestro


# ==============================================================================
# FASE 4 (ADAPTADA): INGENIERIA DE 51 RATIOS Y BANDERAS FORENSES
# ==============================================================================
def _crear_ratios_fase4(df_maestro: pd.DataFrame) -> pd.DataFrame:
    """
    Replica ``script_tres_ratios_2024.py``: crea las 51 ratios/banderas Fase-4 EN EL
    MISMO ORDEN que el original, de modo que el slice contiguo
    'Flag_Reexpresion_Contable':'Flag_Desespero_Gerencial' quede identico al del
    maestro 2024. Elimina la columna redundante 'CxCTotalMiembro' (paso 7) y valida
    ausencia de nulos/infinitos en los indicadores creados.
    """
    df_maestro = df_maestro.copy()

    # --- PASO 1: caratula (banderas de gobernanza) ---
    df_maestro["Flag_Reexpresion_Contable"] = np.where(
        df_maestro["Estos estados financieros presentan información reexpresada?"].astype(str).str.strip().str.upper() == "SI",
        1, 0,
    )
    df_maestro["Flag_Distribucion_Utilidades"] = np.where(
        df_maestro["El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior"].astype(str).str.strip().str.upper() == "SI",
        1, 0,
    )
    dictamenes_alerta = ["02. CON SALVEDAD", "01. ABSTENCIÓN DE OPINIÓN", "04. NEGATIVO"]
    df_maestro["Flag_Alerta_Revisor_Fiscal"] = np.where(
        df_maestro["Concepto del Revisor fiscal en su informe"].isin(dictamenes_alerta), 1, 0
    )

    # --- PASO 2: balance (liquidez, endeudamiento, quiebra patrimonial) ---
    activos_totales = df_maestro["Total de activos (Assets)"]
    pasivos_totales = df_maestro["Total pasivos (Liabilities)"]
    patrimonio_total = df_maestro["Patrimonio total (Equity)"]
    activos_corrientes = df_maestro["Activos corrientes totales (CurrentAssets)"]
    pasivos_corrientes = df_maestro["Pasivos corrientes totales (CurrentLiabilities)"]
    efectivo = df_maestro["Efectivo y equivalentes al efectivo (CashAndCashEquivalents)"]
    inventarios = df_maestro["Inventarios corrientes (Inventories)"]
    intangibles = df_maestro["Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)"]
    impuestos_corrientes = df_maestro["Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)"]
    otros_pasivos_fin_corr = df_maestro["Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)"]

    df_maestro["Razon_Corriente"] = np.where(
        pasivos_corrientes > 0, activos_corrientes / pasivos_corrientes,
        np.where(activos_corrientes > 0, 10.0, 0.0),
    ).clip(0.0, 10.0)
    df_maestro["Prueba_Acida"] = np.where(
        pasivos_corrientes > 0, (activos_corrientes - inventarios) / pasivos_corrientes,
        np.where((activos_corrientes - inventarios) > 0, 10.0, 0.0),
    ).clip(0.0, 10.0)
    df_maestro["Ratio_Caja_Inmediata"] = np.where(
        pasivos_corrientes > 0, efectivo / pasivos_corrientes,
        np.where(efectivo > 0, 5.0, 0.0),
    ).clip(0.0, 5.0)
    df_maestro["Capital_Trabajo_Sobre_Activos"] = np.where(
        activos_totales > 0, (activos_corrientes - pasivos_corrientes) / activos_totales, -1.0
    ).clip(-1.0, 1.0)
    df_maestro["Nivel_Endeudamiento"] = np.where(
        activos_totales > 0, pasivos_totales / activos_totales, 2.0
    ).clip(0.0, 2.0)
    df_maestro["Flag_Quiebra_Patrimonial"] = np.where(patrimonio_total < 0, 1, 0)
    df_maestro["Ratio_Activos_Intangibles_Humo"] = np.where(
        activos_totales > 0, intangibles / activos_totales, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Financiacion_DIAN"] = np.where(
        activos_totales > 0, impuestos_corrientes / activos_totales, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Deuda_Rescate_Corto_Plazo"] = np.where(
        activos_totales > 0, otros_pasivos_fin_corr / activos_totales, 0.0
    ).clip(0.0, 1.0)

    # --- PASO 3: resultados (rentabilidad y eficiencia) ---
    ingresos = df_maestro["Ingresos de actividades ordinarias (Revenue)"]
    ganancia_bruta = df_maestro["Ganancia bruta (GrossProfit)"]
    costo_ventas = df_maestro["Costo de ventas (CostOfSales)"]
    ebit = df_maestro["Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)"]
    utilidad_neta = df_maestro["Ganancia (pérdida) (ProfitLoss)"]
    costos_financieros = df_maestro["Costos financieros (FinanceCosts)"]
    gastos_admin = df_maestro["Gastos de administración (AdministrativeExpense)"]
    gastos_ventas = df_maestro["Gastos de ventas (DistributionCosts)"]
    impuestos = df_maestro["Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)"]
    otros_ingresos = df_maestro["Otros ingresos (OtherIncome)"]

    df_maestro["Margen_Bruto"] = np.where(
        ingresos > 0, ganancia_bruta / ingresos, np.where(ganancia_bruta < 0, -1.0, 0.0)
    ).clip(-1.0, 1.0)
    df_maestro["Margen_Operacional_EBIT"] = np.where(
        ingresos > 0, ebit / ingresos, np.where(ebit < 0, -1.0, np.where(ebit > 0, 1.0, 0.0))
    ).clip(-1.0, 1.0)
    df_maestro["Margen_Neto"] = np.where(
        ingresos > 0, utilidad_neta / ingresos, np.where(utilidad_neta < 0, -1.0, np.where(utilidad_neta > 0, 1.0, 0.0))
    ).clip(-1.0, 1.0)
    df_maestro["ROA_Operativo"] = np.where(activos_totales > 0, ebit / activos_totales, -1.0).clip(-1.0, 1.0)
    df_maestro["Cobertura_Intereses"] = np.where(
        costos_financieros > 0, ebit / costos_financieros,
        np.where(ebit > 0, 10.0, np.where(ebit < 0, -5.0, 0.0)),
    ).clip(-5.0, 10.0)
    df_maestro["Rotacion_Inventario"] = np.where(
        inventarios > 0, costo_ventas / inventarios, 0.0
    ).clip(0.0, 50.0)
    df_maestro["Peso_Burocracia_Admin"] = np.where(
        ganancia_bruta > 0, gastos_admin / ganancia_bruta, np.where(gastos_admin > 0, 2.0, 0.0)
    ).clip(0.0, 2.0)
    df_maestro["Peso_Gasto_Ventas"] = np.where(
        ganancia_bruta > 0, gastos_ventas / ganancia_bruta, np.where(gastos_ventas > 0, 2.0, 0.0)
    ).clip(0.0, 2.0)
    df_maestro["Carga_Impositiva_DIAN"] = np.where(
        impuestos <= 0, 0.0, np.where(ebit <= 0, 1.0, impuestos / ebit)
    ).clip(0.0, 1.0)
    df_maestro["Flag_Rojos_Por_Impuestos"] = np.where(
        (utilidad_neta < 0) & ((utilidad_neta + impuestos) > 0) & (impuestos > 0), 1, 0
    )
    ingresos_totales = ingresos + otros_ingresos
    df_maestro["Ratio_Dependencia_Otros_Ingresos"] = np.where(
        ingresos_totales > 0, otros_ingresos / ingresos_totales, 0.0
    ).clip(0.0, 1.0)
    utilidad_neta_segura = np.where(utilidad_neta > 0, utilidad_neta, 1.0)
    df_maestro["Flag_Sobrevive_Por_Otros_Ingresos"] = np.where(
        ((utilidad_neta > 0) & ((otros_ingresos / utilidad_neta_segura) > 0.30)) | ((ebit <= 0) & (otros_ingresos > 0)),
        1, 0,
    )
    df_maestro["Flag_Sin_Ingresos_Operacionales"] = np.where(ingresos == 0, 1, 0)

    # --- PASO 4: ORI (empresas zombi) ---
    resultado_integral = df_maestro["Resultado integral total (ComprehensiveIncome)"]
    ori_puro = resultado_integral - utilidad_neta
    df_maestro["Proporcion_ORI_Sobre_Activos"] = np.where(
        activos_totales > 0, ori_puro / activos_totales, 0.0
    ).clip(-1.0, 1.0)
    df_maestro["Flag_Zombi_ORI_Neto"] = np.where(
        (utilidad_neta < 0) & (resultado_integral > 0), 1, 0
    )
    df_maestro["Flag_Maquillaje_Patrimonial_ORI"] = np.where(
        ((ebit < 0) | (utilidad_neta < 0)) & (ori_puro > 0), 1, 0
    )

    # --- PASO 5: flujos de efectivo (verdad bancaria) ---
    flujo_operativo = df_maestro["Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)"]
    flujo_inversion = df_maestro["Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)"]
    flujo_financiacion = df_maestro["Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)"]

    df_maestro["Ratio_Calidad_Ganancias"] = np.where(
        utilidad_neta > 0, flujo_operativo / utilidad_neta,
        np.where(flujo_operativo < 0, -1.0, np.where(flujo_operativo > 0, 0.5, 0.0)),
    ).clip(-1.0, 2.0)
    df_maestro["Brecha_Caja_vs_Utilidad"] = np.where(
        activos_totales > 0, (flujo_operativo - utilidad_neta) / activos_totales, 0.0
    ).clip(-1.0, 1.0)
    df_maestro["Ratio_Cobertura_Efectivo_Inmediata"] = np.where(
        pasivos_corrientes > 0, flujo_operativo / pasivos_corrientes,
        np.where(flujo_operativo > 0, 1.0, np.where(flujo_operativo < 0, -1.0, 0.0)),
    ).clip(-1.0, 2.0)
    df_maestro["Flag_Zombi_Caja"] = np.where(
        (utilidad_neta > 0) & (flujo_operativo < 0), 1, 0
    )
    df_maestro["Flag_Doble_Sangria_Caja"] = np.where(
        (utilidad_neta <= 0) & (flujo_operativo < 0), 1, 0
    )
    df_maestro["Ratio_Flujo_Inversion_Activos"] = np.where(
        activos_totales > 0, flujo_inversion / activos_totales, 0.0
    ).clip(-1.0, 1.0)
    df_maestro["Ratio_Flujo_Financiacion_Activos"] = np.where(
        activos_totales > 0, flujo_financiacion / activos_totales, 0.0
    ).clip(-1.0, 1.0)

    # --- PASO 6: patrimonio (inyeccion de capital de socios) ---
    capital_emitido = df_maestro["Capital emitido [miembro] (CapitalEmitidoMiembro)"]
    df_maestro["Ratio_Inyeccion_Capital_Activos"] = np.where(
        activos_totales > 0, capital_emitido / activos_totales, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Flag_Inyeccion_Capital_Socios"] = np.where(capital_emitido > 0, 1, 0)
    en_numeros_rojos = (
        (utilidad_neta < 0)
        | (df_maestro["Razon_Corriente"] < 1.0)
        | (df_maestro["Nivel_Endeudamiento"] > 0.70)
    )
    df_maestro["Flag_Rescate_Socios_En_Rojos"] = np.where(
        en_numeros_rojos & (capital_emitido > 0), 1, 0
    )

    # --- PASO 7: cartera toxica CxC ---
    cxc_saldo_total = df_maestro["Total saldo del ejercicio [miembro] (CxCTotalMiembro)"]
    cxc_vencidas = df_maestro["Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)"]
    cxc_mas_360 = df_maestro["Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)"]

    df_maestro["Ratio_Cartera_Toxica"] = np.where(
        cxc_saldo_total > 0, cxc_vencidas / cxc_saldo_total, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Cartera_Perdida_360d"] = np.where(
        cxc_saldo_total > 0, cxc_mas_360 / cxc_saldo_total, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Flag_Cartera_Perdida_360d"] = np.where(cxc_mas_360 > 0, 1, 0)
    activos_corrientes_sin_mora = np.maximum(0.0, activos_corrientes - cxc_vencidas)
    razon_corriente_depurada = np.where(
        pasivos_corrientes > 0, activos_corrientes_sin_mora / pasivos_corrientes,
        np.where(activos_corrientes_sin_mora > 0, 10.0, 0.0),
    )
    df_maestro["Flag_Falsa_Liquidez_Cartera"] = np.where(
        (df_maestro["Razon_Corriente"] >= 1.0)
        & ((df_maestro["Ratio_Cartera_Toxica"] > 0.30) | (razon_corriente_depurada < 1.0)),
        1, 0,
    )
    # Se elimina la columna redundante de saldo CxC (ya esta en el archivo 2).
    df_maestro = df_maestro.drop(columns=["Total saldo del ejercicio [miembro] (CxCTotalMiembro)"])

    # --- PASO 8: morosidad severa CxP ---
    cxp_vencidas = df_maestro["Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)"]
    cxp_mas_360 = df_maestro["Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)"]

    df_maestro["Ratio_Morosidad_Severa"] = np.where(
        pasivos_corrientes > 0, cxp_vencidas / pasivos_corrientes, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Deuda_Vencida_Mas_360d"] = np.where(
        pasivos_corrientes > 0, cxp_mas_360 / pasivos_corrientes, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Flag_Cobro_Juridico_Proveedores"] = np.where(cxp_mas_360 > 0, 1, 0)
    df_maestro["Flag_Default_Sin_Efectivo"] = np.where(
        (cxp_vencidas > 0) & (cxp_vencidas > efectivo), 1, 0
    )

    # --- PASO 9: desespero gerencial y nomina ---
    ganancias_venta_activos = df_maestro["Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)"]
    perdidas_venta_activos = df_maestro["Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)"]
    gastos_despidos = df_maestro["Gastos por beneficios de terminación (TerminationBenefitsExpense)"]
    total_gastos_empleados = df_maestro["Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)"]

    df_maestro["Ratio_Despidos_Terminacion"] = np.where(
        total_gastos_empleados > 0, gastos_despidos / total_gastos_empleados, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Supervivencia_Venta_Activos"] = np.where(
        utilidad_neta > 0, ganancias_venta_activos / utilidad_neta, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Liquidacion_Activos_Fijos"] = np.where(
        activos_totales > 0, (ganancias_venta_activos + perdidas_venta_activos) / activos_totales, 0.0
    ).clip(0.0, 1.0)
    df_maestro["Ratio_Peso_Nomina_Ingresos"] = np.where(
        ingresos > 0, total_gastos_empleados / ingresos, np.where(total_gastos_empleados > 0, 1.0, 0.0)
    ).clip(0.0, 1.0)
    df_maestro["Flag_Desespero_Gerencial"] = np.where(
        (ganancias_venta_activos > 0) | (perdidas_venta_activos > 0) | (gastos_despidos > 0), 1, 0
    )

    # --- AUDITORIA FINAL: 51 indicadores sin nulos ni infinitos ---
    todas_las_variables = [
        "Flag_Reexpresion_Contable", "Flag_Distribucion_Utilidades", "Flag_Alerta_Revisor_Fiscal",
        "Razon_Corriente", "Prueba_Acida", "Ratio_Caja_Inmediata", "Capital_Trabajo_Sobre_Activos",
        "Nivel_Endeudamiento", "Flag_Quiebra_Patrimonial", "Ratio_Activos_Intangibles_Humo",
        "Ratio_Financiacion_DIAN", "Ratio_Deuda_Rescate_Corto_Plazo",
        "Margen_Bruto", "Margen_Operacional_EBIT", "Margen_Neto", "ROA_Operativo",
        "Cobertura_Intereses", "Rotacion_Inventario", "Peso_Burocracia_Admin", "Peso_Gasto_Ventas",
        "Carga_Impositiva_DIAN", "Flag_Rojos_Por_Impuestos", "Ratio_Dependencia_Otros_Ingresos",
        "Flag_Sobrevive_Por_Otros_Ingresos", "Flag_Sin_Ingresos_Operacionales",
        "Proporcion_ORI_Sobre_Activos", "Flag_Zombi_ORI_Neto", "Flag_Maquillaje_Patrimonial_ORI",
        "Ratio_Calidad_Ganancias", "Brecha_Caja_vs_Utilidad", "Ratio_Cobertura_Efectivo_Inmediata",
        "Flag_Zombi_Caja", "Flag_Doble_Sangria_Caja", "Ratio_Flujo_Inversion_Activos",
        "Ratio_Flujo_Financiacion_Activos", "Ratio_Inyeccion_Capital_Activos",
        "Flag_Inyeccion_Capital_Socios", "Flag_Rescate_Socios_En_Rojos",
        "Ratio_Cartera_Toxica", "Ratio_Cartera_Perdida_360d", "Flag_Cartera_Perdida_360d",
        "Flag_Falsa_Liquidez_Cartera", "Ratio_Morosidad_Severa", "Ratio_Deuda_Vencida_Mas_360d",
        "Flag_Cobro_Juridico_Proveedores", "Flag_Default_Sin_Efectivo",
        "Ratio_Despidos_Terminacion", "Ratio_Supervivencia_Venta_Activos",
        "Ratio_Liquidacion_Activos_Fijos", "Ratio_Peso_Nomina_Ingresos", "Flag_Desespero_Gerencial",
    ]
    nulos = int(df_maestro[todas_las_variables].isnull().sum().sum())
    infinitos = int(np.isinf(df_maestro[todas_las_variables]).sum().sum())
    assert nulos == 0, f"ERROR: hay {nulos} nulos en los ratios Fase-4 creados."
    assert infinitos == 0, f"ERROR: hay {infinitos} infinitos en los ratios Fase-4 creados."
    assert len(todas_las_variables) == 51, "ERROR: la Fase-4 debe producir exactamente 51 indicadores."
    return df_maestro


# ==============================================================================
# VALIDACION DE ESQUEMA CONTRA EL MAESTRO 2024 DE REFERENCIA
# ==============================================================================
def _columnas_requeridas_matriz() -> list:
    """
    Devuelve las columnas que ``construir_matriz_caracteristicas`` necesita para
    correr sin cambios: el slice de 51 ratios (contiguo), las cuentas crudas SIIS
    usadas, y las columnas de contexto (CIIU, razon social, punto de entrada, estado
    actual, NIT, edad). Se usa para asertar paridad de esquema con el maestro 2024.
    """
    cuentas_crudas = [
        "Total de activos (Assets)",
        "Total pasivos (Liabilities)",
        "Patrimonio total (Equity)",
        "Activos corrientes totales (CurrentAssets)",
        "Pasivos corrientes totales (CurrentLiabilities)",
        "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)",
        "Ingresos de actividades ordinarias (Revenue)",
        "Ganancia (pérdida) (ProfitLoss)",
        "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)",
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)",
        "Costos financieros (FinanceCosts)",
        "Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)",
        "Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)",
        "Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)",
        "Gastos de administración (AdministrativeExpense)",
        "Gastos de ventas (DistributionCosts)",
        "Ganancia bruta (GrossProfit)",
    ]
    contexto = [
        "Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)",
        "Razón social de la sociedad",
        "Punto de Entrada",
        "Estado actual",
        "Flag_Quiebra_Patrimonial",
        "NIT",
        "Edad_Empresa_Anios",
    ]
    return cuentas_crudas + contexto


def _validar_esquema_contra_2024(df_maestro: pd.DataFrame, anio: int) -> None:
    """
    Asegura por assert que ``df_maestro`` reproduce el esquema necesario comparando
    contra ``df_maestro_con_ratios_2024.pkl``:

      * El slice contiguo 'Flag_Reexpresion_Contable':'Flag_Desespero_Gerencial'
        tiene exactamente las MISMAS 51 columnas y en el MISMO orden.
      * Todas las columnas requeridas por ``construir_matriz_caracteristicas``
        existen en el maestro reconstruido.
      * El conjunto total de columnas coincide con el del maestro 2024 (paridad de
        esquema; el orden de las columnas fuera del slice puede diferir sin afectar
        a la matriz, que accede por etiqueta).
    """
    ruta_ref = SALIDAS_DIR.parent / NOMBRE_MAESTRO_REFERENCIA
    assert ruta_ref.exists(), f"No se encuentra el maestro 2024 de referencia: {ruta_ref}"
    df_ref = pd.read_pickle(ruta_ref)

    slice_ref = list(df_ref.loc[:, "Flag_Reexpresion_Contable":"Flag_Desespero_Gerencial"].columns)
    slice_nuevo = list(df_maestro.loc[:, "Flag_Reexpresion_Contable":"Flag_Desespero_Gerencial"].columns)
    assert slice_nuevo == slice_ref, (
        f"ERROR ({anio}): el slice de 51 ratios no coincide con el maestro 2024.\n"
        f"Diferencias: faltan={set(slice_ref) - set(slice_nuevo)}, "
        f"sobran={set(slice_nuevo) - set(slice_ref)}, "
        f"mismo_orden={slice_nuevo == slice_ref}"
    )
    assert len(slice_nuevo) == 51, f"ERROR ({anio}): el slice de ratios no tiene 51 columnas."

    requeridas = set(_columnas_requeridas_matriz())
    faltantes = requeridas - set(df_maestro.columns)
    assert not faltantes, f"ERROR ({anio}): faltan columnas requeridas por la matriz: {faltantes}"

    cols_ref = set(df_ref.columns)
    cols_nuevo = set(df_maestro.columns)
    assert cols_nuevo == cols_ref, (
        f"ERROR ({anio}): el conjunto de columnas no coincide con el maestro 2024.\n"
        f"faltan={cols_ref - cols_nuevo}, sobran={cols_nuevo - cols_ref}"
    )


# ==============================================================================
# API PUBLICA: MAESTRO CON RATIOS DE UN ANO (CON CACHE)
# ==============================================================================
def construir_maestro_ratios_anio(anio: int, usar_cache: bool = True) -> pd.DataFrame:
    """
    Reconstruye (o carga de cache) el maestro con las 51 ratios Fase-4 de un ano
    historico, con el MISMO esquema de 94 columnas que ``df_maestro_con_ratios_2024``.

    Pipeline: Fase 1 (merge de anexos) -> Fase 3 (limpieza/imputacion) -> Fase 4
    (ratios) -> validacion de esquema contra el maestro 2024.

    Parametros
    ----------
    anio : int
        Ano a reconstruir. Debe estar en ANIOS_HISTORICOS (2021, 2022, 2023).
    usar_cache : bool
        Si True y existe ``salidas/maestro_ratios_<anio>.pkl``, se carga de cache y
        se evita releer los .xlsx (lectura lenta). Patron identico a
        ``obtener_saldos_periodo_anterior``.

    Devuelve
    --------
    pandas.DataFrame
        Maestro con 94 columnas: cuentas crudas SIIS + 51 ratios + contexto.
    """
    assert anio in ANIOS_HISTORICOS, (
        f"Ano no soportado: {anio}. Solo se reconstruyen {ANIOS_HISTORICOS} "
        "(2024/2025 ya existen como pkl)."
    )
    ruta_cache = SALIDAS_DIR / f"maestro_ratios_{anio}.pkl"
    if usar_cache and ruta_cache.exists():
        print(f"Cargando maestro {anio} desde cache: {ruta_cache.name}")
        df = pd.read_pickle(ruta_cache)
        _validar_esquema_contra_2024(df, anio)
        return df

    print(f"Reconstruyendo maestro {anio} desde los anexos SIIS (lectura lenta)...")
    df_crudo = _construir_maestro_crudo_anio(anio)
    df_limpio = _limpiar_maestro(df_crudo)
    df_ratios = _crear_ratios_fase4(df_limpio)

    # Deduplicacion final defensiva por NIT (la caratula ya garantiza unicidad).
    assert df_ratios["NIT"].is_unique, f"ERROR ({anio}): NITs duplicados en el maestro final."

    _validar_esquema_contra_2024(df_ratios, anio)

    pd.to_pickle(df_ratios, ruta_cache)
    print(f"Maestro {anio} persistido en cache: {ruta_cache.name} (shape {df_ratios.shape}).")
    return df_ratios


# ==============================================================================
# API PUBLICA: ESTADO ACTUAL POR ANO (CARATULAS 10000, CON CACHE)
# ==============================================================================
def construir_estado_actual_por_anio(usar_cache: bool = True) -> dict:
    """
    Lee la caratula 10000 de cada carpeta ``pymes_individuales-<anio>`` (2021..2024),
    extrae NIT + Estado actual (normalizados), deduplica por NIT en la Fecha de Corte
    mas reciente y devuelve un dict {anio: DataFrame[NIT, estado_actual_norm]}.

    Este snapshot anual del estado legal es la base de la parte LEGAL de la etiqueta
    ``y_estricto`` de las transiciones historicas: una empresa sana en t que pasa a
    un estado concursal en t+1 cuenta como positivo.

    Parametros
    ----------
    usar_cache : bool
        Si True y existe ``salidas/estado_actual_por_anio.pkl``, se carga de cache.

    Devuelve
    --------
    dict[int, pandas.DataFrame]
        Claves = anos. Valores = DataFrame con columnas 'NIT', 'estado_actual_norm'.

    Aserciones
    ----------
    * ACTIVA debe ser el estado dominante de cada ano.
    * Debe existir al menos un estado no-ACTIVA (distress) por ano.
    """
    ruta_cache = SALIDAS_DIR / NOMBRE_CACHE_ESTADO
    if usar_cache and ruta_cache.exists():
        print(f"Cargando estado actual por ano desde cache: {ruta_cache.name}")
        return pd.read_pickle(ruta_cache)

    estado_por_anio = {}
    activa_norm = normalizar_texto("ACTIVA")
    for anio in ANIOS_ESTADO:
        carpeta = _ruta_carpeta_anio(anio)
        ruta = buscar_archivo_siis(carpeta, "10000")
        df = pd.read_excel(ruta, usecols=["NIT", "Estado actual", "Fecha de Corte"])
        _limpiar_nit_inplace(df)
        df["Fecha de Corte"] = pd.to_datetime(df["Fecha de Corte"], errors="coerce")
        # Deduplicacion forense: conservar la fecha de corte mas reciente por NIT.
        df = df.sort_values(["NIT", "Fecha de Corte"]).drop_duplicates(subset=["NIT"], keep="last")
        # Imputamos ACTIVA en los pocos nulos de estado (identico a la Fase 3).
        df["Estado actual"] = df["Estado actual"].fillna("ACTIVA")
        df["estado_actual_norm"] = df["Estado actual"].astype(str).map(normalizar_texto)
        df = df[["NIT", "estado_actual_norm"]].reset_index(drop=True)

        _verificar_sin_nit(df, f"Estado actual {anio}")
        assert df["NIT"].is_unique, f"ERROR: NITs duplicados en estado actual {anio}."

        conteo = df["estado_actual_norm"].value_counts()
        assert conteo.idxmax() == activa_norm, (
            f"ERROR ({anio}): el estado dominante no es ACTIVA (fue {conteo.idxmax()!r})."
        )
        n_no_activa = int((df["estado_actual_norm"] != activa_norm).sum())
        assert n_no_activa > 0, f"ERROR ({anio}): no hay estados no-ACTIVA (distress)."

        estado_por_anio[anio] = df

    pd.to_pickle(estado_por_anio, ruta_cache)
    print(f"Estado actual por ano persistido en cache: {ruta_cache.name}.")
    return estado_por_anio


# ==============================================================================
# DIMENSIONAMIENTO DE POSITIVOS POR TRANSICION (DIAGNOSTICO)
# ==============================================================================
def _positivos_transicion(df_maestro_t, estado_t, estado_tp1):
    """
    Calcula, para una transicion t -> t+1, el numero de empresas SANAS en t y el
    numero de POSITIVOS y_estricto en t+1, replicando la definicion del pipeline:

      * SANA en t: Estado actual(t) in {ACTIVA, EN ETAPA PREOPERATIVA} y
        Flag_Quiebra_Patrimonial(t) == 0.
      * POSITIVO (colapso en t+1):
          - colapso legal: Estado actual(t+1) NO sano (presente en t+1 y fuera de
            los estados sanos), O
          - colapso patrimonial: para armar la parte patrimonial haria falta el
            maestro de t+1; en este diagnostico de FEAT-001 se usa SOLO la parte
            legal observable con las caratulas (cota conservadora que ya dimensiona
            el aumento de positivos). La parte patrimonial de t+1 la completa FEAT-002
            cuando cruce con el maestro reconstruido de t+1.

    Anti-fuga: el estado de t se usa solo para filtrar las sanas; el estado de t+1
    solo para definir la etiqueta. No se mezclan features de t+1 en X.

    Parametros
    ----------
    df_maestro_t : pandas.DataFrame
        Maestro reconstruido del ano t (aporta Flag_Quiebra_Patrimonial de t).
    estado_t, estado_tp1 : pandas.DataFrame
        Estado actual (NIT, estado_actual_norm) de t y t+1.

    Devuelve
    --------
    dict con empresas_sanas_t, positivos_legales_tp1 y cobertura de cruce.
    """
    # Estado de t por NIT (para filtrar sanas) unido al Flag patrimonial de t.
    base_t = df_maestro_t[["NIT", "Flag_Quiebra_Patrimonial"]].copy()
    base_t["NIT"] = normalizar_nit(base_t["NIT"])
    base_t = base_t.merge(estado_t, on="NIT", how="left")

    sana_t = base_t["estado_actual_norm"].isin(ESTADOS_SANOS_NORM) & (base_t["Flag_Quiebra_Patrimonial"] == 0)
    sanas = base_t[sana_t].copy()
    n_sanas = int(len(sanas))

    # Cruce con el estado de t+1 para evaluar el colapso legal.
    cruce = sanas.merge(estado_tp1.rename(columns={"estado_actual_norm": "estado_tp1"}), on="NIT", how="inner")
    n_con_estado_tp1 = int(len(cruce))
    colapso_legal = ~cruce["estado_tp1"].isin(ESTADOS_SANOS_NORM)
    n_positivos_legales = int(colapso_legal.sum())

    return {
        "empresas_sanas_t": n_sanas,
        "sanas_con_estado_t_mas_1": n_con_estado_tp1,
        "positivos_legales_t_mas_1": n_positivos_legales,
    }


def generar_diagnostico_transiciones(maestros=None, estado_por_anio=None) -> dict:
    """
    Construye y persiste en ``salidas/diagnostico_transiciones.json`` el
    dimensionamiento de positivos de las transiciones historicas 2021->2022,
    2022->2023 y 2023->2024, para estimar el aumento de positivos antes de entrenar.

    Parametros
    ----------
    maestros : dict[int, DataFrame] | None
        Maestros reconstruidos por ano. Si None, se construyen (o cargan de cache).
    estado_por_anio : dict[int, DataFrame] | None
        Estado actual por ano. Si None, se construye (o carga de cache).

    Devuelve
    --------
    dict con el diagnostico completo (ya serializado a JSON en disco).
    """
    if maestros is None:
        maestros = {anio: construir_maestro_ratios_anio(anio) for anio in ANIOS_HISTORICOS}
    if estado_por_anio is None:
        estado_por_anio = construir_estado_actual_por_anio()

    transiciones = [(2021, 2022), (2022, 2023), (2023, 2024)]
    detalle_transiciones = {}
    total_positivos_extra = 0
    for (t, tp1) in transiciones:
        det = _positivos_transicion(maestros[t], estado_por_anio[t], estado_por_anio[tp1])
        detalle_transiciones[f"{t}_a_{tp1}"] = det
        total_positivos_extra += det["positivos_legales_t_mas_1"]

    # Referencia: el panel actual 2024->2025 aporta 304 positivos (PISO).
    positivos_panel_actual = 304
    diagnostico = {
        "descripcion": (
            "Dimensionamiento de positivos y_estricto por transicion historica. La "
            "columna positivos_legales_t_mas_1 usa SOLO la parte LEGAL de la etiqueta "
            "(cambio de estado a no-sano en t+1), observable desde las caratulas 10000; "
            "es una cota conservadora. La parte patrimonial (Flag_Quiebra_Patrimonial "
            "de t+1) la completa FEAT-002 al cruzar con el maestro reconstruido de t+1."
        ),
        "estados_sanos": sorted(ESTADOS_SANOS_NORM),
        "nits_por_maestro": {int(a): int(maestros[a]["NIT"].nunique()) for a in ANIOS_HISTORICOS},
        "shape_por_maestro": {int(a): list(maestros[a].shape) for a in ANIOS_HISTORICOS},
        "estados_por_anio": {
            int(a): estado_por_anio[a]["estado_actual_norm"].value_counts().to_dict()
            for a in ANIOS_ESTADO
        },
        "transiciones": detalle_transiciones,
        "total_positivos_legales_extra": int(total_positivos_extra),
        "positivos_panel_actual_2024_2025": positivos_panel_actual,
        "factor_aumento_estimado_vs_panel_actual": round(
            (total_positivos_extra + positivos_panel_actual) / positivos_panel_actual, 3
        ),
    }

    ruta_json = SALIDAS_DIR / NOMBRE_DIAGNOSTICO_TRANSICIONES
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False)
    print(f"Diagnostico de transiciones escrito en: {ruta_json}")
    return diagnostico


# ==============================================================================
# BLOQUE DE VERIFICACION EJECUTABLE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 88)
    print("FEAT-001: RECONSTRUCCION DE MAESTROS HISTORICOS CON RATIOS FASE-4")
    print("=" * 88)

    maestros = {}
    for anio in ANIOS_HISTORICOS:
        df = construir_maestro_ratios_anio(anio)
        maestros[anio] = df
        print(f"  Maestro {anio}: shape={df.shape} | NITs unicos={df['NIT'].nunique():,}")

    print("\nConstruyendo serie de Estado actual por ano (caratulas 10000)...")
    estado_por_anio = construir_estado_actual_por_anio()
    for anio in ANIOS_ESTADO:
        conteo = estado_por_anio[anio]["estado_actual_norm"].value_counts()
        print(f"\n  Estado actual {anio} ({conteo.sum():,} empresas):")
        for estado, n in conteo.items():
            print(f"    {estado:<32}: {n:>7,}")

    print("\nDimensionando positivos por transicion historica...")
    diag = generar_diagnostico_transiciones(maestros, estado_por_anio)

    print("\nResumen de transiciones (parte legal de y_estricto):")
    for clave, det in diag["transiciones"].items():
        print(
            f"  {clave}: sanas_t={det['empresas_sanas_t']:,} | "
            f"con_estado_t+1={det['sanas_con_estado_t_mas_1']:,} | "
            f"positivos_legales_t+1={det['positivos_legales_t_mas_1']:,}"
        )
    print(
        f"\nTotal positivos legales extra (3 transiciones): "
        f"{diag['total_positivos_legales_extra']:,} "
        f"(panel actual 2024->2025 = {diag['positivos_panel_actual_2024_2025']}; "
        f"factor estimado x{diag['factor_aumento_estimado_vs_panel_actual']})"
    )

    print("\nProceso FEAT-001 completado con exito.")
