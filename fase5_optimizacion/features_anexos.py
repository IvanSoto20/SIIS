# -*- coding: utf-8 -*-
"""
FEATURES DE ANEXOS NO EXPLOTADOS DEL SIIS (FEAT-002): PALANCA 1 DE INFORMACION
==============================================================================
Objetivo de negocio
--------------------
La ZONA ROJA del modelo de insolvencia esta limitada por la escasez de senales
finas en el corte t. El SIIS publica anexos con informacion granular que el
pipeline aun NO explota. Esta feature los lee a fondo y deriva variables
predictoras acotadas para que, detras de una bandera, sean CANDIDATAS a entrar en
la matriz X:

  * 520000 (Estado de flujos de efectivo, metodo indirecto): el maestro ya trae los
    3 flujos NETOS (operacion/inversion/financiacion). Lo NUEVO aqui son los
    COMPONENTES granulares (depreciacion/amortizacion, ajustes por deterioro,
    ajustes por provisiones, ajustes por costos financieros, intereses pagados,
    prestamos recibidos/reembolsados, dividendos pagados, efectivo al principio y al
    final del periodo, incremento neto de efectivo) y los derivados de CALIDAD DE
    CAJA que miden si la utilidad se convierte en efectivo y si la caja alcanza para
    el servicio de la deuda. Senales clasicas de insolvencia inminente.
  * 801500 (Deterioro de activos): perdidas por deterioro reconocidas en resultados.
    Cobertura dispersa (~1272 de ~23000 empresas), por lo que la inmensa mayoria NO
    reporta: imputacion-neutral 0 + flag de cobertura es ESENCIAL.
  * 801700 (Otras provisiones y pasivos contingentes): provisiones por procesos
    legales, reestructuracion, garantias, contratos onerosos. Cobertura ~1074 de
    ~23000: mismo tratamiento neutral + flag.

Estrategia de faltantes (IMPUTACION-NEUTRAL + FLAGS DE COBERTURA)
-----------------------------------------------------------------
Identica filosofia a ``features_multianio.py`` (FEAT-005): se usa imputacion-neutral
+ banderas de cobertura, NO el camino NaN-nativo. Razones:

  1. Conserva el contrato ``assert X.isnull().sum().sum() == 0`` de la matriz X sin
     listas blancas ni relajaciones, reduciendo el riesgo de regresion.
  2. Mantiene operativo a ``RandomForestClassifier`` de scikit-learn, que NO acepta
     NaN (lanzaria ``ValueError: Input X contains NaN``); asi el super-ensamble de 4
     modelos devuelve una probabilidad valida en [0, 1] para toda empresa, incluida
     la que no reporta ningun anexo.
  3. El requisito de "entrada parcial" del usuario se cumple: las BANDERAS de
     cobertura (``Flag_Reporta_Flujos``, ``Flag_Reporta_Deterioro``,
     ``Flag_Reporta_Provisiones``) capturan explicitamente la ausencia de reporte, de
     modo que el modelo distingue "monto cero real" de "no reporto".

Convencion por tipo de feature:
  * Montos (componentes de flujo, deterioro, provisiones): si la empresa no reporta
    el anexo, el monto se imputa 0 (neutral: no inventa deterioro ni caja).
  * Ratios derivados: via ``division_segura`` (proteccion /0, NaN, inf, con ``.clip``)
    y ``numpy.clip`` adicional; si falta el componente, el ratio toma su valor neutro.
  * Banderas: 0/1, donde 0 incluye tanto "no reporta" como "reporta pero monto 0".

Anti-fuga (autoritativo)
------------------------
Las variables del modelo 2024 -> 2025 se leen EXCLUSIVAMENTE del corte 2024
(carpeta ``SIIS/pymes_individuales-2024``); para la proyeccion del semaforo 2026 se
leen del corte 2025 (carpeta principal ``SIIS/SIIS``). Jamas el corte 2025 entra en
el modelo 2024 -> 2025. ``construir_features_anexos`` elige la carpeta segun
``anio_t`` y ninguna columna de salida lleva el literal '2025' en su nombre, de modo
que el assert anti-fuga del pipeline (``not any('2025' in str(col) for col in
X.columns)``) sigue pasando.

Reutilizacion de codigo
-----------------------
Se replica el ESCUDO PRE-MERGE de ``maestros_historicos._leer_anexo`` (usecols,
limpiar NIT, filtro de 'Concepto', dedupe por 'Fecha de Corte' mas reciente
keep='last', asserts de duplicados/sin-NIT) y se reutilizan por import las utilidades
ya auditadas del pipeline activo (``normalizar_nit``, ``buscar_archivo_siis``,
``division_segura``, ``SIIS_DIR``, ``SALIDAS_DIR``). El import es seguro porque
``script_cinco_optimizado`` protege su ejecucion con el guardian ``__main__``.
"""

import json
from typing import Dict, List

import numpy as np
import pandas as pd

from script_cinco_optimizado import (
    SALIDAS_DIR,
    SIIS_DIR,
    buscar_archivo_siis,
    division_segura,
    normalizar_nit,
)

# ------------------------------------------------------------------------------
# PARAMETROS GLOBALES
# ------------------------------------------------------------------------------
# Mapeo anio del corte t -> carpeta del SIIS de la que se leen los anexos. Anti-fuga:
# el modelo 2024->2025 usa SOLO el corte 2024; la proyeccion 2026 usa el corte 2025.
CARPETA_POR_ANIO = {
    2024: SIIS_DIR / "pymes_individuales-2024",
    2025: SIIS_DIR,
}

# Umbral monetario para estabilizar denominadores de efectivo inicial muy pequenos
# en la variable de quema de caja (evita cocientes explosivos por saldos cercanos a 0).
UMBRAL_EFECTIVO = 1.0

# ------------------------------------------------------------------------------
# NOMBRES EXACTOS DE COLUMNAS DE LOS ANEXOS (verificados leyendo los .xlsx)
# ------------------------------------------------------------------------------
# 520000: componentes granulares NUEVOS (el maestro ya tenia los 3 flujos netos).
COL_520000 = {
    "flujo_operativo": "Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)",
    "depreciacion": "( + ) Ajustes por gastos de depreciación y amortización (AdjustmentsForDepreciationAndAmortisationExpense)",
    "deterioro_conciliacion": "(+/-) Ajustes por deterioro de valor (reversiones de pérdidas por deterioro de valor) reconocidas en el resultado del periodo (AdjustmentsForImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss)",
    "ajuste_provisiones": "(+) Ajustes por provisiones (AdjustmentsForProvisions)",
    "ajuste_costos_fin": "(+) Ajustes por costos financieros (AdjustmentsForFinanceCosts)",
    "intereses_pagados": "(-) Intereses pagados (InterestPaidClassifiedAsFinancingActivities)",
    "prestamos_recibidos": "(+) Importes procedentes de préstamos (ProceedsFromBorrowingsClassifiedAsFinancingActivities)",
    "reembolsos_prestamos": "(-) Reembolsos de préstamos (RepaymentsOfBorrowingsClassifiedAsFinancingActivities)",
    "dividendos_pagados": "(-) Dividendos pagados (DividendsPaidClassifiedAsFinancingActivities)",
    "efectivo_inicial": "Efectivo y equivalentes al efectivo al principio del periodo (CashAndCashEquivalents)",
    "efectivo_final": "Efectivo y equivalentes al efectivo al final del periodo (CashAndCashEquivalents)",
    "incremento_neto_efectivo": "Incremento (disminución) neto de efectivo y equivalentes al efectivo (IncreaseDecreaseInCashAndCashEquivalents)",
}

# 801500 Deterioro: columna 'Concepto' multi-fila; nos quedamos con el total del
# deterioro reconocido en resultados del periodo.
CONCEPTO_DETERIORO = "Pérdidas por deterioro de valor reconocidas en el resultado del periodo"
COL_801500_TOTAL = "Total deterioro [miembro] (TotalDeterioro_Miembro)"

# 801700 Provisiones: columna 'Concepto' multi-fila; nos quedamos con el saldo total
# de otras provisiones al final del periodo y con las columnas de miembros (procesos
# legales en particular, como senal de litigios materiales).
CONCEPTO_PROVISION_TOTAL = "Total otras provisiones al final del periodo"
COL_801700_MIEMBROS = [
    "Provisión por garantías [miembro] (WarrantyProvisionMember)",
    "Provisión por reestructuración [miembro] (RestructuringProvisionMember)",
    "Provisiones por procesos legales [miembro] (LegalProceedingsProvisionMember)",
    "Provisión por contratos onerosos [miembro] (OnerousContractsProvisionMember)",
    "Desmantelamiento Miembro (DesmantelamientoMiembro)",
    "Otra provisión relacionada con el medioambiente [miembro] (Otra_provisión_relacionada_con_el_medioambiente_miembro)",
    "Otras provisiones diversas [miembro] (MiscellaneousOtherProvisionsMember)",
    "Otras provisiones [miembro] (OtherProvisionsMember)",
]
COL_801700_LITIGIOS = "Provisiones por procesos legales [miembro] (LegalProceedingsProvisionMember)"

# Nombres EXACTOS de las columnas que produce este modulo. Se expone como constante
# para que la ablacion/adopcion (FEAT-004) pueda quitarlas o anadirlas sin re-derivar
# la lista a mano, y para documentar el contrato del bloque.
COLUMNAS_ANEXOS = [
    # --- 520000: calidad de caja y servicio de deuda ---
    "Flag_Reporta_Flujos",
    "Calidad_Ganancias_Caja",
    "Flag_Flujo_Operativo_Negativo",
    "Cobertura_Deuda_Con_Caja",
    "Cobertura_Intereses_Con_Caja",
    "Quema_Caja_Pct",
    "Flag_Quema_Caja",
    "Carga_Servicio_Deuda",
    "Peso_Depreciacion_En_Conciliacion",
    "Peso_Deterioro_En_Conciliacion",
    # --- 801500: deterioro de activos ---
    "Flag_Reporta_Deterioro",
    "Deterioro_Sobre_Activos",
    "Flag_Deterioro_Material",
    # --- 801700: provisiones y contingencias ---
    "Flag_Reporta_Provisiones",
    "Provisiones_Sobre_Patrimonio",
    "Flag_Provision_Litigios_Material",
]


# ==============================================================================
# UTILIDADES DE LECTURA FORENSE DE ANEXOS (replican maestros_historicos._leer_anexo)
# ==============================================================================
def _verificar_duplicados(df: pd.DataFrame, nombre: str) -> None:
    """Aborta si quedan NITs duplicados tras la deduplicacion forense."""
    duplicados = df[df["NIT"].duplicated(keep=False)]
    assert duplicados.empty, (
        f"ERROR: [{nombre}] tiene NITs duplicados tras la deduplicacion forense: "
        f"{duplicados['NIT'].value_counts().to_dict()}"
    )


def _verificar_sin_nit(df: pd.DataFrame, nombre: str) -> None:
    """Aborta si hay filas sin NIT."""
    assert df["NIT"].isnull().sum() == 0, f"ERROR: [{nombre}] trae filas sin NIT."


def _leer_anexo(
    carpeta,
    prefijo: str,
    columnas: List[str],
    nombre: str,
    filtro_concepto: str = None,
    columna_concepto: str = "Concepto",
) -> pd.DataFrame:
    """
    Lee un anexo del SIIS replicando el 'ESCUDO PRE-MERGE' de
    ``maestros_historicos._leer_anexo``: carga solo las columnas necesarias, limpia
    el NIT, filtra por un 'Concepto' concreto cuando aplica, se queda con la Fecha de
    Corte mas reciente por NIT (keep='last') y valida unicidad.

    Parametros
    ----------
    carpeta : pathlib.Path
        Carpeta del corte (``SIIS/SIIS`` para 2025, ``.../pymes_individuales-2024``).
    prefijo : str
        Prefijo numerico del anexo (p.ej. '520000'); se resuelve con
        ``buscar_archivo_siis`` (glob determinista, un unico archivo).
    columnas : list[str]
        Columnas a materializar. Debe incluir 'NIT' y 'Fecha de Corte' (y la columna
        de concepto cuando se filtra por ella).
    nombre : str
        Etiqueta para los mensajes de assert.
    filtro_concepto : str | None
        Si se indica, conserva solo las filas cuya columna de concepto es igual a este
        valor (comparacion exacta sobre el texto original, identica a la Fase 1).
    columna_concepto : str
        Nombre de la columna sobre la que aplica ``filtro_concepto`` (por defecto
        'Concepto').

    Devuelve
    --------
    pandas.DataFrame
        Una fila por NIT con las columnas de datos pedidas (se eliminan la de concepto
        y 'Fecha de Corte', usadas solo como filtro/orden).

    Notas tecnicas (pandas)
    -----------------------
    * ``pandas.read_excel`` con ``usecols`` como lista materializa solo esas columnas,
      acelerando la lectura de estos .xlsx grandes (documentacion de pandas).
    * ``drop_duplicates(keep='last')`` tras ``sort_values`` por 'Fecha de Corte'
      conserva el reporte mas reciente por NIT (deduplicacion forense, identica a la
      Fase 1). 520000 no trae columna 'Periodo' (es flujo del ejercicio), por lo que
      solo se deduplica por Fecha de Corte, como hace el maestro con 520000.
    """
    ruta = buscar_archivo_siis(carpeta, prefijo)
    df = pd.read_excel(ruta, usecols=columnas)
    df["NIT"] = normalizar_nit(df["NIT"])

    if filtro_concepto is not None:
        df = df[df[columna_concepto] == filtro_concepto]

    df = df.sort_values(by=["NIT", "Fecha de Corte"]).drop_duplicates(subset=["NIT"], keep="last")

    columnas_a_borrar = [c for c in [columna_concepto, "Fecha de Corte"] if c in df.columns]
    df = df.drop(columns=columnas_a_borrar)

    _verificar_duplicados(df, nombre)
    _verificar_sin_nit(df, nombre)
    return df.reset_index(drop=True)


def _leer_crudos_anexos(carpeta, anio: int, usar_cache: bool) -> Dict[str, pd.DataFrame]:
    """
    Lee (o carga de cache) los extractos crudos por NIT de los tres anexos del corte
    indicado y los devuelve en un dict {clave: DataFrame[NIT, <montos>]}.

    El cache es por anio y por anexo (``anexo_520000_<anio>.pkl``, etc.) para no releer
    los .xlsx en cada corrida; las lecturas de .xlsx son costosas (hallazgo del
    contexto). Si ``usar_cache`` es False se fuerza la relectura y se reescribe el pkl.
    """
    out: Dict[str, pd.DataFrame] = {}

    # --- 520000: flujos de efectivo (componentes granulares) ---
    nombre_cache_520 = f"anexo_520000_{anio}.pkl"
    ruta_520 = SALIDAS_DIR / nombre_cache_520
    if usar_cache and ruta_520.exists():
        df_520 = pd.read_pickle(ruta_520)
    else:
        columnas_520 = ["NIT", "Fecha de Corte"] + list(COL_520000.values())
        df_520 = _leer_anexo(carpeta, "520000", columnas_520, f"Flujos 520000 {anio}")
        # Renombrado a alias estables e independientes del idioma/XBRL.
        df_520 = df_520.rename(columns={v: k for k, v in COL_520000.items()})
        df_520.to_pickle(ruta_520)
    out["flujos"] = df_520

    # --- 801500: deterioro de activos (concepto multi-fila) ---
    nombre_cache_515 = f"anexo_801500_{anio}.pkl"
    ruta_515 = SALIDAS_DIR / nombre_cache_515
    if usar_cache and ruta_515.exists():
        df_515 = pd.read_pickle(ruta_515)
    else:
        columnas_515 = ["NIT", "Fecha de Corte", "Concepto", COL_801500_TOTAL]
        df_515 = _leer_anexo(
            carpeta, "801500", columnas_515, f"Deterioro 801500 {anio}",
            filtro_concepto=CONCEPTO_DETERIORO,
        )
        df_515 = df_515.rename(columns={COL_801500_TOTAL: "deterioro_total"})
        df_515.to_pickle(ruta_515)
    out["deterioro"] = df_515

    # --- 801700: provisiones y contingencias (concepto multi-fila) ---
    nombre_cache_517 = f"anexo_801700_{anio}.pkl"
    ruta_517 = SALIDAS_DIR / nombre_cache_517
    if usar_cache and ruta_517.exists():
        df_517 = pd.read_pickle(ruta_517)
    else:
        columnas_517 = ["NIT", "Fecha de Corte", "Concepto"] + COL_801700_MIEMBROS
        df_517 = _leer_anexo(
            carpeta, "801700", columnas_517, f"Provisiones 801700 {anio}",
            filtro_concepto=CONCEPTO_PROVISION_TOTAL,
        )
        # Provision total = suma de los miembros al final del periodo (la fila del
        # concepto 'Total otras provisiones al final del periodo'). Se suman todas las
        # columnas de miembro con fillna(0) porque un miembro vacio es $0 de provision.
        miembros_presentes = [c for c in COL_801700_MIEMBROS if c in df_517.columns]
        df_517["provision_total"] = df_517[miembros_presentes].fillna(0).sum(axis=1)
        col_litigios = "provision_litigios"
        if COL_801700_LITIGIOS in df_517.columns:
            df_517[col_litigios] = df_517[COL_801700_LITIGIOS].fillna(0)
        else:
            df_517[col_litigios] = 0.0
        df_517 = df_517[["NIT", "provision_total", col_litigios]]
        df_517.to_pickle(ruta_517)
    out["provisiones"] = df_517

    return out


# ==============================================================================
# CONSTRUCTOR DEL BLOQUE DE FEATURES DE ANEXOS
# ==============================================================================
def _alinear_por_nit(df_anexo: pd.DataFrame, nit_key: pd.Series, columnas: List[str]) -> pd.DataFrame:
    """
    Alinea un extracto de anexo (una fila por NIT) al orden/indice de ``nit_key``
    mediante ``set_index('NIT').reindex``. Los NITs sin reporte quedan NaN; se anade
    un indicador ``_presente`` (1 si el NIT reporto el anexo, 0 si no).

    Notas tecnicas (pandas)
    -----------------------
    * ``DataFrame.reindex`` deja NaN donde el indice objetivo no tiene correspondencia
      (documentacion de pandas: "Conform ... to new index ... with optional filling").
      Esa ausencia es justo lo que se traduce en imputacion-neutral + flag de cobertura.
    """
    idx_anexo = normalizar_nit(df_anexo["NIT"])
    alineado = (
        df_anexo[columnas]
        .set_axis(idx_anexo, axis=0)
        .reindex(nit_key.values)
    )
    alineado.index = nit_key.index
    # Un NIT reporto el anexo si al menos la primera columna de monto no es NaN.
    alineado["_presente"] = (~alineado[columnas[0]].isna()).astype(int)
    return alineado


def construir_features_anexos(
    df_maestro_t: pd.DataFrame,
    anio_t: int,
    usar_cache: bool = True,
) -> pd.DataFrame:
    """
    Construye el bloque de columnas de FEATURES DE ANEXOS indexado igual que
    ``df_maestro_t`` (mismo indice de filas), listo para concatenarse a la matriz X.

    Parametros
    ----------
    df_maestro_t : pandas.DataFrame
        Corte t (debe tener columna 'NIT' y los saldos contables del año t). Define el
        universo de empresas y el orden/indice de filas de salida.
    anio_t : int
        Año del corte t: 2024 para el entrenamiento 2024->2025 (lee el corte 2024);
        2025 para la proyeccion del semaforo 2026 (lee el corte 2025). Anti-fuga: la
        carpeta se elige por ``anio_t``, de modo que el corte 2025 nunca alimenta el
        modelo 2024->2025.
    usar_cache : bool
        Si True, reutiliza los extractos .pkl de ``salidas/`` (lectura de .xlsx cara).

    Devuelve
    --------
    pandas.DataFrame
        Columnas = ``COLUMNAS_ANEXOS``, indice = ``df_maestro_t.index``, SIN NaN
        (imputacion-neutral + flags de cobertura). No dropea filas: una empresa que no
        reporta ningun anexo recibe todo neutral (montos 0, ratios en su valor neutro,
        flags en 0).
    """
    assert anio_t in CARPETA_POR_ANIO, (
        f"anio_t={anio_t} no tiene carpeta de anexos mapeada (validos: {sorted(CARPETA_POR_ANIO)})."
    )
    carpeta = CARPETA_POR_ANIO[anio_t]
    crudos = _leer_crudos_anexos(carpeta, anio_t, usar_cache)

    idx = df_maestro_t.index
    nit_key = normalizar_nit(df_maestro_t["NIT"])
    nit_key.index = idx

    # --------------------------------------------------------------------------
    # Saldos del corte t (denominadores). Se toman del propio maestro con la MISMA
    # imputacion-neutral que usa ``construir_matriz_caracteristicas`` (fillna(0) +
    # clip para los no negativos), de modo que no se introduce informacion externa.
    # --------------------------------------------------------------------------
    activos = df_maestro_t["Total de activos (Assets)"].fillna(0).clip(lower=0).astype(float).to_numpy()
    patrimonio = df_maestro_t["Patrimonio total (Equity)"].fillna(0).clip(lower=0).astype(float).to_numpy()
    pas_corr = df_maestro_t["Pasivos corrientes totales (CurrentLiabilities)"].fillna(0).clip(lower=0).astype(float).to_numpy()
    ingresos = df_maestro_t["Ingresos de actividades ordinarias (Revenue)"].fillna(0).clip(lower=0).astype(float).to_numpy()
    utilidad = df_maestro_t["Ganancia (pérdida) (ProfitLoss)"].fillna(0).astype(float).to_numpy()
    costos_fin = df_maestro_t["Costos financieros (FinanceCosts)"].fillna(0).clip(lower=0).astype(float).to_numpy()

    X = pd.DataFrame(index=idx)

    # ==========================================================================
    # BLOQUE 520000: CALIDAD DE CAJA Y SERVICIO DE DEUDA
    # ==========================================================================
    cols_flujos = list(COL_520000.keys())
    flujos = _alinear_por_nit(crudos["flujos"], nit_key, cols_flujos)
    reporta_flujos = flujos["_presente"].to_numpy()

    # Montos imputados de forma neutral: NaN (no reporta) -> 0.
    flujo_op = np.nan_to_num(flujos["flujo_operativo"].to_numpy(dtype=float), nan=0.0)
    depreciacion = np.nan_to_num(flujos["depreciacion"].to_numpy(dtype=float), nan=0.0)
    deterioro_concil = np.nan_to_num(flujos["deterioro_conciliacion"].to_numpy(dtype=float), nan=0.0)
    intereses_pagados = np.abs(np.nan_to_num(flujos["intereses_pagados"].to_numpy(dtype=float), nan=0.0))
    reembolsos = np.abs(np.nan_to_num(flujos["reembolsos_prestamos"].to_numpy(dtype=float), nan=0.0))
    efectivo_ini = np.nan_to_num(flujos["efectivo_inicial"].to_numpy(dtype=float), nan=0.0)
    efectivo_fin = np.nan_to_num(flujos["efectivo_final"].to_numpy(dtype=float), nan=0.0)
    incremento_neto = np.nan_to_num(flujos["incremento_neto_efectivo"].to_numpy(dtype=float), nan=0.0)

    # Flag_Reporta_Flujos: cobertura del anexo 520000 (1 si reporto, 0 si no).
    X["Flag_Reporta_Flujos"] = reporta_flujos.astype(int)

    # Calidad_Ganancias_Caja: flujo operativo / utilidad neta. Mide si las ganancias
    # contables se convierten en caja (un ratio bajo o negativo con utilidad positiva
    # es senal de calidad de resultados pobre). Solo se calcula con utilidad > 0; si
    # la utilidad es <= 0, el ratio no es interpretable y se fija al neutro 0.0.
    # division_segura acota a [-5, 5] y blinda contra /0, NaN e inf (numpy where +
    # nan_to_num + clip, documentado en el pipeline).
    calidad = division_segura(flujo_op, np.where(utilidad > 0, utilidad, np.nan), 0.0, -5.0, 5.0)
    X["Calidad_Ganancias_Caja"] = calidad

    # Flag_Flujo_Operativo_Negativo: la operacion consume caja (senal de asfixia).
    # Solo tiene sentido si la empresa reporto flujos; si no reporto, 0 (neutral).
    X["Flag_Flujo_Operativo_Negativo"] = ((reporta_flujos == 1) & (flujo_op < 0)).astype(int)

    # Cobertura_Deuda_Con_Caja: flujo operativo / pasivo corriente (cuanto del pasivo
    # de corto plazo cubre la caja generada por la operacion). Acotado a [0, 5].
    X["Cobertura_Deuda_Con_Caja"] = division_segura(flujo_op, pas_corr, 0.0, 0.0, 5.0)

    # Cobertura_Intereses_Con_Caja: flujo operativo / intereses pagados. Si no hay
    # intereses pagados en el anexo, se usa como respaldo los costos financieros del
    # estado de resultados (ambos miden la carga financiera). Acotado a [0, 10].
    denom_intereses = np.where(intereses_pagados > 0, intereses_pagados, costos_fin)
    X["Cobertura_Intereses_Con_Caja"] = division_segura(flujo_op, denom_intereses, 0.0, 0.0, 10.0)

    # Quema_Caja_Pct: variacion relativa de la caja en el periodo
    # (efectivo_final - efectivo_inicial) / max(efectivo_inicial, umbral). Negativo =
    # la empresa quemo caja. El umbral estabiliza denominadores diminutos. Acotado a
    # [-1, 5]. Cuando no reporta flujos, ambos efectivos son 0 => 0.0 (neutral).
    denom_efectivo = np.maximum(efectivo_ini, UMBRAL_EFECTIVO)
    quema = division_segura(efectivo_fin - efectivo_ini, denom_efectivo, 0.0, -1.0, 5.0)
    X["Quema_Caja_Pct"] = quema

    # Flag_Quema_Caja: el incremento neto de efectivo del periodo fue negativo (la
    # empresa termino el ano con menos caja). Solo si reporto flujos.
    X["Flag_Quema_Caja"] = ((reporta_flujos == 1) & (incremento_neto < 0)).astype(int)

    # Carga_Servicio_Deuda: (reembolsos de prestamos + intereses pagados) / ingresos.
    # Mide cuanto de la facturacion se va al servicio de la deuda. Acotado a [0, 5].
    servicio = reembolsos + intereses_pagados
    X["Carga_Servicio_Deuda"] = division_segura(servicio, ingresos, 0.0, 0.0, 5.0)

    # Peso_Depreciacion_En_Conciliacion: ajuste de depreciacion/amortizacion / activos.
    # Proxy de intensidad de activos de larga vida en la conciliacion de caja. [0, 1].
    X["Peso_Depreciacion_En_Conciliacion"] = division_segura(
        np.abs(depreciacion), activos, 0.0, 0.0, 1.0
    )

    # Peso_Deterioro_En_Conciliacion: ajuste por deterioro de valor / activos. Es un
    # PROXY de deterioro reconocido disponible INCLUSO cuando la empresa no diligencia
    # el anexo 801500 (el ajuste del flujo esta en 520000). Acotado a [0, 1] en valor
    # absoluto (se toma magnitud porque interesa el tamano del deterioro, no el signo).
    X["Peso_Deterioro_En_Conciliacion"] = division_segura(
        np.abs(deterioro_concil), activos, 0.0, 0.0, 1.0
    )

    # ==========================================================================
    # BLOQUE 801500: DETERIORO DE ACTIVOS
    # ==========================================================================
    deterioro = _alinear_por_nit(crudos["deterioro"], nit_key, ["deterioro_total"])
    reporta_deterioro = deterioro["_presente"].to_numpy()
    monto_deterioro = np.abs(np.nan_to_num(deterioro["deterioro_total"].to_numpy(dtype=float), nan=0.0))

    X["Flag_Reporta_Deterioro"] = reporta_deterioro.astype(int)
    # Deterioro_Sobre_Activos: perdida por deterioro reconocida / activos. [0, 1].
    X["Deterioro_Sobre_Activos"] = division_segura(monto_deterioro, activos, 0.0, 0.0, 1.0)
    # Flag_Deterioro_Material: la empresa reconocio deterioro > 0 en el periodo.
    X["Flag_Deterioro_Material"] = ((reporta_deterioro == 1) & (monto_deterioro > 0)).astype(int)

    # ==========================================================================
    # BLOQUE 801700: PROVISIONES Y CONTINGENCIAS
    # ==========================================================================
    provisiones = _alinear_por_nit(crudos["provisiones"], nit_key, ["provision_total", "provision_litigios"])
    reporta_provisiones = provisiones["_presente"].to_numpy()
    monto_provision = np.abs(np.nan_to_num(provisiones["provision_total"].to_numpy(dtype=float), nan=0.0))
    monto_litigios = np.abs(np.nan_to_num(provisiones["provision_litigios"].to_numpy(dtype=float), nan=0.0))

    X["Flag_Reporta_Provisiones"] = reporta_provisiones.astype(int)
    # Provisiones_Sobre_Patrimonio: saldo de otras provisiones / patrimonio. Mide el
    # peso de pasivos estimados (litigios, garantias, etc.) sobre el colchon. [0, 5].
    X["Provisiones_Sobre_Patrimonio"] = division_segura(monto_provision, patrimonio, 0.0, 0.0, 5.0)
    # Flag_Provision_Litigios_Material: hay provision por procesos legales > 0 (senal
    # de contingencia judicial relevante).
    X["Flag_Provision_Litigios_Material"] = ((reporta_provisiones == 1) & (monto_litigios > 0)).astype(int)

    # --------------------------------------------------------------------------
    # BLINDAJE FINAL: imputacion-neutral garantiza CERO NaN en el bloque.
    # --------------------------------------------------------------------------
    for col in COLUMNAS_ANEXOS:
        assert col in X.columns, f"Falta la columna de anexos esperada: {col}"
    assert list(X.columns) == COLUMNAS_ANEXOS, (
        "El orden/conjunto de columnas de anexos no coincide con COLUMNAS_ANEXOS."
    )
    assert X.isnull().sum().sum() == 0, (
        f"NaN detectado en el bloque de anexos: {X.columns[X.isnull().any()].tolist()}"
    )
    # Anti-fuga: ninguna columna del bloque lleva el literal '2025' en su nombre, de
    # modo que el assert anti-fuga del pipeline sigue pasando (la seleccion de la
    # carpeta por anio_t es lo que garantiza que el corte 2025 no entra en 2024->2025).
    assert not any("2025" in str(col) for col in X.columns), (
        "ALERTA ANTI-FUGA: alguna columna de anexos lleva el literal '2025'."
    )
    return X


# ==============================================================================
# GUARD __main__: DIAGNOSTICO DE COBERTURA Y ESTADISTICOS
# ==============================================================================
def _estadisticos_bloque(X: pd.DataFrame) -> dict:
    """Resume medias/medianas de los ratios y % de flags activos del bloque."""
    resumen = {}
    for col in X.columns:
        serie = X[col]
        if col.startswith("Flag_"):
            resumen[col] = {
                "tipo": "flag",
                "pct_activos": round(float(serie.mean()) * 100.0, 4),
                "n_activos": int(serie.sum()),
            }
        else:
            resumen[col] = {
                "tipo": "ratio",
                "media": round(float(serie.mean()), 6),
                "mediana": round(float(serie.median()), 6),
                "min": round(float(serie.min()), 6),
                "max": round(float(serie.max()), 6),
            }
    return resumen


def _diagnostico() -> dict:
    """
    Construye el bloque de features de anexos para los cortes 2024 y 2025, mide la
    cobertura de cada anexo (cuantas empresas reportan) y los estadisticos basicos de
    cada variable derivada, y persiste todo en ``salidas/diagnostico_features_anexos.json``.
    """
    import pandas as pd  # import local para claridad del guard

    diagnostico = {
        "proposito": (
            "PALANCA 1 (FEAT-002): leer a fondo los anexos 520000/801500/801700 del "
            "SIIS (no explotados) para los cortes 2024 y 2025, derivar variables de "
            "calidad de caja, deterioro y provisiones con imputacion-neutral + flags "
            "de cobertura, y exponerlas como candidatas a X detras de la bandera "
            "USAR_FEATURES_ANEXOS (default False en esta feature)."
        ),
        "columnas_derivadas": list(COLUMNAS_ANEXOS),
        "anti_fuga": (
            "El modelo 2024->2025 usa SOLO el corte 2024; la proyeccion 2026 usa el "
            "corte 2025. Ninguna columna lleva el literal '2025' en su nombre."
        ),
        "cortes": {},
    }

    rutas_maestros = {
        2024: "df_maestro_con_ratios_2024.pkl",
        2025: "df_maestro_con_ratios.pkl",
    }
    from script_cinco_optimizado import BASE_DIR

    for anio, nombre_pkl in rutas_maestros.items():
        df_maestro = pd.read_pickle(BASE_DIR / nombre_pkl)
        bloque = construir_features_anexos(df_maestro, anio_t=anio, usar_cache=True)
        n = len(df_maestro)
        diagnostico["cortes"][str(anio)] = {
            "n_empresas": int(n),
            "cobertura": {
                "reporta_flujos_520000": int(bloque["Flag_Reporta_Flujos"].sum()),
                "reporta_deterioro_801500": int(bloque["Flag_Reporta_Deterioro"].sum()),
                "reporta_provisiones_801700": int(bloque["Flag_Reporta_Provisiones"].sum()),
                "pct_reporta_flujos": round(float(bloque["Flag_Reporta_Flujos"].mean()) * 100.0, 4),
                "pct_reporta_deterioro": round(float(bloque["Flag_Reporta_Deterioro"].mean()) * 100.0, 4),
                "pct_reporta_provisiones": round(float(bloque["Flag_Reporta_Provisiones"].mean()) * 100.0, 4),
            },
            "estadisticos": _estadisticos_bloque(bloque),
            "sin_nan": bool(bloque.isnull().sum().sum() == 0),
        }

    ruta_json = SALIDAS_DIR / "diagnostico_features_anexos.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False)
    return diagnostico, ruta_json


if __name__ == "__main__":
    print("=" * 90)
    print("DIAGNOSTICO DE FEATURES DE ANEXOS (FEAT-002): 520000 / 801500 / 801700")
    print("=" * 90)
    diag, ruta = _diagnostico()
    for anio, info in diag["cortes"].items():
        cob = info["cobertura"]
        print(f"\nCorte {anio} ({info['n_empresas']:,} empresas):")
        print(f"  Reporta 520000 (flujos)      : {cob['reporta_flujos_520000']:>6,}  ({cob['pct_reporta_flujos']:.2f}%)")
        print(f"  Reporta 801500 (deterioro)   : {cob['reporta_deterioro_801500']:>6,}  ({cob['pct_reporta_deterioro']:.2f}%)")
        print(f"  Reporta 801700 (provisiones) : {cob['reporta_provisiones_801700']:>6,}  ({cob['pct_reporta_provisiones']:.2f}%)")
        print(f"  Bloque sin NaN               : {info['sin_nan']}")
    print(f"\nDiagnostico persistido en: {ruta}")
    print("\nProceso de diagnostico completado con exito.")
