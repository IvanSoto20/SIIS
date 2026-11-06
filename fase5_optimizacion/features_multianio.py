# -*- coding: utf-8 -*-
"""
FEATURES DE TENDENCIA PLURIANUAL (FEAT-005): VENTANA 2020..2024
==============================================================================
Objetivo
--------
Extender la filosofia del BLOQUE 2.C de ``construir_matriz_caracteristicas``
(tendencias t-1 -> t con ``Flag_Sin_Historia_Prev`` + imputacion neutral) a una
ventana de historia PLURIANUAL. Para el modelo 2024 -> 2025 la ventana abarca los
anios 2020..2024 (TODO <= 2024, CERO 2025). El objetivo de negocio, segun el
usuario, es que una empresa pueda aportar los datos que tenga disponibles (no
necesariamente todos los anios) y el modelo igual produzca una prediccion para el
anio siguiente. Por eso estas features:

  * NO dropean filas: una empresa sin historia plurianual recibe de todas formas
    su vector de features (y por tanto su prediccion).
  * Marcan COBERTURA de historia con banderas explicitas
    (``Anios_Historia_Disponible``, ``Flag_Sin_Historia_3A``,
    ``Flag_Historia_Parcial``), de modo que el modelo pueda condicionar su lectura
    de las tendencias a cuanta historia real existe.

Estrategia de faltantes (decidida y documentada POR COLUMNA)
------------------------------------------------------------
Se usa IMPUTACION-NEUTRAL + FLAGS DE COBERTURA (la misma que ya aplica el BLOQUE
2.C con ``df['activos_prev'].fillna(activos)``), NO el camino NaN-nativo. Razones:

  1. Conserva el contrato ``assert X.isnull().sum().sum() == 0`` de la matriz X sin
     listas blancas ni relajaciones, reduciendo el riesgo de regresion.
  2. Mantiene operativo a ``RandomForestClassifier`` de scikit-learn, que NO acepta
     NaN (lanzaria ``ValueError: Input X contains NaN``); asi el super-ensamble de
     4 modelos sigue devolviendo una probabilidad valida en [0, 1] para toda
     empresa, incluida la que no tiene historia.
  3. El requisito de "entrada parcial" del usuario se cumple igual: las BANDERAS de
     cobertura capturan explicitamente la ausencia de historia, de modo que el
     modelo distingue "tendencia estable" de "sin dato" sin necesidad de NaN.

Convencion de imputacion neutral por tipo de feature:
  * Pendientes / variaciones / CAGR / aceleraciones: si falta el saldo historico
    necesario, la feature toma 0.0 (variacion nula => "sin tendencia observable"),
    y el flag de cobertura indica que no hubo dato real. Esto es neutro: no inventa
    deterioro ni mejora.
  * Conteos (p.ej. anios con perdida): cuentan solo sobre los anios realmente
    observados; su magnitud se interpreta junto con ``Anios_Historia_Disponible``.

Anti-fuga (autoritativo)
------------------------
La ventana de historia para el corte t usa EXCLUSIVAMENTE anios en
``[anio_t - 4, anio_t - 1]`` intersectados con los anios realmente presentes en la
serie (2020..2024). ``validar_sin_fuga_multianio`` asevera que ninguna etiqueta de
anio empleada supera a ``anio_t`` y que, para el modelo 2024 -> 2025, jamas se
referencia el anio 2025 (la etiqueta y se deriva de columnas *_2025 que no entran
en X). Ver el assert del propio pipeline 'no 2025 en X'.

Reutilizacion de blindaje numerico
----------------------------------
Se reutilizan ``division_segura`` (proteccion contra division por cero/NaN/inf con
``.clip()`` y ``numpy.nan_to_num``) y ``numpy.log1p`` (escala logarithmica estable
para saldos no negativos) del pipeline activo, citando numpy/pandas en los
comentarios de cada bloque.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Reutilizamos utilidades ya auditadas del pipeline activo. El import es seguro
# porque script_cinco_optimizado protege su ejecucion con el guardian __main__.
from script_cinco_optimizado import division_segura, normalizar_nit

# ------------------------------------------------------------------------------
# PARAMETROS DE LA VENTANA PLURIANUAL
# ------------------------------------------------------------------------------
# Profundidad maxima de historia hacia atras desde el corte t (en anios). Para el
# modelo 2024 -> 2025 (anio_t = 2024) la ventana de historia es 2020..2023, mas el
# propio corte t (2024) como punto final de las pendientes. Nunca se usa > anio_t.
PROFUNDIDAD_HISTORIA = 4

# Alias de saldos disponibles en la serie multi-anual (FEAT-004). Se documentan
# aqui para dejar explicito que estas columnas NUNCA mezclan anios distintos: cada
# DataFrame de la serie corresponde a UN unico anio (columna 'anio').
ALIAS_SALDOS_SERIE = [
    "activos_prev",
    "pasivos_prev",
    "patrimonio_prev",
    "ingresos_prev",
    "utilidad_prev",
    "ebit_prev",
]

# Nombres EXACTOS de las columnas multi-anuales que produce este modulo. Se expone
# como constante para que la ablacion (analisis_multianio.py) pueda quitarlas o
# anadirlas sin re-derivar la lista a mano, y para documentar el contrato.
COLUMNAS_MULTIANIO = [
    "Anios_Historia_Disponible",
    "Flag_Sin_Historia_3A",
    "Flag_Historia_Parcial",
    "Pendiente_Patrimonio_3A_Sobre_Activos",
    "Var_Patrimonio_2A_Sobre_Activos",
    "CAGR_Ingresos_Multianio",
    "Aceleracion_Endeudamiento",
    "Anios_Con_Perdida_Ventana",
    "Persistencia_Perdidas_3A",
    "Flag_Deterioro_Patrimonial_Sostenido",
    "Flag_Paralisis_Sostenida_Multianio",
]


def _saldos_anio(
    series_multianio: Dict[int, pd.DataFrame],
    anio: int,
    nit_key: pd.Series,
) -> Optional[pd.DataFrame]:
    """
    Alinea los saldos de un anio concreto de la serie multi-anual al orden de NITs
    del corte t (``nit_key``), devolviendo un DataFrame con los alias de saldos y un
    booleano ``_presente`` por fila (True si ese NIT tiene dato ese anio).

    Devuelve None si el anio no existe en la serie (p.ej. anios fuera de 2020..2024).

    Notas tecnicas (pandas)
    -----------------------
    * Se usa ``DataFrame.set_index('NIT').reindex(nit_key)`` para alinear por NIT:
      ``reindex`` deja NaN en los NITs sin dato, lo que luego se traduce en el flag
      ``_presente`` = False y en imputacion neutral de la feature correspondiente.
    """
    if anio not in series_multianio:
        return None
    df_anio = series_multianio[anio]
    # Clave de cruce homogenea con el resto del pipeline (quita sufijo .0).
    idx = normalizar_nit(df_anio["NIT"])
    cols = [c for c in ALIAS_SALDOS_SERIE if c in df_anio.columns]
    alineado = (
        df_anio[cols]
        .set_axis(idx, axis=0)
        .reindex(nit_key.values)
    )
    alineado.index = nit_key.index
    # Un NIT se considera "presente" ese anio si tiene al menos activos informados.
    alineado["_presente"] = (~alineado["activos_prev"].isna()).astype(int)
    return alineado


def construir_features_multianio(
    df_maestro_t: pd.DataFrame,
    series_multianio: Dict[int, pd.DataFrame],
    anio_t: int,
    saldos_t: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Construye el bloque de columnas de TENDENCIA PLURIANUAL indexado igual que
    ``df_maestro_t`` (mismo indice de filas), listo para concatenarse a la matriz X.

    Parametros
    ----------
    df_maestro_t : pandas.DataFrame
        Corte t (debe tener columna 'NIT'). Define el universo de empresas y el
        orden/indice de filas de salida.
    series_multianio : dict[int, pandas.DataFrame]
        Serie de saldos por anio (FEAT-004): {anio: DataFrame por NIT}.
    anio_t : int
        Anio del corte t (2024 para el entrenamiento 2024->2025; 2025 para la
        proyeccion del semaforo 2026). La ventana de historia es
        ``[anio_t-4, anio_t-1]`` intersectada con los anios presentes en la serie.
    saldos_t : dict[str, pandas.Series]
        Saldos del propio corte t ya calculados y alineados al indice de
        ``df_maestro_t`` (activos, pasivos, patrimonio, ingresos, utilidad). Se usan
        como punto final de las pendientes para no releer el corte t.

    Devuelve
    --------
    pandas.DataFrame
        Columnas = ``COLUMNAS_MULTIANIO``, indice = ``df_maestro_t.index``, SIN NaN
        (imputacion-neutral + flags). No dropea filas.
    """
    nit_key = normalizar_nit(df_maestro_t["NIT"])
    nit_key.index = df_maestro_t.index
    n = len(df_maestro_t)
    idx = df_maestro_t.index

    # Saldos del corte t (punto final). Se toman tal cual del pipeline (ya clipeados).
    activos_t = saldos_t["activos"].to_numpy(dtype=float)
    pasivos_t = saldos_t["pasivos"].to_numpy(dtype=float)
    patrimonio_t = saldos_t["patrimonio"].to_numpy(dtype=float)
    ingresos_t = saldos_t["ingresos"].to_numpy(dtype=float)
    utilidad_t = saldos_t["utilidad"].to_numpy(dtype=float)

    # Anios de historia estrictos (< anio_t) presentes en la serie, mas antiguos
    # primero. Para anio_t=2024 => [2020, 2021, 2022, 2023] intersectado con la serie.
    anios_historia: List[int] = sorted(
        a for a in series_multianio
        if (anio_t - PROFUNDIDAD_HISTORIA) <= a <= (anio_t - 1)
    )

    # Mapa anio -> saldos alineados (o None si no estuviera). Guardamos tambien el
    # vector _presente por anio para construir las banderas de cobertura.
    saldos_hist: Dict[int, pd.DataFrame] = {}
    for a in anios_historia:
        al = _saldos_anio(series_multianio, a, nit_key)
        if al is not None:
            saldos_hist[a] = al
    anios_obs = sorted(saldos_hist.keys())

    X = pd.DataFrame(index=idx)

    # --------------------------------------------------------------------------
    # 1) COBERTURA DE HISTORIA (banderas explicitas de cuanta historia hay)
    # --------------------------------------------------------------------------
    # Anios_Historia_Disponible: conteo de anios de la ventana con dato real para el
    # NIT. numpy suma los indicadores _presente por anio (0/1) a lo largo del eje.
    if anios_obs:
        presencia = np.column_stack([saldos_hist[a]["_presente"].to_numpy() for a in anios_obs])
        anios_disp = presencia.sum(axis=1).astype(int)
    else:
        anios_disp = np.zeros(n, dtype=int)
    X["Anios_Historia_Disponible"] = anios_disp

    # Flag_Sin_Historia_3A: no hay NINGUN anio de historia plurianual (0 observados).
    X["Flag_Sin_Historia_3A"] = (anios_disp == 0).astype(int)
    # Flag_Historia_Parcial: tiene algo de historia pero menos que la ventana plena.
    X["Flag_Historia_Parcial"] = ((anios_disp > 0) & (anios_disp < len(anios_historia))).astype(int)

    # --------------------------------------------------------------------------
    # 2) PENDIENTE / VARIACION DE PATRIMONIO (deterioro patrimonial sostenido)
    # --------------------------------------------------------------------------
    # Normalizamos siempre por los activos del corte t para obtener una magnitud
    # comparable entre empresas de distinto tamano (ratio adimensional acotado).
    # division_segura blinda contra activos_t == 0 (numpy: where + nan_to_num + clip).
    anio_3a = anio_t - 3  # p.ej. 2021 para anio_t=2024 (pendiente a ~3 anios)
    anio_2a = anio_t - 2  # p.ej. 2022 (variacion a ~2 anios)

    def patrimonio_de(anio: int) -> Optional[np.ndarray]:
        if anio in saldos_hist:
            return saldos_hist[anio]["patrimonio_prev"].to_numpy(dtype=float)
        return None

    pat_3a = patrimonio_de(anio_3a)
    pat_2a = patrimonio_de(anio_2a)

    # Pendiente patrimonial a 3 anios: (patrimonio_t - patrimonio_{t-3}) / activos_t.
    # Si falta el saldo historico, variacion neutra 0.0 (y el flag de cobertura avisa).
    if pat_3a is not None:
        delta_pat_3a = np.where(np.isnan(pat_3a), 0.0, patrimonio_t - np.nan_to_num(pat_3a, nan=patrimonio_t))
        X["Pendiente_Patrimonio_3A_Sobre_Activos"] = division_segura(
            delta_pat_3a, activos_t, 0.0, -2.0, 2.0
        )
    else:
        X["Pendiente_Patrimonio_3A_Sobre_Activos"] = 0.0

    if pat_2a is not None:
        delta_pat_2a = np.where(np.isnan(pat_2a), 0.0, patrimonio_t - np.nan_to_num(pat_2a, nan=patrimonio_t))
        X["Var_Patrimonio_2A_Sobre_Activos"] = division_segura(
            delta_pat_2a, activos_t, 0.0, -2.0, 2.0
        )
    else:
        X["Var_Patrimonio_2A_Sobre_Activos"] = 0.0

    # --------------------------------------------------------------------------
    # 3) CAGR DE INGRESOS MULTI-ANIO (tasa de crecimiento anual compuesta)
    # --------------------------------------------------------------------------
    # CAGR = (V_final / V_inicial)^(1/num_anios) - 1. Usamos el anio observado mas
    # antiguo como V_inicial y los ingresos del corte t como V_final. np.log1p y la
    # exponencial se evitan: calculamos la potencia directa con proteccion de signo.
    if anios_obs:
        anio_inicial = anios_obs[0]
        ingresos_ini = saldos_hist[anio_inicial]["ingresos_prev"].to_numpy(dtype=float)
        num_anios = max(1, anio_t - anio_inicial)
        # Solo es valida si hay ingresos iniciales > 0 y el NIT estaba presente.
        base_valida = (~np.isnan(ingresos_ini)) & (ingresos_ini > 0)
        ratio = np.where(base_valida, ingresos_t / np.where(base_valida, ingresos_ini, 1.0), 1.0)
        # ratio se acota a [0, 100] para que la raiz no explote con outliers.
        ratio = np.clip(np.nan_to_num(ratio, nan=1.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
        cagr = np.power(ratio, 1.0 / num_anios) - 1.0
        # Donde no hay base valida, CAGR neutro 0.0 (sin crecimiento observable).
        cagr = np.where(base_valida, cagr, 0.0)
        X["CAGR_Ingresos_Multianio"] = np.clip(np.nan_to_num(cagr, nan=0.0), -1.0, 3.0)
    else:
        X["CAGR_Ingresos_Multianio"] = 0.0

    # --------------------------------------------------------------------------
    # 4) ACELERACION DEL ENDEUDAMIENTO (delta de delta: segunda diferencia)
    # --------------------------------------------------------------------------
    # Endeudamiento por anio = pasivos / activos (acotado a [0, 2] como en el
    # pipeline). La aceleracion es la diferencia de dos variaciones consecutivas:
    #   accel = (end_t - end_{t-1}) - (end_{t-1} - end_{t-2})
    # Capta si el endeudamiento NO solo sube, sino que sube cada vez mas rapido.
    end_t = division_segura(pasivos_t, activos_t, 0.0, 0.0, 2.0)

    def endeudamiento_de(anio: int) -> Optional[np.ndarray]:
        if anio in saldos_hist:
            pa = saldos_hist[anio]["pasivos_prev"].to_numpy(dtype=float)
            ac = saldos_hist[anio]["activos_prev"].to_numpy(dtype=float)
            end = division_segura(np.nan_to_num(pa, nan=0.0), np.nan_to_num(ac, nan=0.0), np.nan, 0.0, 2.0)
            return end  # NaN donde activos era NaN/0 -> se trata como faltante abajo
        return None

    end_t1 = endeudamiento_de(anio_t - 1)
    end_t2 = endeudamiento_de(anio_t - 2)
    if end_t1 is not None and end_t2 is not None:
        # Imputacion neutral: donde falte un tramo, la variacion de ese tramo es 0.
        delta_1 = end_t - np.where(np.isnan(end_t1), end_t, end_t1)
        delta_2 = np.where(np.isnan(end_t1), 0.0, end_t1) - np.where(np.isnan(end_t2), np.where(np.isnan(end_t1), 0.0, end_t1), end_t2)
        accel = np.clip(np.nan_to_num(delta_1 - delta_2, nan=0.0), -2.0, 2.0)
        X["Aceleracion_Endeudamiento"] = accel
    else:
        X["Aceleracion_Endeudamiento"] = 0.0

    # --------------------------------------------------------------------------
    # 5) PERSISTENCIA DE PERDIDAS EN LA VENTANA (conteo de anios con utilidad<0)
    # --------------------------------------------------------------------------
    # Contamos cuantos anios de la VENTANA OBSERVADA tuvieron utilidad negativa.
    # Solo se cuentan los anios realmente presentes (imputacion por conteo: la
    # ausencia NO suma perdida). Incluimos el corte t como un anio mas de la cuenta.
    conteo_perdida = (utilidad_t < 0).astype(int)
    anios_contados = np.ones(n, dtype=int)  # el corte t siempre cuenta
    for a in anios_obs:
        util_a = saldos_hist[a]["utilidad_prev"].to_numpy(dtype=float)
        presente_a = saldos_hist[a]["_presente"].to_numpy()
        conteo_perdida = conteo_perdida + np.where(
            (presente_a == 1) & (~np.isnan(util_a)) & (util_a < 0), 1, 0
        )
        anios_contados = anios_contados + presente_a
    X["Anios_Con_Perdida_Ventana"] = conteo_perdida.astype(int)
    # Persistencia relativa: fraccion de anios observados con perdida (0..1).
    X["Persistencia_Perdidas_3A"] = np.clip(
        division_segura(conteo_perdida.astype(float), anios_contados.astype(float), 0.0, 0.0, 1.0),
        0.0,
        1.0,
    )

    # --------------------------------------------------------------------------
    # 6) BANDERAS DE DETERIORO SOSTENIDO (senales binarias de paralisis)
    # --------------------------------------------------------------------------
    # Deterioro patrimonial sostenido: patrimonio del corte t estrictamente menor
    # que el del anio inicial observado (cae en la ventana) Y hay >=2 anios de
    # historia (para que "sostenido" tenga sentido). Si no hay historia, 0.
    if anios_obs:
        pat_ini = saldos_hist[anios_obs[0]]["patrimonio_prev"].to_numpy(dtype=float)
        deterioro = (
            (~np.isnan(pat_ini))
            & (patrimonio_t < pat_ini)
            & (anios_disp >= 2)
        )
        X["Flag_Deterioro_Patrimonial_Sostenido"] = deterioro.astype(int)
    else:
        X["Flag_Deterioro_Patrimonial_Sostenido"] = 0

    # Paralisis sostenida: ingresos nulos en el corte t Y en el anio previo observado
    # (actividad comercial detenida de forma sostenida, senal fuerte Ley 1116).
    if (anio_t - 1) in saldos_hist:
        ing_t1 = saldos_hist[anio_t - 1]["ingresos_prev"].to_numpy(dtype=float)
        paralisis = (ingresos_t == 0) & (~np.isnan(ing_t1)) & (ing_t1 == 0)
        X["Flag_Paralisis_Sostenida_Multianio"] = paralisis.astype(int)
    else:
        X["Flag_Paralisis_Sostenida_Multianio"] = 0

    # --------------------------------------------------------------------------
    # BLINDAJE FINAL: imputacion-neutral garantiza CERO NaN en el bloque.
    # --------------------------------------------------------------------------
    # Forzamos float en las columnas continuas y entero en conteos/flags, y
    # aseveramos que no quedo ningun NaN (contrato con el assert de la matriz X).
    for col in COLUMNAS_MULTIANIO:
        assert col in X.columns, f"Falta la columna multi-anio esperada: {col}"
    assert list(X.columns) == COLUMNAS_MULTIANIO, (
        "El orden/conjunto de columnas multi-anio no coincide con COLUMNAS_MULTIANIO."
    )
    assert X.isnull().sum().sum() == 0, (
        f"NaN detectado en el bloque multi-anio: {X.columns[X.isnull().any()].tolist()}"
    )
    # Anti-fuga: ninguna etiqueta de anio usada supera anio_t.
    validar_sin_fuga_multianio(anios_historia + anios_obs, anio_t)
    return X


def validar_sin_fuga_multianio(anios_usados: List[int], anio_t: int) -> None:
    """
    Asevera la garantia anti-fuga de la ventana plurianual: ningun anio empleado en
    la construccion de las features puede superar ``anio_t``. Para el modelo
    2024 -> 2025 esto significa, en particular, que jamas se referencia el anio 2025
    (la etiqueta y se deriva de columnas *_2025 que NO entran en X).

    Lanza AssertionError si algun anio > anio_t aparece en ``anios_usados``.
    """
    anios_fuga = sorted({a for a in anios_usados if a > anio_t})
    assert not anios_fuga, (
        f"ALERTA ANTI-FUGA MULTI-ANIO: se referencian anios > {anio_t}: {anios_fuga}. "
        "La ventana plurianual solo puede usar datos <= anio_t."
    )
    # Refuerzo explicito para el modelo 2024->2025: nunca el anio 2025 cuando anio_t<=2024.
    if anio_t <= 2024:
        assert 2025 not in anios_usados, (
            "ALERTA ANTI-FUGA: el anio 2025 no puede entrar en las features del modelo 2024->2025."
        )
