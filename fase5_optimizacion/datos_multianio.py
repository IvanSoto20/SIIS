# -*- coding: utf-8 -*-
"""
CAPA DE DATOS MULTI-ANUAL (FEAT-004): SERIE DE SALDOS 2020..2024
==============================================================================
Objetivo
--------
Reconstruir, por NIT, los saldos contables (Balance 210030 + Resultados 310030)
de los anos 2020 a 2024 a partir de los anexos crudos de SIIS, aprovechando que
cada carpeta ``SIIS/pymes_individuales-YYYY`` contiene DOS periodos reportados:

  * 'periodo actual'   -> saldos del propio ano YYYY (reporte primario).
  * 'periodo anterior' -> saldos del ano YYYY-1 (reporte comparativo).

Mapa de observabilidad (cada ano se puede ver desde una o dos carpetas):

  pymes_individuales-2021 -> 2021 (actual) + 2020 (anterior)
  pymes_individuales-2022 -> 2022 (actual) + 2021 (anterior)
  pymes_individuales-2023 -> 2023 (actual) + 2022 (anterior)
  pymes_individuales-2024 -> 2024 (actual) + 2023 (anterior)

Asi, los anos 2021, 2022 y 2023 son observables desde DOS fuentes distintas, lo
que permite medir solapamiento de NITs y discrepancia de saldos entre fuentes.

Esta feature SOLO construye y verifica la capa de datos. NO toca la ingenieria de
features del modelo, ni el entrenamiento, ni el semaforo (eso es FEAT-005/006).

Reutilizacion de codigo
-----------------------
Se importan y REUTILIZAN las utilidades forenses ya validadas en
``script_cinco_optimizado.py`` (normalizar_texto NFKD, normalizar_nit,
buscar_archivo_siis, MAPA_BALANCE_PREV, MAPA_RESULTADOS_PREV, SIIS_DIR,
SALIDAS_DIR). La lectura por periodo generaliza ``leer_periodo_anterior()``:
la funcion nueva ``leer_periodo()`` parametriza el filtro para aceptar tanto
'periodo actual' como 'periodo anterior'.

Prohibiciones (autoritativas del usuario)
-----------------------------------------
  * py/ y SIIS/ son SOLO LECTURA. Aqui solo se LEEN los .xlsx de SIIS.
  * Todo el codigo y los artefactos de salida viven en fase5_optimizacion/.
  * No introducir fuga de datos: no se mezclan saldos de anos distintos; cada
    ano queda identificado explicitamente por su columna 'anio'.
"""

import json

import numpy as np
import pandas as pd

# Reutilizamos las utilidades ya auditadas del pipeline activo. El import es
# seguro porque ``script_cinco_optimizado.py`` protege su ejecucion con el
# guardian ``if __name__ == "__main__"``, de modo que importarlo NO dispara el
# entrenamiento ni la escritura de reportes; solo expone funciones y constantes.
from script_cinco_optimizado import (
    MAPA_BALANCE_PREV,
    MAPA_RESULTADOS_PREV,
    SALIDAS_DIR,
    SIIS_DIR,
    buscar_archivo_siis,
    normalizar_nit,
    normalizar_texto,
)

# ------------------------------------------------------------------------------
# PARAMETROS DE OBSERVABILIDAD Y TOLERANCIAS
# ------------------------------------------------------------------------------
# Mapa carpeta -> (ano_periodo_actual, ano_periodo_anterior). Se deriva del
# nombre de la carpeta: 'actual' es el propio ano YYYY y 'anterior' es YYYY-1.
CARPETAS_PYMES = {
    "pymes_individuales-2021": {"periodo actual": 2021, "periodo anterior": 2020},
    "pymes_individuales-2022": {"periodo actual": 2022, "periodo anterior": 2021},
    "pymes_individuales-2023": {"periodo actual": 2023, "periodo anterior": 2022},
    "pymes_individuales-2024": {"periodo actual": 2024, "periodo anterior": 2023},
}

# Conteos de NITs unicos en 'periodo anterior' de 210030 verificados por el
# orquestador. Se usan solo como banda de cordura (+/- 5%); NO se asumen cifras
# exactas porque el usuario advirtio de posible drift entre anos.
CONTEOS_PERIODO_ANTERIOR_VERIFICADOS = {
    2020: 23175,  # pymes-2021 -> 'periodo anterior' (ano 2020)
    2021: 24757,  # pymes-2022 -> 'periodo anterior' (ano 2021)
    2022: 24760,  # pymes-2023 -> 'periodo anterior' (ano 2022)
}
TOLERANCIA_CONTEO = 0.05  # +/- 5% respecto a los valores verificados

# Nombre del artefacto de cache (patron identico a obtener_saldos_periodo_anterior).
NOMBRE_CACHE_SERIES = "series_saldos_multianio.pkl"
NOMBRE_DIAGNOSTICO = "diagnostico_multianio.json"

# Umbral relativo para considerar que dos fuentes discrepan en el saldo de activos
# del mismo ano: |a - b| / max(|a|, |b|) > 0.01 (1%).
UMBRAL_DISCREPANCIA_ACTIVOS = 0.01


# ==============================================================================
# LECTURA GENERALIZADA POR PERIODO (generaliza leer_periodo_anterior)
# ==============================================================================
def leer_periodo(ruta_excel, mapa_columnas, periodo_objetivo):
    """
    Lee un anexo de SIIS quedandose solo con las columnas necesarias y con las
    filas del periodo solicitado ('periodo actual' o 'periodo anterior'),
    validando unicidad por NIT.

    Generaliza ``leer_periodo_anterior()`` parametrizando el filtro de periodo.

    Parametros
    ----------
    ruta_excel : pathlib.Path
        Ruta al anexo .xlsx (p.ej. 210030_*.xlsx o 310030_*.xlsx).
    mapa_columnas : dict
        Mapa {nombre_cuenta_original: alias_destino} de las cuentas a extraer
        (p.ej. MAPA_BALANCE_PREV o MAPA_RESULTADOS_PREV).
    periodo_objetivo : str
        'periodo actual' o 'periodo anterior'. Se compara tras normalizar con
        normalizar_texto (NFKD) para ser inmune a tildes/mayusculas/espacios.

    Devuelve
    --------
    pandas.DataFrame
        Columnas: 'NIT' + los alias de ``mapa_columnas``. Una fila por NIT.

    Notas tecnicas (pandas/numpy)
    -----------------------------
    * pandas.read_excel con ``usecols`` CALLABLE: el callable recibe el nombre de
      cada columna y devuelve True/False; asi solo se materializan las columnas
      necesarias, lo que acelera mucho la lectura de estos .xlsx grandes
      (ref. pandas.read_excel, parametro usecols admite funcion).
    * pandas.to_datetime(errors='coerce'): convierte fechas invalidas a NaT en
      lugar de lanzar excepcion, de modo que el ordenamiento por fecha es robusto.
    * pandas.to_numeric(errors='coerce'): convierte saldos no numericos a NaN sin
      romper el pipeline.
    * drop_duplicates(keep='last') tras sort_values por Fecha_Corte: conserva el
      reporte mas reciente por NIT (deduplicacion forense).
    """
    periodo_norm = normalizar_texto(periodo_objetivo)
    assert periodo_norm in {"periodo actual", "periodo anterior"}, (
        f"periodo_objetivo invalido: {periodo_objetivo!r}. "
        "Debe ser 'periodo actual' o 'periodo anterior'."
    )

    columnas_objetivo = {normalizar_texto(k): v for k, v in mapa_columnas.items()}
    columnas_objetivo.update({
        "nit": "NIT",
        "periodo": "Periodo",
        "fecha de corte": "Fecha_Corte",
    })

    # Lectura optimizada: usecols callable sobre el nombre normalizado de columna.
    df = pd.read_excel(ruta_excel, usecols=lambda c: normalizar_texto(c) in columnas_objetivo)
    df = df.rename(columns=lambda c: columnas_objetivo[normalizar_texto(c)])

    columnas_faltantes = set(columnas_objetivo.values()) - set(df.columns)
    assert not columnas_faltantes, (
        f"Faltan columnas obligatorias en {ruta_excel.name}: {columnas_faltantes}"
    )

    df["NIT"] = normalizar_nit(df["NIT"])

    # Filtro por periodo parametrizado (inmune a tildes gracias a NFKD).
    df = df[df["Periodo"].astype(str).map(normalizar_texto) == periodo_norm].copy()
    df["Fecha_Corte"] = pd.to_datetime(df["Fecha_Corte"], errors="coerce")

    for col_alias in mapa_columnas.values():
        df[col_alias] = pd.to_numeric(df[col_alias], errors="coerce")

    # Deduplicacion forense: conservar la ultima fecha de corte reportada por NIT.
    df = (
        df.sort_values(["NIT", "Fecha_Corte"])
        .drop_duplicates(subset=["NIT"], keep="last")
        .drop(columns=["Periodo", "Fecha_Corte"])
    )
    assert df["NIT"].is_unique, f"NIT duplicado detectado en {ruta_excel.name}"
    return df


# ==============================================================================
# EXTRACCION DE SALDOS DE UN (carpeta, periodo) CONCRETO
# ==============================================================================
def _extraer_saldos_carpeta_periodo(carpeta, periodo_objetivo):
    """
    Lee Balance (210030) + Resultados (310030) de una carpeta pymes para un
    periodo dado, los mergea por NIT y aplica la correccion forense de escala.

    Parametros
    ----------
    carpeta : pathlib.Path
        Carpeta ``SIIS/pymes_individuales-YYYY``.
    periodo_objetivo : str
        'periodo actual' o 'periodo anterior'.

    Devuelve
    --------
    pandas.DataFrame
        Una fila por NIT con los alias de MAPA_BALANCE_PREV + MAPA_RESULTADOS_PREV.

    Correccion forense de escala
    ----------------------------
    Identica a la de ``obtener_saldos_periodo_anterior``: si un pasivo supera a los
    activos por un factor > 50 (con activos > 0), se interpreta como un error de
    digitacion de tres ceros de mas y se divide el pasivo entre 1000.
    """
    balance = leer_periodo(buscar_archivo_siis(carpeta, "210030"), MAPA_BALANCE_PREV, periodo_objetivo)
    resultados = leer_periodo(buscar_archivo_siis(carpeta, "310030"), MAPA_RESULTADOS_PREV, periodo_objetivo)

    # Merge Balance + Resultados por NIT. validate='one_to_one' garantiza que no
    # haya duplicados ocultos en ninguna de las dos fuentes (si los hubiera, pandas
    # lanzaria MergeError, exponiendo el problema en lugar de corromperlo en silencio).
    df = balance.merge(resultados, on="NIT", how="outer", validate="one_to_one")

    # Correccion forense de escala de pasivos (tres ceros de mas por digitacion).
    descuadre_extremo = (df["pasivos_prev"] > df["activos_prev"] * 50) & (df["activos_prev"] > 0)
    df.loc[descuadre_extremo, "pasivos_prev"] /= 1000.0

    return df


# ==============================================================================
# CONSTRUCCION DE LA SERIE MULTI-ANUAL 2020..2024
# ==============================================================================
def construir_series_saldos_multianio(usar_cache=True):
    """
    Construye el diccionario {anio: DataFrame_por_NIT} con los saldos de los anos
    2020..2024, leyendo ambos periodos de las 4 carpetas pymes_individuales.

    Para cada ano observable desde DOS carpetas (2021, 2022, 2023), se PREFIERE la
    fuente 'periodo actual' de la carpeta del propio ano, por ser el reporte
    primario; la fuente 'periodo anterior' de la carpeta siguiente se usa solo para
    diagnostico de solapamiento/discrepancia (ver diagnostico_multianio.json), no
    para sustituir saldos. Los NITs presentes solo en la fuente comparativa se
    agregan como complemento (left-anti join), de modo que no se pierde cobertura.

    Parametros
    ----------
    usar_cache : bool
        Si True y existe salidas/series_saldos_multianio.pkl, se carga de cache y
        se evita releer los .xlsx (lectura lenta). Patron identico al de
        obtener_saldos_periodo_anterior.

    Devuelve
    --------
    dict[int, pandas.DataFrame]
        Claves = anos (2020, 2021, 2022, 2023, 2024). Valores = DataFrame con una
        fila por NIT y columnas: 'NIT', 'anio', 'fuente' + alias de saldos.
    """
    ruta_cache = SALIDAS_DIR / NOMBRE_CACHE_SERIES
    if usar_cache and ruta_cache.exists():
        print(f"Cargando series multi-anuales desde cache: {ruta_cache.name}")
        return pd.read_pickle(ruta_cache)

    # 1) Extraccion cruda: para cada (carpeta, periodo) obtenemos un DataFrame y lo
    #    etiquetamos con el ano que observa y la fuente de procedencia.
    extracciones = {}  # (anio, fuente_str) -> DataFrame
    for nombre_carpeta, mapa_periodos in CARPETAS_PYMES.items():
        carpeta = SIIS_DIR / nombre_carpeta
        assert carpeta.is_dir(), f"No existe la carpeta esperada: {carpeta}"
        for periodo_objetivo, anio in mapa_periodos.items():
            print(f"Extrayendo {nombre_carpeta} :: {periodo_objetivo} -> ano {anio}...")
            df = _extraer_saldos_carpeta_periodo(carpeta, periodo_objetivo)
            fuente = f"{nombre_carpeta}/{periodo_objetivo}"
            extracciones[(anio, periodo_objetivo, fuente)] = df

    # 2) Consolidacion por ano, prefiriendo 'periodo actual' como fuente primaria.
    series = {}
    for anio in sorted({clave[0] for clave in extracciones}):
        # Fuentes que observan este ano, ordenadas con 'periodo actual' primero.
        fuentes_anio = [k for k in extracciones if k[0] == anio]
        fuentes_anio.sort(key=lambda k: 0 if k[1] == "periodo actual" else 1)

        df_primario = None
        for (_, periodo_objetivo, fuente) in fuentes_anio:
            df_fuente = extracciones[(anio, periodo_objetivo, fuente)].copy()
            df_fuente["fuente"] = fuente
            if df_primario is None:
                df_primario = df_fuente
            else:
                # Complemento (left-anti): solo los NITs ausentes en el primario.
                nuevos = df_fuente[~df_fuente["NIT"].isin(df_primario["NIT"])]
                df_primario = pd.concat([df_primario, nuevos], ignore_index=True)

        df_primario.insert(1, "anio", anio)
        assert df_primario["NIT"].is_unique, f"NIT duplicado al consolidar el ano {anio}."
        series[anio] = df_primario

    # 3) Aserciones de integridad por ano sobre la fuente 'periodo anterior' de
    #    210030 (banda +/- 5% respecto a los conteos verificados por el orquestador).
    for anio, conteo_esperado in CONTEOS_PERIODO_ANTERIOR_VERIFICADOS.items():
        clave = next(k for k in extracciones if k[0] == anio and k[1] == "periodo anterior")
        n_real = extracciones[clave]["NIT"].nunique()
        lim_inf = conteo_esperado * (1 - TOLERANCIA_CONTEO)
        lim_sup = conteo_esperado * (1 + TOLERANCIA_CONTEO)
        assert lim_inf <= n_real <= lim_sup, (
            f"Conteo de NITs 'periodo anterior' del ano {anio} fuera de banda: "
            f"{n_real} no esta en [{lim_inf:.0f}, {lim_sup:.0f}] "
            f"(verificado ~{conteo_esperado})."
        )

    # 4) Persistencia en cache (patron identico a obtener_saldos_periodo_anterior).
    pd.to_pickle(series, ruta_cache)
    print(f"Series multi-anuales persistidas en cache: {ruta_cache.name}")
    return series


# ==============================================================================
# DIAGNOSTICO DE SOLAPAMIENTO Y DISCREPANCIA ENTRE FUENTES
# ==============================================================================
def _discrepancia_activos_entre_fuentes(df_a, df_b):
    """
    Mide, sobre los NITs comunes a dos fuentes del mismo ano, la fraccion de NITs
    cuyo saldo de activos discrepa en mas de UMBRAL_DISCREPANCIA_ACTIVOS (1%)
    relativo al maximo absoluto de ambos.

    Formula por NIT: |a - b| / max(|a|, |b|) > 0.01 (si max == 0, se considera 0
    discrepancia, pues ambos saldos son nulos).

    Devuelve
    --------
    dict con nits_comunes, nits_discrepantes y pct_discrepancia.
    """
    comun = df_a.merge(
        df_b[["NIT", "activos_prev"]].rename(columns={"activos_prev": "activos_b"}),
        on="NIT",
        how="inner",
        validate="one_to_one",
    )
    a = comun["activos_prev"].to_numpy(dtype=float)
    b = comun["activos_b"].to_numpy(dtype=float)
    # np.nan_to_num evita que NaN contamine la comparacion (se tratan como 0).
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    maximo = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(maximo > 0, np.abs(a - b) / maximo, 0.0)
    n_comunes = int(len(comun))
    n_discrepantes = int((rel > UMBRAL_DISCREPANCIA_ACTIVOS).sum())
    pct = (n_discrepantes / n_comunes) if n_comunes > 0 else 0.0
    return {
        "nits_comunes": n_comunes,
        "nits_discrepantes": n_discrepantes,
        "pct_discrepancia": round(float(pct), 6),
    }


def generar_diagnostico_multianio(series=None):
    """
    Construye el diccionario de diagnostico multi-anual y lo persiste en
    salidas/diagnostico_multianio.json.

    Contenido
    ---------
    * nits_por_anio: nº de NITs consolidados por ano (2020..2024).
    * solapamientos_multianio: 2021∩2022, 2022∩2023, 2021∩2023 y triple.
    * discrepancia_entre_fuentes: para cada ano observable desde dos carpetas
      (2021, 2022, 2023), solapamiento de NITs y % de discrepancia de activos
      entre 'periodo actual' (carpeta propia) y 'periodo anterior' (carpeta
      siguiente).
    * decision_fuente: documentacion de la regla de preferencia de fuente.

    Parametros
    ----------
    series : dict[int, DataFrame] | None
        Si None, se construye (o se carga de cache) con
        construir_series_saldos_multianio().

    Devuelve
    --------
    dict con el diagnostico completo (ya serializado a JSON en disco).
    """
    if series is None:
        series = construir_series_saldos_multianio()

    # 1) Conteo de NITs por ano consolidado.
    nits_por_anio = {int(anio): int(df["NIT"].nunique()) for anio, df in series.items()}

    # 2) Solapamientos multi-anuales sobre los anos consolidados.
    def conjunto(anio):
        return set(series[anio]["NIT"]) if anio in series else set()

    s2021, s2022, s2023 = conjunto(2021), conjunto(2022), conjunto(2023)
    solapamientos = {
        "2021_inter_2022": len(s2021 & s2022),
        "2022_inter_2023": len(s2022 & s2023),
        "2021_inter_2023": len(s2021 & s2023),
        "triple_2021_2022_2023": len(s2021 & s2022 & s2023),
    }

    # 3) Discrepancia entre fuentes que observan el mismo ano. Para ello releemos
    #    las dos fuentes crudas de cada ano doblemente observable. Si existe cache
    #    de series, estas relecturas son baratas comparadas con el total y solo
    #    afectan a los 3 anos con doble fuente.
    discrepancia = {}
    # Mapa: ano -> (carpeta_actual, carpeta_anterior)
    pares_doble_fuente = {
        2021: ("pymes_individuales-2021", "pymes_individuales-2022"),
        2022: ("pymes_individuales-2022", "pymes_individuales-2023"),
        2023: ("pymes_individuales-2023", "pymes_individuales-2024"),
    }
    for anio, (carpeta_actual, carpeta_anterior) in pares_doble_fuente.items():
        df_actual = _extraer_saldos_carpeta_periodo(SIIS_DIR / carpeta_actual, "periodo actual")
        df_anterior = _extraer_saldos_carpeta_periodo(SIIS_DIR / carpeta_anterior, "periodo anterior")
        detalle = _discrepancia_activos_entre_fuentes(df_actual, df_anterior)
        detalle["fuente_actual"] = f"{carpeta_actual}/periodo actual"
        detalle["fuente_anterior"] = f"{carpeta_anterior}/periodo anterior"
        detalle["nits_fuente_actual"] = int(df_actual["NIT"].nunique())
        detalle["nits_fuente_anterior"] = int(df_anterior["NIT"].nunique())
        discrepancia[str(anio)] = detalle

    diagnostico = {
        "anios_cubiertos": sorted(nits_por_anio.keys()),
        "nits_por_anio": nits_por_anio,
        "solapamientos_multianio": solapamientos,
        "discrepancia_entre_fuentes": discrepancia,
        "umbral_discrepancia_activos": UMBRAL_DISCREPANCIA_ACTIVOS,
        "decision_fuente": (
            "Ante discrepancia entre fuentes que observan el mismo ano, se PREFIERE "
            "la fuente 'periodo actual' de la carpeta del propio ano por ser el "
            "reporte primario; la fuente 'periodo anterior' de la carpeta siguiente "
            "solo aporta NITs ausentes (complemento) y sirve de control cruzado."
        ),
        "conteos_verificados_periodo_anterior": CONTEOS_PERIODO_ANTERIOR_VERIFICADOS,
        "tolerancia_conteo": TOLERANCIA_CONTEO,
    }

    ruta_json = SALIDAS_DIR / NOMBRE_DIAGNOSTICO
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(diagnostico, f, indent=2, ensure_ascii=False)
    print(f"Diagnostico multi-anual escrito en: {ruta_json}")
    return diagnostico


# ==============================================================================
# BLOQUE DE VERIFICACION EJECUTABLE
# ==============================================================================
if __name__ == "__main__":
    print("Construyendo serie de saldos multi-anual 2020..2024 (FEAT-004)...")
    series = construir_series_saldos_multianio()

    print("\nNITs por ano consolidado:")
    for anio in sorted(series):
        print(f"  {anio}: {series[anio]['NIT'].nunique():>7,} NITs")

    diag = generar_diagnostico_multianio(series)

    print("\nSolapamientos multi-anuales:")
    for clave, valor in diag["solapamientos_multianio"].items():
        print(f"  {clave}: {valor:,}")

    print("\nDiscrepancia entre fuentes (mismo ano, dos carpetas):")
    for anio, det in diag["discrepancia_entre_fuentes"].items():
        print(
            f"  {anio}: comunes={det['nits_comunes']:,} "
            f"discrepantes={det['nits_discrepantes']:,} "
            f"({det['pct_discrepancia']:.4%})"
        )

    print("\nProceso FEAT-004 completado con exito.")
