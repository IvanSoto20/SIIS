# -*- coding: utf-8 -*-
"""
FASE 5 / FEAT-002: CONTROL DE DERIVA ENTRE LOS MAESTROS 2021-2024
==============================================================================
Antes de apilar transiciones de distintos anios hay que comprobar que los maestros
reconstruidos (2021, 2022, 2023) son comparables con el de 2024. Si una columna
cambiara de escala o de definicion entre anios, el modelo la confundiria con senal.

Metodo: estadistico de Kolmogorov-Smirnov de dos muestras por columna
(``scipy.stats.ks_2samp``): maxima distancia entre las dos funciones de distribucion
empirica, en [0, 1]. Se aplica a las columnas numericas continuas (mas de 2 valores
distintos), ya que en binarias el KS solo compara proporciones.

Lectura: el valor critico del KS al 5% con n y m observaciones es aproximadamente
1.36 * sqrt((n + m) / (n * m)); con ~23 mil empresas por anio ronda 0.013. Por eso una
diferencia de 0.02 es estadisticamente detectable pero pequena en magnitud. La
referencia practica es la deriva natural entre anios consecutivos (2022 vs 2023).

Solo lectura sobre los maestros; escribe salidas/control_deriva_maestros.json.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import script_cinco_optimizado as base

ANIOS = (2021, 2022, 2023)
UMBRAL_ALTO = 0.15


def _ks_por_columna(a: pd.DataFrame, b: pd.DataFrame, columnas: list) -> pd.Series:
    return pd.Series({c: ks_2samp(a[c].to_numpy(float), b[c].to_numpy(float)).statistic for c in columnas})


def main():
    ref = pd.read_pickle(base.BASE_DIR / "df_maestro_con_ratios_2024.pkl")
    maestros = {a: pd.read_pickle(base.SALIDAS_DIR / f"maestro_ratios_{a}.pkl") for a in ANIOS}
    continuas = [c for c in ref.select_dtypes("number").columns if ref[c].nunique() > 2]
    assert len(continuas) > 30, "Se esperaban decenas de columnas continuas."
    for a, m in maestros.items():
        assert set(continuas) <= set(m.columns), f"El maestro {a} no tiene todas las columnas de 2024."

    res = {"n_columnas_continuas": len(continuas), "umbral_ks_alto": UMBRAL_ALTO, "vs_2024": {}}
    for a, m in maestros.items():
        ks = _ks_por_columna(m, ref, continuas).sort_values(ascending=False)
        n, k = len(m), len(ref)
        res["vs_2024"][str(a)] = {
            "ks_medio": round(float(ks.mean()), 4),
            "ks_max": round(float(ks.iloc[0]), 4), "ks_max_columna": ks.index[0],
            "columnas_ks_mayor_umbral": [c for c in ks.index if ks[c] > UMBRAL_ALTO],
            "valor_critico_5pct": round(float(1.36 * np.sqrt((n + k) / (n * k))), 4),
        }
    ks_ref = _ks_por_columna(maestros[2023], maestros[2022], continuas)
    res["deriva_natural_2022_vs_2023"] = {"ks_medio": round(float(ks_ref.mean()), 4)}
    res["medianas_clave"] = {
        c: {str(a): round(float(m[c].median()), 3) for a, m in {**maestros, 2024: ref}.items()}
        for c in ("Edad_Empresa_Anios", "Nivel_Endeudamiento", "Razon_Corriente", "ROA_Neto") if c in ref.columns}

    for a, d in res["vs_2024"].items():
        print(f"{a} vs 2024 | KS medio {d['ks_medio']:.3f} | max {d['ks_max']:.3f} ({d['ks_max_columna']}) | "
              f"columnas > {UMBRAL_ALTO}: {len(d['columnas_ks_mayor_umbral'])} | valor critico 5%: {d['valor_critico_5pct']}")
    print(f"Deriva natural 2022 vs 2023: KS medio {res['deriva_natural_2022_vs_2023']['ks_medio']:.3f}")
    with open(base.SALIDAS_DIR / "control_deriva_maestros.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
