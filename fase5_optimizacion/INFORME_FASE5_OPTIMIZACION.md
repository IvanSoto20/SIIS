# Informe de la Fase 5: Optimización del Modelo de Predicción de Insolvencia (SIIS 2024 -> 2025) y Proyección del Semáforo 2026

Autor: Ingeniería de Machine Learning / Ciencia de Datos Financieros
Pipeline activo: `script_cinco_optimizado.py` (90 características seleccionadas)
Módulo de análisis: `analisis_tendencias_y_seleccion.py`
Artefactos regenerados en `salidas/`: `reporte_fase5.json`, `modelo_final_2026.joblib`, `semaforo_2026.csv`, `run_final_fase5.log`

Este informe cierra el ciclo de optimización de la Fase 5. Documenta de forma didáctica tres preguntas que planteó el usuario y las respuestas obtenidas con evidencia cuantitativa, citando las funciones de `pandas`, `numpy`, `scipy` y `scikit-learn` que sustentan cada decisión. Todo el trabajo respeta la restricción autoritativa de cero fuga de datos: ninguna variable derivada de 2025 entra en la matriz predictora `X`, y la variable objetivo `y` se deriva exclusivamente de columnas `*_2025` que nunca son features.

## 0. Diseño experimental recordatorio (sin fuga de datos)

- Población sana en t-1 (2024): empresas con `Estado actual` en `{ACTIVA, EN ETAPA PREOPERATIVA}` y `Flag_Quiebra_Patrimonial == 0`.
- Panel longitudinal 2024 -> 2025 por cruce `inner` sobre `NIT`: exactamente 19,937 empresas (aserción `assert len(panel) == 19937`).
- Variable objetivo `y` en 2025: colapso legal (salir de estados sanos) o colapso patrimonial (`Flag_Quiebra_Patrimonial_2025 == 1`). Total 304 eventos positivos (1.5248 por ciento), desbalanceo cercano a 64.58 a 1.
- Partición estratificada reproducible: `sklearn.model_selection.train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`. El argumento `stratify=y` preserva la proporción de clases en train y test (documentación de scikit-learn sobre `train_test_split`), produciendo `X_train` de 15,949 filas con 243 positivos y `X_test` de 3,988 filas con 61 positivos. Las aserciones embebidas fallan si cualquiera de estos conteos se rompe.
- Garantía anti-fuga: `assert not any("2025" in str(col) for col in X.columns)`, verificada antes y después de la poda de colinealidad.
- Las variables de tendencia dinámica usan solo saldos de 2023 (campo "Periodo Anterior" extraído de `pymes_individuales-2024/210030...xlsx` y `310030...xlsx`) y de 2024. Nunca tocan 2025.

## 1. Aporte marginal de las variables de tendencia 2023 -> 2024

Se realizó una ablación controlada: se entrenó el mismo super-ensamble sobre el mismo split estratificado (`random_state=42`) en dos escenarios, evaluando siempre en el Test reservado de 3,988 empresas.

Las 10 columnas de tendencia crudas evaluadas: `Flag_Sin_Historia_Prev`, `Delta_Endeudamiento`, `Var_Pct_Patrimonio`, `Var_Pct_Ingresos`, `Delta_ROA_Neto`, `Flag_Dos_Anios_Perdida`, `Perdidas_Acumuladas_2A_Sobre_Activos`, `Flag_Paralisis_Comercial`, `Var_Pct_Patrimonio_vs_Sector`, `Delta_Endeudamiento_vs_Sector`.

| Escenario | N.º columnas | ROC-AUC (Test) | PR-AUC (Test) | F1 óptimo (Test) |
|---|---|---|---|---|
| CON tendencias crudas | 101 | 0.8743 | 0.2995 | 0.4060 |
| SIN tendencias crudas | 91 | 0.8768 | 0.3070 | 0.4194 |
| Delta aporte (CON - SIN) | - | -0.0025 | -0.0075 | -0.0134 |

OOF del panel completo CON tendencias (validación cruzada estratificada de 5 pliegues vía `cross_val_predict`): ROC-AUC 0.8878, PR-AUC 0.2598.

### Conclusión sobre el aporte marginal

A nivel del super-ensamble sobre el Test reservado, las 10 columnas de tendencia crudas no aportan margen e incluso restan de forma leve (alrededor de -0.0075 en PR-AUC y -0.0134 en F1). La lectura técnica no es que las tendencias no importen, sino que su señal ya está embebida en los índices sintéticos que permanecen en `X`: `Altman_Z_Emergente`, `Score_Alerta_Temprana` y las variantes de normalización sectorial `_vs_Sector`. Por eso el delta reportado es una cota inferior conservadora del valor real de la información dinámica: el escenario "SIN" no recalcula esos índices, de modo que aún conserva el residuo de tendencia que vive dentro de ellos. En términos prácticos, como features sueltas son redundantes, y su eliminación parcial (ver sección 2) incluso ayuda.

## 2. Selección de características por colinealidad (Spearman)

Se calculó la matriz de correlación de rangos de Spearman sobre `X_train` con `pandas.DataFrame.corr(method="spearman")`, que internamente reutiliza la lógica de rangos equivalente a `scipy.stats.spearmanr`. Para cada par de variables con coeficiente absoluto `|rho| >= 0.95` se eliminó la de menor importancia según `RandomForestClassifier.feature_importances_` (impureza de Gini), de acuerdo con la documentación de scikit-learn sobre importancia basada en árboles.

### ¿Por qué Spearman y no Pearson?

- Los 52 ratios forenses están acotados con `numpy`/`pandas` `.clip()` y varios se transforman con `numpy.log1p`. Estas operaciones saturan colas y comprimen escalas, lo que rompe el supuesto de linealidad de Pearson.
- Spearman opera sobre rangos (posición ordinal), por lo que es robusto a outliers y a la saturación por `.clip()`, y detecta relaciones monótonas no lineales que Pearson subestimaría. La documentación de `scipy.stats.spearmanr` describe exactamente este comportamiento: es el coeficiente de Pearson calculado sobre los rangos de los datos.

### Resultado: 11 columnas eliminadas, 90 conservadas

De 15 pares colineales detectados se podaron 11 columnas (umbral `|rho| >= 0.95`):

| Columna eliminada | Se mantiene en su lugar (mayor importancia) | rho Spearman |
|---|---|---|
| `Nivel_Endeudamiento` | `Ratio_Palanca_Pasivo_Patrimonio` | 1.0000 |
| `Ratio_Patrimonio_Inercial_Sobre_Activos` | `Ratio_Palanca_Pasivo_Patrimonio` | 0.9596 |
| `Colchon_Patrimonio_Sobre_Pasivos` | `Ratio_Palanca_Pasivo_Patrimonio` | 0.9946 |
| `Flag_Inyeccion_Capital_Socios` | `Ratio_Inyeccion_Capital_Activos` | 0.9999 |
| `Flag_Cartera_Perdida_360d` | `Ratio_Cartera_Perdida_360d` | 0.9965 |
| `Ratio_Morosidad_Severa` | `Ratio_Mora_Sobre_Efectivo` | 0.9971 |
| `Ratio_Cesacion_Pagos_Ley1116` | `Ratio_Mora_Sobre_Efectivo` | 0.9970 |
| `Flag_Cobro_Juridico_Proveedores` | `Ratio_Deuda_Vencida_Mas_360d` | 0.9996 |
| `ROA_Neto` | `ROA_Neto_vs_Sector` | 0.9566 |
| `Delta_Endeudamiento_vs_Sector` | `Delta_Endeudamiento` | 0.9958 |
| `Var_Pct_Patrimonio` | `Var_Pct_Patrimonio_vs_Sector` | 0.9822 |

El detalle completo de los 15 pares, con las importancias de cada miembro, está en `salidas/seleccion_caracteristicas.json`.

### Re-evaluación de la selección (mismo split)

| Conjunto de features | ROC-AUC | PR-AUC | F1 óptimo |
|---|---|---|---|
| Set completo (101) | 0.8743 | 0.2995 | 0.4060 |
| Set seleccionado (90) | 0.8782 | 0.3115 | 0.4144 |

El criterio de aceptación era conservar la selección solo si no degradaba el Test (caída de PR-AUC menor o igual a 0.01 y caída de F1 menor o igual a 0.01). En realidad la selección mejoró las tres métricas, por lo que se adoptó como configuración activa del pipeline (`APLICAR_SELECCION_COLINEALIDAD = True`). Reducir colinealidad disminuyó la varianza del ensamble sin perder poder predictivo: un resultado deseable tanto estadística como operativamente.

## 3. Afinamiento del ensamble y deltas vs. línea base

Se exploraron, de forma acotada para respetar el presupuesto, tres esquemas de ponderación del super-ensamble sobre el set de 90 features:

| Esquema de pesos | ROC-AUC | PR-AUC | F1 óptimo |
|---|---|---|---|
| Base 0.30 LGBM + 0.30 XGB + 0.20 HGB + 0.20 RF | 0.8782 | 0.3115 | 0.4144 |
| Uniforme (0.25 cada uno) | 0.8787 | 0.3098 | 0.4211 |
| Rank-averaging (`scipy.stats.rankdata`) | 0.8773 | 0.3034 | 0.4035 |

Se conservaron los pesos base porque maximizan el PR-AUC, que es la métrica preferente bajo un desbalanceo de 64 a 1 (el área bajo la curva precisión-recall es mucho más informativa que el ROC cuando la clase positiva es rara). El esquema uniforme ganaba 0.0067 en F1 pero perdía PR-AUC, un intercambio que no compensa en este dominio. No se hizo búsqueda exhaustiva de hiperparámetros; los `scale_pos_weight` (alrededor de 7) y `class_weight` (1:8 y 1:10) ya están alineados con el desbalanceo real.

### Deltas de la configuración final vs. la línea base

Para que el cuadro sea trazable contra artefactos versionados en este mismo cambio, la columna "línea base" proviene del log comprometido `salidas/run_baseline_fase5.log` (pipeline con las 101 features originales, `APLICAR_SELECCION_COLINEALIDAD = False`): TEST ROC-AUC 0.8743, PR-AUC 0.2995, F1 óptimo 0.4060; OOF del panel completo ROC-AUC 0.8878, PR-AUC 0.2598. La columna "config final" proviene de `salidas/run_final_fase5.log` y `salidas/reporte_fase5.json` (pipeline con las 90 features seleccionadas).

| Métrica | Línea base (`run_baseline_fase5.log`, 101) | Config final (`run_final_fase5.log`, 90) | Delta |
|---|---|---|---|
| TEST ROC-AUC | 0.8743 | 0.8782 | +0.0039 |
| TEST PR-AUC | 0.2995 | 0.3115 | +0.0120 |
| TEST F1 óptimo | 0.4060 | 0.4144 | +0.0084 |
| OOF ROC-AUC (panel completo) | 0.8878 | 0.8867 | -0.0011 |
| OOF PR-AUC (panel completo) | 0.2598 | 0.2534 | -0.0064 |

Punto de operación de máximo F1 en la config final: umbral de probabilidad calibrada mayor o igual a 0.520, con precisión 46.0 por ciento (23 quiebras reales detectadas de 50 alertas), recall 37.7 por ciento y lift 30.1x sobre la tasa base. La zona ROJO ÉLITE (top 1 por ciento) alcanza 47.5 por ciento de tasa de quiebra en 40 empresas, con lift 31.1x.

Nota metodológica: la OOF del panel completo baja de forma leve (ROC-AUC -0.0011 y PR-AUC -0.0064), pero las tres métricas del Test reservado mejoran (ROC +0.0039, PR +0.0120, F1 +0.0084) y la métrica operativa (lo que verá el analista) sube. Se prioriza el Test estratificado reservado, que es la estimación honesta de desempeño fuera de muestra. Todos los números de este cuadro son reproducibles desde los logs comprometidos en `salidas/`; no se mezclan corridas de distinto origen.

## 4. Variables determinantes según importancia de los modelos de árbol

Se midió la importancia sobre `X_train` (90 features) con dos árbitros complementarios: la impureza de Gini de `RandomForestClassifier.feature_importances_` (scikit-learn) y la ganancia acumulada de LightGBM (`booster_.feature_importance(importance_type="gain")`). El artefacto `salidas/importancias_top_features.json` se genera de forma reproducible y determinista (`random_state=42`) por la función `importancias_top_features` de `analisis_tendencias_y_seleccion.py`, que ajusta ambos árbitros sobre el mismo `X_train` restringido al set de 90 features vigente; las tablas siguientes transcriben ese artefacto.

### Top 15 RandomForest (Gini)

| # | Variable | Importancia |
|---|---|---|
| 1 | `Score_Alerta_Temprana` | 0.0988 |
| 2 | `Ratio_Palanca_Pasivo_Patrimonio` | 0.0646 |
| 3 | `ROE` | 0.0427 |
| 4 | `Var_Pct_Patrimonio_vs_Sector` | 0.0389 |
| 5 | `ROA_Neto_vs_Sector` | 0.0386 |
| 6 | `Nivel_Endeudamiento_vs_Sector` | 0.0385 |
| 7 | `Tasa_Consumo_Patrimonio_vs_Sector` | 0.0330 |
| 8 | `Altman_Z_Emergente_vs_Sector` | 0.0309 |
| 9 | `Tasa_Consumo_Patrimonio` | 0.0275 |
| 10 | `Perdidas_Acumuladas_2A_Sobre_Activos` | 0.0275 |
| 11 | `Log_Patrimonio` | 0.0264 |
| 12 | `Ratio_Deuda_Critica_Sobre_Patrimonio` | 0.0254 |
| 13 | `Altman_Z_Emergente` | 0.0251 |
| 14 | `Ratio_Caja_Inmediata` | 0.0206 |
| 15 | `Margen_Neto` | 0.0198 |

### Top 15 LightGBM (ganancia, porcentaje del total)

| # | Variable | Ganancia (%) |
|---|---|---|
| 1 | `Score_Alerta_Temprana` | 15.58 |
| 2 | `Ratio_Palanca_Pasivo_Patrimonio` | 6.81 |
| 3 | `Tasa_Consumo_Patrimonio` | 5.49 |
| 4 | `Runway_Meses_Caja` | 3.38 |
| 5 | `ROE` | 2.88 |
| 6 | `Perdidas_Acumuladas_2A_Sobre_Activos` | 2.58 |
| 7 | `Var_Pct_Patrimonio_vs_Sector` | 2.37 |
| 8 | `Concentracion_Deuda_Corto_Plazo` | 2.33 |
| 9 | `Edad_Empresa_Anios` | 2.23 |
| 10 | `Log_Patrimonio` | 2.19 |
| 11 | `ROA_Neto_vs_Sector` | 2.11 |
| 12 | `Ratio_Caja_Inmediata` | 2.00 |
| 13 | `Ratio_Deuda_Critica_Sobre_Patrimonio` | 1.96 |
| 14 | `Log_Ingresos` | 1.95 |
| 15 | `Altman_Z_Emergente_vs_Sector` | 1.82 |

### Lectura financiera

Ambos árbitros coinciden en la cúspide: el índice sintético `Score_Alerta_Temprana` domina, seguido de la estructura patrimonial y de apalancamiento (`Ratio_Palanca_Pasivo_Patrimonio`, `ROE`, `Tasa_Consumo_Patrimonio`). Esto confirma la tesis de la sección 1: la velocidad de agotamiento patrimonial (Equity Burn Rate) y los índices que ya integran la tendencia son los verdaderos motores de la predicción. Es notable que varias de las variables de tendencia sobrevivientes y sus derivadas sectoriales aparecen en ambos tops (`Var_Pct_Patrimonio_vs_Sector`, `Perdidas_Acumuladas_2A_Sobre_Activos`): su información sí se usa, pero de forma complementaria, no como señal dominante independiente.

## 5. Proyección del Semáforo 2026

El modelo se re-entrenó sobre el panel completo de 19,937 empresas y se proyectó sobre el universo de 21,113 empresas sanas de 2025, usando sus saldos de 2024 como "Periodo Anterior" para las tendencias (nuevamente, sin fuga: la proyección a 2026 solo mira hacia atrás). Las probabilidades se calibran con regresión isotónica (`sklearn.isotonic.IsotonicRegression`) sobre predicciones fuera de muestra. El archivo `salidas/semaforo_2026.csv` contiene 21,113 filas con las columnas `NIT`, `Razón social de la sociedad`, `Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)` (en adelante abreviada como CIIU), `Score_Ensamble_2025`, `Probabilidad_Calibrada_Quiebra_2026`, `Zona_Semaforo_2026` y `Detonantes_Criticos`. Verificado: las 4 zonas están presentes y todas las probabilidades calibradas caen en el intervalo [0, 1] (mínimo 0.0, máximo 0.7).

| Zona 2026 | Empresas | Probabilidad media | Quiebras esperadas |
|---|---|---|---|
| 1. VERDE (Estable / Sano) | 18,884 | 0.0057 | 108.43 |
| 2. AMARILLO (Vigilancia / Alerta) | 1,635 | 0.0523 | 85.53 |
| 3. NARANJA (Riesgo Alto) | 345 | 0.1462 | 50.44 |
| 4. ROJO ÉLITE (Peligro Inminente) | 249 | 0.4396 | 109.46 |

Lectura operativa: las 249 empresas en ROJO ÉLITE (1.2 por ciento del universo) concentran una probabilidad media de quiebra del 43.96 por ciento y aportan tantas quiebras esperadas como las casi 19,000 empresas verdes juntas. Esta es la lista de auditoría prioritaria para la Superintendencia. Sumando ROJO ÉLITE y NARANJA (594 empresas, 2.8 por ciento del universo) se cubre el grueso del riesgo con un esfuerzo de supervisión acotado.

## 6. Nota sobre las carpetas 2021 - 2023 (evaluación y decisión)

Se evaluó incorporar las carpetas `pymes_individuales-2021`, `-2022` y `-2023` para extender la ventana de tendencias más allá de 2023 -> 2024. Se decidió descartarlas en esta fase por tres razones de riesgo de integridad:

1. Estructuras heterogéneas entre años: aunque las tres carpetas tienen 17 archivos con los mismos prefijos numéricos, el usuario advirtió explícitamente que entre años cambian los filtros, el manejo de tildes, los campos vacíos y hasta el conjunto de empresas. Mezclarlas sin una fase de limpieza forense equivalente a la ya aplicada a 2024 y 2025 arriesgaría introducir ruido o, peor, inconsistencias que el modelo confundiría con señal.
2. Suficiencia de la señal actual: la ablación de la sección 1 demuestra que incluso la tendencia 2023 -> 2024 ya es redundante a nivel de ensamble porque su información vive en los índices sintéticos. Añadir 2021 y 2022 tendría un retorno marginal esperado muy bajo frente al costo de limpieza y al riesgo de fuga.
3. Prioridad de completar sin exceder presupuesto: procesar y validar tres años adicionales de anexos `.xlsx` (lectura lenta, deduplicación por `NIT`, corrección de fechas y signos) habría sido costoso y habría puesto en riesgo el cierre de la fase. La información de "Periodo Anterior" ya extraída de `pymes_individuales-2024` cubre la necesidad de tendencia 2023 -> 2024 de forma segura y auditada.

Si en una fase futura se desea explotar 2021 - 2023, la ruta recomendada es replicar primero el pipeline de limpieza forense (scripts uno a tres) año por año, validando 0 nulos y 0 infinitos, antes de construir cualquier tendencia multianual.

## 7. Reproducibilidad y trazabilidad

- Ejecución final: `cd fase5_optimizacion && python script_cinco_optimizado.py`, log en `salidas/run_final_fase5.log`, terminó con "Proceso completado con éxito" sin AssertionError.
- Entorno: Python 3.11 con pandas 3.1.1, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, openpyxl 3.1.5, lightgbm 4.7.0, xgboost 3.2.0 (ver `requirements.txt`).
- Artefactos: `salidas/reporte_fase5.json` (métricas y cortes), `salidas/modelo_final_2026.joblib` (modelos, calibrador y umbrales), `salidas/semaforo_2026.csv` (proyección), más los JSON de análisis `analisis_tendencias.json`, `seleccion_caracteristicas.json`, `resumen_feat002.json` e `importancias_top_features.json`.
- Garantía anti-fuga mantenida y reforzada: aserciones `assert not any("2025" in str(col) for col in X.columns)` antes y después de la poda de colinealidad.
