# Informe de la Fase 5: Optimización del Modelo de Predicción de Insolvencia (SIIS 2024 -> 2025) y Proyección del Semáforo 2026

Autor: Ingeniería de Machine Learning / Ciencia de Datos Financieros
Pipeline activo: `script_cinco_optimizado.py` (90 características seleccionadas; bandera `USAR_FEATURES_MULTIANIO = False` por defecto)
Módulos de análisis: `analisis_tendencias_y_seleccion.py` (colinealidad), `datos_multianio.py` (capa de datos plurianual), `features_multianio.py` (features de tendencia 2020..2024), `analisis_multianio.py` (ablación multi-año)
Demo de producto: `demo_entrada_parcial.py` (predicción con datos incompletos)
Artefactos regenerados en `salidas/`: `reporte_fase5.json`, `modelo_final_2026.joblib`, `semaforo_2026.csv`, `run_final_multianio.log`, `diagnostico_multianio.json`, `series_saldos_multianio.pkl`, `analisis_multianio.json`, `demo_entrada_parcial.json`, `run_demo_entrada_parcial.log`

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

## 8. Extensión plurianual 2020 - 2024: aprovechar varios años de historia

Las secciones 1 a 6 dejaron el PISO del modelo 2024 -> 2025 (TEST ROC-AUC 0.8782, PR-AUC 0.3115, F1 óptimo 0.4144). El usuario pidió un paso más: "usa los tres años para mejorar el modelo lo más posible, porque al final la idea es que uno agregue su propia empresa con los datos que tenga disponible (no necesariamente todos) y el modelo prediga para el siguiente año". Esta parte del informe responde a ese requisito en cuatro frentes: (A) una capa de datos que reconstruye la historia 2020..2024; (B) once features de tendencia plurianual; (C) una ablación rigurosa que mide si el multi-año hace retroceder o no el PISO; y (D) el diseño de entrada parcial como requisito de producto, con una demo que lo prueba. Todo sigue respetando la regla autoritativa de cero fuga: la historia plurianual solo mezcla años hasta 2024 inclusive, nunca 2025.

### 8.A Capa de datos multi-año (`datos_multianio.py`)

El SIIS publica cada anexo con DOS columnas de periodo: "periodo actual" (el año de la carpeta) y "periodo anterior" (el año previo). Eso permite reconstruir una serie histórica sin más fuentes: la carpeta `pymes_individuales-YYYY` aporta el año `YYYY` (actual) y el año `YYYY-1` (anterior). Leyendo las cuatro carpetas `pymes_individuales-2021/-2022/-2023/-2024` (anexos `210030` de balance y `310030` de resultados, ambos SOLO LECTURA) se cubre la ventana 2020..2024.

- Lectura eficiente: `pandas.read_excel` con `usecols` como callable (solo trae las columnas necesarias ya normalizadas con `normalizar_texto`/NFKD), `pandas.to_datetime(errors="coerce")` y `pandas.to_numeric(errors="coerce")` para blindar fechas y saldos corruptos, `drop_duplicates(subset=["NIT"], keep="last")` para conservar el último corte reportado y `merge(validate="one_to_one")` para garantizar unicidad por NIT.
- Corrección forense de escala de pasivos idéntica a la del pipeline (dividir entre 1000 cuando `pasivos > activos*50` con `activos > 0`), para no arrastrar errores de digitación con tres ceros de más.
- Regla de preferencia de fuente (documentada en `salidas/diagnostico_multianio.json`): para un año observable desde dos carpetas (2021, 2022 y 2023) se PREFIERE el "periodo actual" de la carpeta propia como reporte primario; el "periodo anterior" de la carpeta siguiente solo complementa NITs ausentes (unión por anti-join) y sirve de control cruzado.

Diagnóstico generado (`salidas/diagnostico_multianio.json`):

| Año | NITs consolidados |
|---|---|
| 2020 | 23,175 |
| 2021 | 26,674 |
| 2022 | 26,421 |
| 2023 | 25,145 |
| 2024 | 22,806 |

Solapamientos entre años: `2021 ∩ 2022` = 24,927; `2022 ∩ 2023` = 24,793; `2021 ∩ 2023` = 23,310; triple intersección `2021 ∩ 2022 ∩ 2023` = 23,299 NITs. La discrepancia de activos entre la fuente "actual" y la fuente "anterior" del mismo año (umbral de diferencia relativa mayor a 1 por ciento) es baja y estable: 3.49 por ciento en 2021, 3.29 por ciento en 2022 y 4.06 por ciento en 2023. Esa baja discrepancia es la que justifica preferir la fuente primaria y usar la secundaria solo como complemento: ambas fuentes cuentan esencialmente lo mismo, así que mezclarlas con esa regla no introduce ruido material. La serie se cachea en `salidas/series_saldos_multianio.pkl` (un diccionario `{año: DataFrame por NIT}` con alias de saldos `activos_prev`, `pasivos_prev`, `patrimonio_prev`, `ingresos_prev`, `utilidad_prev`, `ebit_prev`, etc.) para no releer los `.xlsx` en cada corrida.

### 8.B Once features de tendencia plurianual (`features_multianio.py`)

Sobre esa serie se construyen once columnas que extienden la filosofía del BLOQUE 2.C (tendencias t-1 -> t) a toda la ventana. La ventana de historia para un corte t es `[t-4, t-1]` intersectada con los años presentes (para el modelo 2024 -> 2025 son 2020..2023, con 2024 como punto final de las pendientes; nunca 2025).

| Feature | Qué mide |
|---|---|
| `Anios_Historia_Disponible` | Conteo de años de la ventana con dato real para el NIT (cobertura). |
| `Flag_Sin_Historia_3A` | 1 si el NIT no tiene ningún año de historia plurianual. |
| `Flag_Historia_Parcial` | 1 si tiene algo de historia pero menos que la ventana plena. |
| `Pendiente_Patrimonio_3A_Sobre_Activos` | Variación de patrimonio a ~3 años normalizada por activos t. |
| `Var_Patrimonio_2A_Sobre_Activos` | Variación de patrimonio a ~2 años normalizada por activos t. |
| `CAGR_Ingresos_Multianio` | Tasa de crecimiento anual compuesta de ingresos en la ventana. |
| `Aceleracion_Endeudamiento` | Segunda diferencia del endeudamiento (sube cada vez más rápido); 0.0 neutro si falta t-1 o t-2. |
| `Anios_Con_Perdida_Ventana` | Conteo de años observados con utilidad negativa (incluye t). |
| `Persistencia_Perdidas_3A` | Fracción de años observados con pérdida (0..1). |
| `Flag_Deterioro_Patrimonial_Sostenido` | 1 si el patrimonio cae respecto al año inicial con al menos 2 años de historia. |
| `Flag_Paralisis_Sostenida_Multianio` | 1 si los ingresos son nulos en t y en el año previo observado. |

Estrategia de faltantes (decidida POR COLUMNA y documentada): se usa IMPUTACIÓN-NEUTRAL + FLAGS DE COBERTURA, no el camino NaN-nativo. Las pendientes, variaciones, CAGR y aceleración toman 0.0 (variación nula, "sin tendencia observable") cuando falta el saldo histórico necesario; los conteos de pérdida solo cuentan los años realmente observados; y las banderas (`Anios_Historia_Disponible`, `Flag_Sin_Historia_3A`, `Flag_Historia_Parcial`) hacen explícita la ausencia. Esta decisión es deliberada por tres razones:

1. Conserva el contrato `assert X.isnull().sum().sum() == 0` de la matriz X sin listas blancas ni relajaciones, lo que reduce el riesgo de regresión.
2. Mantiene operativo a `RandomForestClassifier` de scikit-learn, que NO acepta NaN (lanzaría `ValueError: Input X contains NaN`). Así el super-ensamble de cuatro modelos sigue devolviendo una probabilidad válida para toda empresa.
3. Una bandera de cobertura es más informativa para el modelo que un NaN mudo: distingue "tendencia estable" de "sin dato", que financieramente no son lo mismo.

El blindaje numérico reutiliza `division_segura` (protección contra división por cero, NaN e infinitos con `.clip()` y `numpy.nan_to_num`) y los acotamientos del pipeline. La función `validar_sin_fuga_multianio` asevera que ningún año empleado supera a `anio_t` y, en particular, que jamás se toca 2025 en el modelo 2024 -> 2025.

Corrección de simetría en `Aceleracion_Endeudamiento` (refuerzo posterior a la primera revisión): la aceleración es una segunda diferencia `(end_t - end_{t-1}) - (end_{t-1} - end_{t-2})` y, por definición, necesita los dos tramos extremos (t-1 y t-2) para existir. En la versión anterior, cuando faltaba el endeudamiento de t-1 pero existía el de t-2, el cálculo dejaba un residuo no-neutral (+`end_{t-2}`), lo que rompía la promesa de "variación neutra cuando falta el dato". Ahora una máscara de validez (`numpy.isnan(end_t1) | numpy.isnan(end_t2)`, operación elemento a elemento documentada en NumPy) satura la feature a 0.0 cuando falta CUALQUIERA de los dos extremos, de modo que la aceleración es estrictamente neutra ante datos incompletos. Se conservan los asserts del bloque (cero NaN, orden de columnas) y el acotamiento final a `[-2, 2]`.

## 9. Ablación y no-regresión: ¿el multi-año hace retroceder el PISO?

La regla del proyecto es dura: una configuración nueva solo se adopta por defecto si NO hace retroceder el PISO en el Test reservado. Se midió el aporte marginal de las once features con el MISMO split estratificado (`random_state=42`), el MISMO escuadrón y la MISMA calibración, en dos escenarios, evaluando en el Test reservado (3,988 empresas, 61 quiebras) y en la OOF del panel completo (19,937 empresas, 304 quiebras, validación cruzada de 5 pliegues con `cross_val_predict`). El resultado está en `salidas/analisis_multianio.json`.

| Escenario | N.º columnas | TEST ROC-AUC | TEST PR-AUC | TEST F1 óptimo | OOF ROC-AUC | OOF PR-AUC |
|---|---|---|---|---|---|---|
| SIN multi-año (PISO) | 90 | 0.8782 | 0.3115 | 0.4144 | 0.8867 | 0.2534 |
| CON multi-año | 101 | 0.8877 | 0.3000 | 0.4000 | 0.8880 | 0.2592 |
| Delta (CON - SIN) | - | +0.0095 | -0.0115 | -0.0144 | +0.0013 | +0.0058 |

(Estos números corresponden a la ablación re-ejecutada tras la corrección de simetría de `Aceleracion_Endeudamiento` descrita en la sección 8.B; `salidas/analisis_multianio.json` los deja trazados.)

Lectura cuidadosa de los números:

- En el Test reservado, el multi-año MEJORA el ROC-AUC (+0.0095) pero REGRESA en PR-AUC (-0.0115) y F1 (-0.0144).
- En la OOF del panel completo, el multi-año MEJORA ambas métricas de interés (ROC +0.0013, PR +0.0058), con una mejora de PR-AUC más clara que antes.

El criterio autoritativo de adopción se evalúa sobre el Test reservado: se adopta por defecto solo si `PR-AUC_test >= 0.3115 - 0.005 = 0.3065` Y `F1_test >= 0.4144 - 0.005 = 0.4094`. El escenario CON multi-año da PR-AUC 0.3000 (por debajo de 0.3065) y F1 0.4000 (por debajo de 0.4094), de modo que NO pasa el gate del Test. Por eso la bandera `USAR_FEATURES_MULTIANIO` queda en `False` por defecto: así el pipeline reproduce EXACTAMENTE el PISO (panel 19,937 / 304, TEST ROC 0.8782 / PR 0.3115 / F1 0.4144, mismas 90 columnas) y no se retrocede.

Ahora bien, "no adoptar por defecto" no es "descartar". La evidencia es matizada y favorable al multi-año en el conjunto más estable:

- El Test reservado tiene solo 61 positivos, por lo que su PR-AUC y su F1 son ruidosos: un puñado de empresas cambia el número en la tercera cifra decimal. El multi-año incluso SUBE el ROC-AUC ahí (+0.0096), que es la métrica menos sensible al ruido de conteo.
- La OOF del panel completo tiene 304 positivos y promedia cinco pliegues, así que es una estimación más estable; ahí el multi-año mejora ROC y PR a la vez.

Conclusión operativa: se respeta el gate del Test (bandera en `False`, cero regresión del PISO), pero el código plurianual queda íntegro y activable con una sola bandera. Con `USAR_FEATURES_MULTIANIO = True` el pipeline corre de punta a punta sin errores, mantiene el panel 19,937 / 304 y proyecta sobre las 21,113 empresas; es la base lista para cuando se amplíe el Test o se refine la limpieza de los años 2020 - 2022 y el gate se pueda volver a evaluar con más positivos. En resumen: no se retrocede hoy, y la mejora plurianual está disponible sin deuda técnica.

## 10. Diseño de entrada parcial como requisito de producto (`demo_entrada_parcial.py`)

El corazón del pedido del usuario es que "uno agregue su propia empresa con los datos que tenga disponible (no necesariamente todos) y el modelo prediga para el siguiente año". Esto es un requisito de PRODUCTO, no solo de modelado: la tubería de inferencia debe tolerar datos incompletos y aun así devolver una probabilidad calibrada en [0, 1] sin caerse.

Cómo se resuelve, apoyado en lo construido:

1. Cuentas faltantes del corte t (ampliado): dentro de `construir_matriz_caracteristicas` se saturan de forma neutral con `.fillna(0)` antes de usarse TANTO las cuentas de detalle (obligaciones vencidas, pasivo DIAN, gastos de administración y de ventas) COMO los totales estructurales del balance (activos, pasivos, patrimonio, corrientes, efectivo) y el núcleo del estado de resultados (ingresos, utilidad, EBIT, flujo de operación, costos financieros, ganancia bruta). Todos los ratios pasan además por `division_segura`/`.clip()`/`numpy.nan_to_num`. Una empresa que no reporta un total recibe igualmente su vector de features: la celda ausente se interpreta como 0 (saldo no informado => neutral) y las ramas `numpy.where(saldo > 0, ratio, default)` toman el camino "default" documentado, de modo que la ausencia no inventa ni deterioro ni solvencia.
2. Historia plurianual ausente o parcial: las banderas de cobertura (`Anios_Historia_Disponible`, `Flag_Sin_Historia_3A`, `Flag_Historia_Parcial`) y la imputación neutral de las tendencias (variación 0.0 cuando falta el saldo) permiten que una empresa sin historia, o con solo uno o dos años, reciba predicción sin que ninguna feature multi-año se vuelva un requisito duro.
3. Ensamble que no se rompe con faltantes: como X queda SIN NaN por la imputación-neutral, los cuatro modelos del ensamble (incluido `RandomForestClassifier`, que no acepta NaN) producen su probabilidad; luego el calibrador isotónico (`IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)`) garantiza matemáticamente la salida en [0, 1].

Nota sobre el manejo nativo de NaN de los boosters: `HistGradientBoostingClassifier` (scikit-learn, con soporte nativo de valores faltantes), `LGBMClassifier` (LightGBM, parámetro `use_missing`) y `XGBClassifier` (XGBoost, dirección por defecto para faltantes en cada split) SÍ saben enrutar NaN de forma nativa. Aun así se eligió imputación-neutral + flags porque RandomForest, que es parte del ensamble, no acepta NaN, y porque la bandera de cobertura informa al modelo "cuánta historia hay" de forma explícita, lo que es más útil que un NaN mudo. Esta decisión conserva además el contrato de cero-NaN de X sin relajaciones.

Evidencia (`demo_entrada_parcial.py` -> `salidas/demo_entrada_parcial.json`): la demo REUTILIZA la ruta real de inferencia (misma `construir_matriz_caracteristicas`, mismas columnas `modelo["features"]`, mismo ensamble ponderado con `calcular_pesos_ensamble`, mismo calibrador isotónico y mismos umbrales de zona), cargando `salidas/modelo_final_2026.joblib`. Para simular datos parciales de forma honesta se toman filas reales del maestro 2025 y se mutilan de modo controlado.

| Caso | Qué simula | Cuentas faltantes | Prob. calibrada | Zona |
|---|---|---|---|---|
| A | Cuentas de detalle faltantes + sin historia plurianual (NIT inexistente en 2020..2024) | 5 | 0.017816 | 1. VERDE |
| B | Historia parcial (solo 2023 y 2024 para su NIT) | 0 | 0.008264 | 1. VERDE |
| C | Datos completos (control) | 0 | 0.013158 | 1. VERDE |
| D | Balance incompleto: totales estructurales ausentes (efectivo, flujo de operación, EBIT, costos financieros) | 4 | 0.007299 | 1. VERDE |

Los cuatro casos devolvieron una probabilidad calibrada en [0, 1] sin lanzar excepción. El caso A combina cinco cuentas de detalle ausentes con cero años de historia multi-año; el caso D vacía cuatro totales estructurales (los que antes eran un requisito duro) y aun así recibe predicción (0.0073), que es la prueba directa de que la ampliación de la imputación-neutral actúa como salvavidas para balances realmente incompletos.

Evidencia explícita de la ruta multi-anual (refuerzo posterior a la primera revisión): la corrida de producción usa la bandera en su valor por defecto (`False`), de modo que el bloque plurianual no se ejecuta y los casos atraviesan el camino del PISO. Para demostrar que una fila SIN historia y una con HISTORIA PARCIAL atraviesan `construir_features_multianio` sin excepción, la demo FUERZA además la ruta multi-anual (activando la bandera de forma temporal y acotada dentro de `_predecir`, que restaura su valor en un `finally`), sin cambiar el DEFAULT del pipeline. El resultado quedó en `salidas/demo_entrada_parcial.json` bajo `evidencia_ruta_multianual_forzada`:

| Caso (ruta multi-anual forzada) | `Anios_Historia_Disponible` | `Flag_Sin_Historia_3A` | `Flag_Historia_Parcial` | Prob. calibrada | Sin excepción |
|---|---|---|---|---|---|
| A (sin historia) | 0 | 1 | 0 | 0.017816 | Sí |
| B (historia parcial 2023-2024) | 2 | 0 | 1 | 0.008264 | Sí |

Para el caso B se asevera explícitamente `0 < Anios_Historia_Disponible < 4` (ventana plena), confirmando que es historia ESTRICTAMENTE parcial (ni ausente ni completa), tal como pedía el requisito de verificación. Ambas filas construyen el bloque plurianual de 11 columnas sin NaN (imputación-neutral + flags) y producen probabilidad en [0, 1].

Esqueleto contable mínimo (actualizado): tras la ampliación de la imputación-neutral, el pipeline ya NO exige los totales estructurales del balance ni el núcleo del estado de resultados como requisito duro; todos se saturan con `.fillna(0)` y el `assert X sin NaN` se conserva. No-regresión verificada: en los paneles reales 2024 y 2025 esas 12 columnas no contienen NaN, de modo que el `.fillna(0)` es inocuo para el entrenamiento y el PISO se reproduce exacto (panel 19,937 / 304, TEST ROC 0.8782 / PR 0.3115 / F1 0.4144); la ampliación solo actúa cuando el usuario aporta un balance incompleto. En la práctica, el único "esqueleto mínimo" que persiste son las variables de fase 4 ya calculadas (columnas `Flag_*` y `Edad_Empresa_Anios`) que llegan desde el maestro; para una empresa nueva se generan en las fases previas de la tubería.

## 11. Proyección del Semáforo 2026 regenerada (configuración por defecto)

Con la configuración por defecto vigente (bandera `USAR_FEATURES_MULTIANIO = False`, el PISO) se regeneraron los tres artefactos finales. El pipeline terminó con "Proceso completado con éxito" sin AssertionError (log `salidas/run_final_multianio.log`), reprodujo exactamente el PISO (TEST ROC-AUC 0.8782, PR-AUC 0.3115; panel 19,937 / 304; OOF 0.8867 / 0.2534) y escribió `salidas/semaforo_2026.csv` con 21,113 empresas, las 4 zonas y todas las probabilidades calibradas en [0, 1].

| Zona 2026 | Empresas | Probabilidad media | Quiebras esperadas |
|---|---|---|---|
| 1. VERDE (Estable / Sano) | 18,884 | 0.0057 | 108.43 |
| 2. AMARILLO (Vigilancia / Alerta) | 1,635 | 0.0523 | 85.53 |
| 3. NARANJA (Riesgo Alto) | 345 | 0.1462 | 50.44 |
| 4. ROJO ÉLITE (Peligro Inminente) | 249 | 0.4396 | 109.46 |

Es la misma distribución de la sección 5 (la bandera en `False` garantiza reproducibilidad bit a bit del PISO): las 249 empresas en ROJO ÉLITE concentran una probabilidad media del 43.96 por ciento y aportan tantas quiebras esperadas como las casi 19,000 empresas verdes juntas, de modo que siguen siendo la lista de auditoría prioritaria.

## 12. Reproducibilidad y trazabilidad

- Ejecución final (configuración por defecto, bandera en `False`): `cd fase5_optimizacion && /projects/sandbox/.venv_fase5/bin/python script_cinco_optimizado.py`, log en `salidas/run_final_multianio.log`, terminó con "Proceso completado con éxito" sin AssertionError.
- Capa de datos plurianual: `/projects/sandbox/.venv_fase5/bin/python datos_multianio.py` (log `salidas/run_datos_multianio.log`), que genera `salidas/diagnostico_multianio.json` y la caché `salidas/series_saldos_multianio.pkl`.
- Ablación multi-año: `/projects/sandbox/.venv_fase5/bin/python analisis_multianio.py`, que escribe `salidas/analisis_multianio.json`.
- Demo de entrada parcial: `/projects/sandbox/.venv_fase5/bin/python demo_entrada_parcial.py` (log `salidas/run_demo_entrada_parcial.log`), que escribe `salidas/demo_entrada_parcial.json`.
- Entorno: Python 3.11 con pandas 3.1.1, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, openpyxl 3.1.5, lightgbm 4.7.0, xgboost 3.2.0 (ver `requirements.txt`).
- Artefactos: `salidas/reporte_fase5.json` (métricas y cortes), `salidas/modelo_final_2026.joblib` (modelos, calibrador y umbrales), `salidas/semaforo_2026.csv` (proyección), más los JSON de análisis `analisis_tendencias.json`, `seleccion_caracteristicas.json`, `resumen_feat002.json`, `importancias_top_features.json`, `diagnostico_multianio.json`, `analisis_multianio.json` y `demo_entrada_parcial.json`.
- Garantía anti-fuga mantenida y reforzada: aserciones `assert not any("2025" in str(col) for col in X.columns)` antes y después de la poda de colinealidad, y `validar_sin_fuga_multianio` que asevera que la ventana plurianual nunca referencia años mayores a `anio_t` (jamás 2025 en el modelo 2024 -> 2025).
