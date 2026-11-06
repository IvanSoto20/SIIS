# Informe de la Fase 5: Optimización del Modelo de Predicción de Insolvencia (SIIS 2024 -> 2025) y Proyección del Semáforo 2026

Autor: Ingeniería de Machine Learning / Ciencia de Datos Financieros
Pipeline activo: `script_cinco_optimizado.py` (90 características seleccionadas; bandera `USAR_FEATURES_MULTIANIO = False` por defecto)
Módulos de análisis: `analisis_tendencias_y_seleccion.py` (colinealidad), `datos_multianio.py` (capa de datos plurianual), `features_multianio.py` (features de tendencia 2020..2024), `analisis_multianio.py` (ablación multi-año)
Demo de producto: `demo_entrada_parcial.py` (predicción con datos incompletos)
Evaluación de transiciones históricas (sección 13): `maestros_historicos.py`, `control_deriva_maestros.py`, `analisis_apilado_transiciones.py`, `replica_confirmatoria_apilado.py`
Zona roja: anexos y cascada (sección 14): `features_anexos.py` (16 variables de 520000/801500/801700), `modelo_cascada.py` (cascada de dos etapas), `evaluacion_zona_roja.py` (banco de pruebas honesto y decisión); artefactos `salidas/evaluacion_zona_roja.json`, `salidas/evaluacion_zona_roja_base.json`
Guía de uso del producto: `COMO_USAR_EL_MODELO.md` (cómo puntuar una empresa, interpretar el output y leer la efectividad en lenguaje natural)
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

## 13. Más empresas para afinar la zona roja: apilar transiciones históricas (evaluada y NO aplicada)

Pedido del usuario: usar la información de empresas de años anteriores, con todos los años disponibles, para combatir el cuello de botella de pocos positivos, evaluar si es buena estrategia y aplicarla solo si lo es. Esta sección documenta la evaluación completa. Conclusión: la estrategia mejora la discriminación global (ROC-AUC) pero no mejora la zona roja, así que no se aplicó. El modelo de producción, `reporte_fase5.json`, `modelo_final_2026.joblib` y `semaforo_2026.csv` quedan sin cambios (el diff de esos archivos es vacío).

### 13.1 Qué se construyó

- Maestros históricos 2021, 2022 y 2023 con las 94 columnas y los 51 ratios de la Fase 4 (`maestros_historicos.py`, FEAT-001), reconstruidos desde los anexos del SIIS en modo solo lectura. El esquema coincide columna a columna con el de 2024.
- Saldos del "Periodo Anterior" (t-1) de cada transición: `prev_2020.pkl`, `prev_2021.pkl` y `prev_2022.pkl`, que se suman a los de 2023 y 2024.
- Tres transiciones adicionales con la misma representación de 90 variables y la misma definición `y_estricto` del panel 2024 -> 2025 (`analisis_apilado_transiciones.py`):

| Transición | Empresas | Positivos | Tasa | Colapso legal | Colapso patrimonial |
|---|---|---|---|---|---|
| 2021 -> 2022 | 19,828 | 364 | 1.836% | 149 | 223 |
| 2022 -> 2023 | 21,523 | 320 | 1.487% | 135 | 190 |
| 2023 -> 2024 | 20,911 | 344 | 1.645% | 141 | 215 |
| Historia apilada | 62,262 | 1,028 | 1.651% | | |

La historia aporta 1,028 positivos contra los 304 del panel 2024 -> 2025, es decir 4.38 veces más positivos totales. El dimensionamiento de FEAT-001 había estimado 425 porque solo contaba la parte legal de la etiqueta; al cruzar con el maestro de t+1 se suma la parte patrimonial (`Flag_Quiebra_Patrimonial`) y el total real es mayor.

Controles de calidad. La deriva entre años es baja: el estadístico de Kolmogorov-Smirnov de dos muestras (`scipy.stats.ks_2samp`) promedia 0.017 entre 2023 y 2024, 0.028 para 2022 y 0.035 para 2021, frente a 0.018 de deriva natural entre 2022 y 2023. La edad de la empresa se calcula contra la `Fecha de Corte` de cada año (no contra un año fijo), por lo que no hay desfase entre transiciones. Cada transición asevera que sus columnas coinciden con las del modelo vigente, que X no tiene nulos y que ninguna columna proviene de t+1.

### 13.2 Protocolo de evaluación (tres niveles)

1. Backtest hacia adelante: se entrena solo con transiciones que terminan en 2024 o antes y se evalúa en las 19,937 empresas de 2024 -> 2025 (304 positivos). Ninguna fila de 2025 toca el entrenamiento. Se repite un año atrás (evaluando en 2023 -> 2024) como réplica.
2. Protocolo del modelo vigente: Test reservado de 3,988 empresas (61 positivos) y OOF de 5 pliegues sobre las 19,937 empresas (`StratifiedKFold(shuffle=True, random_state=42)`). El arnés reproduce los números publicados (OOF base: ROC-AUC 0.8867 y PR-AUC 0.2534; Test base: 0.8782, 0.3115 y F1 0.4144), lo que valida la comparación.
3. Control anti-memorización: el mismo OOF, pero retirando de la historia las empresas (NIT) que se están validando en ese pliegue.

La incertidumbre se midió con bootstrap pareado (`numpy.random.default_rng`, 500 remuestreos con reemplazo sobre las mismas empresas), porque con 61 positivos las diferencias de tercera cifra decimal son ruido.

Regla de adopción declarada antes de ver resultados, para el candidato "historia apilada con peso 1.0": G1, PR-AUC del OOF mayor que la base y ROC-AUC sin caer más de 0.002; G2, aciertos del OOF en el top 1% y en el top 2.5% sin empeorar y con mejora estricta en al menos uno; G3, Test publicado dentro de 0.005 del piso en las tres métricas; G4, la versión agrupada (sin los NIT validados) sigue superando la base en PR-AUC; G5, el backtest con 3 años de historia no queda por debajo del backtest con 1 año.

### 13.3 Resultados

OOF de 5 pliegues sobre 2024 -> 2025 (304 positivos, 1,994 empresas en el top 10%):

| Variante | ROC-AUC | PR-AUC | Aciertos top 0.5% (de 100) | Aciertos top 1% (de 200) | Aciertos top 2.5% (de 499) | Aciertos top 10% (de 1,994) |
|---|---|---|---|---|---|---|
| Base (modelo vigente) | 0.8867 | 0.2534 | 50 | 85 | 124 | 206 |
| Historia apilada, peso 1.0 | 0.9002 | 0.2621 | 48 | 82 | 128 | 214 |
| Historia apilada, peso 0.5 | 0.9010 | 0.2681 | 50 | 85 | 131 | 216 |
| Peso 1.0, agrupado (sin NIT validados) | 0.9003 | 0.2514 | 46 | 84 | 122 | 214 |
| Peso 0.5, agrupado | 0.9007 | 0.2633 | 48 | 87 | 125 | 213 |

Diferencias contra la base con bootstrap pareado (media e intervalo de confianza al 95%), para el candidato declarado (peso 1.0):

| KPI | Diferencia | IC 95% | Prob. de mejora |
|---|---|---|---|
| ROC-AUC | +0.0137 | [+0.0063, +0.0210] | 1.00 |
| PR-AUC | +0.0066 | [-0.0177, +0.0289] | 0.70 |
| Precisión top 1% | -0.0130 | [-0.0550, +0.0250] | 0.23 |
| Recall top 2.5% | +0.0109 | [-0.0220, +0.0421] | 0.73 |

Test reservado (61 positivos): la base tiene ROC-AUC 0.8782, PR-AUC 0.3115 y F1 0.4144. El candidato (peso 1.0) da 0.8957, 0.3055 y 0.4211: el PR-AUC cae 0.0060, más que la tolerancia de 0.005, por lo que G3 no se cumple (el ROC-AUC y el F1 sí quedan dentro de tolerancia). En la zona roja, la base acierta 19 de 40 en el top 1% y el candidato también 19 de 40.

Backtest hacia adelante, evaluado en 2024 -> 2025 sin ninguna fila de 2025 en el entrenamiento:

| Entrenamiento | Positivos | ROC-AUC | PR-AUC | Aciertos top 1% (de 200) | F1 máximo |
|---|---|---|---|---|---|
| 1 año (2023 -> 2024) | 344 | 0.8908 | 0.2513 | 83 | 0.3300 |
| 2 años (2022 -> 2024) | 664 | 0.8926 | 0.2513 | 78 | 0.3390 |
| 3 años (2021 -> 2024) | 1,028 | 0.8992 | 0.2454 | 78 | 0.3444 |

Resultado de la regla: se cumple solo G1. Fallan G2 (el top 1% baja de 85 a 82 aciertos), G3, G4 (el PR-AUC agrupado, 0.2514, no supera la base, 0.2534) y G5 (el PR-AUC con 3 años, 0.2454, queda por debajo del de 1 año, 0.2513). Decisión automática: no adoptar.

### 13.4 La variante de peso 0.5 y su confirmación independiente

La variante con peso 0.5 se veía mejor, pero se examinó después de ver los resultados de 2024 -> 2025. Adoptarla directamente habría sido elegir al ganador a posteriori. Para resolverlo sin sesgo se replicó el experimento donde ese peso no se había elegido: panel objetivo 2023 -> 2024 (344 positivos) con historia 2021 -> 2022 y 2022 -> 2023 (684 positivos), con una regla declarada antes de ejecutar (`replica_confirmatoria_apilado.py`).

| Variante (panel 2023 -> 2024) | ROC-AUC | PR-AUC | Aciertos top 1% (de 210) | Aciertos top 2.5% (de 523) | Aciertos top 10% (de 2,092) |
|---|---|---|---|---|---|
| Base | 0.8975 | 0.2645 | 84 | 145 | 238 |
| Peso 1.0 | 0.9036 | 0.2773 | 81 | 147 | 246 |
| Peso 0.5 | 0.9045 | 0.2765 | 85 | 144 | 246 |
| Peso 0.5, agrupado | 0.9053 | 0.2760 | 81 | 148 | 246 |

Reglas: C1 (PR-AUC mejora y ROC-AUC no cae) se cumple; C3 (la versión agrupada supera la base) se cumple; C2 (zona roja sin empeorar y con mejora estricta) falla porque el top 2.5% baja de 145 a 144 aciertos. El peso 0.5 no se confirma. Se respeta la regla tal como se declaró, sin reinterpretarla a favor del resultado.

### 13.5 Lectura de los resultados

- Lo que sí mejora: la discriminación global. En las diez comparaciones de una variante apilada contra su base, el ROC-AUC de la variante es mayor. El intervalo del bootstrap queda por encima de cero en el OOF (+0.0137 con peso 1.0), en el Test (+0.0178) y en la réplica con peso 0.5 (+0.0068); con peso 1.0 en la réplica el intervalo roza el cero ([-0.0007, +0.0129]). El recall del top 10% también sube en ambos paneles (de 67.8% a 70.4% en 2024 -> 2025 con peso 1.0, y de 69.2% a 71.5% en 2023 -> 2024). La historia ayuda a ordenar mejor el cuerpo de la distribución.
- Lo que no mejora: la cabeza de la distribución, que es lo que define la zona roja. Los aciertos del top 1% quedan en 82 a 87 contra 85 de la base, y los intervalos del PR-AUC y de la precisión del top 1% incluyen el cero en todos los casos.
- La cantidad de positivos no parece ser el límite del extremo superior, aunque la evidencia sobre cuántos años de historia conviene no es consistente entre años. Un modelo entrenado solo con 2023 -> 2024 (344 positivos, sin ninguna fila de 2025) rinde en 2024 -> 2025 lo mismo que el OOF del modelo vigente (PR-AUC 0.2513 contra 0.2534), y pasar de 1 a 3 años de historia no mejora el PR-AUC (-0.0058, IC 95% [-0.0249, +0.0104]) ni el top 1% (de 83 a 78 aciertos). En la réplica evaluada en 2023 -> 2024, en cambio, pasar de 1 a 2 años de historia sí mejora de forma clara: PR-AUC +0.0306 (IC 95% [+0.0140, +0.0507]) y precisión del top 1% +5.8 puntos (IC 95% [+1.4, +10.3]), con 82 aciertos de 210 contra 68. Esa comparación parte de un modelo de un solo año de historia (320 positivos) más débil que el OOF de su propio panel (PR-AUC 0.2250 contra 0.2645), y la ganancia no reaparece cuando la historia se suma a un panel que ya se entrena con datos del mismo año (tablas de 13.3 y 13.4), que es la situación de producción. La lectura prudente es que más filas no resuelven por sí solas la zona roja y que la señal sobre el número de años de historia es ruidosa.
- Con 61 positivos en el Test, los intervalos de confianza del bootstrap son muy anchos (PR-AUC de aproximadamente -0.06 a +0.06 y precisión del top 1% de -7.5 a +12.5 puntos para el candidato de peso 1.0), así que ninguna de esas mejoras aparentes se puede distinguir del azar. Por eso se decidió con el OOF de 304 positivos y con la réplica, no con el Test.

### 13.6 Decisión y qué queda disponible

No se aplica. La regla declarada de antemano no se cumple, la variante alternativa no se confirmó en una réplica independiente, y el beneficio observable es de discriminación global, no de la zona roja que motivó el pedido. Mantener el modelo vigente cumple la regla de no retroceder: ROC-AUC 0.8782, PR-AUC 0.3115 y F1 0.4144 en el Test, y ROJO ÉLITE con 40 empresas, 47.5% de tasa de quiebra y lift 31x, sin cambios.

Queda construido y documentado, sin costo para producción: los maestros históricos, los saldos t-1 y las tres transiciones, más los dos módulos de análisis, que se pueden volver a ejecutar para reevaluar cuando haya más positivos (por ejemplo al incorporar el corte 2026). Si el objetivo pasa a ser la discriminación global (ROC-AUC) en vez de la zona roja, la historia apilada (con peso 0.5 o 1.0) es una candidata razonable, pero debería evaluarse con una regla de adopción propia, declarada antes de ejecutar, y asumiendo el costo de reentrenar con 62 mil filas adicionales.

Lo que sí podría mover la zona roja, dado que la cantidad de positivos no es el límite, es aportar información nueva en vez de más filas: ampliar la definición de positivo (`y_integral`, `Flag_Default_Sin_Efectivo`), incorporar más anexos del SIIS que hoy no se usan (flujos de efectivo, notas de ingresos y de propiedad, planta y equipo) o rediseñar variables para empresas ya en deterioro.

### 13.7 Reproducibilidad

- `cd fase5_optimizacion && /projects/sandbox/.venv_fase5/bin/python analisis_apilado_transiciones.py` regenera `salidas/analisis_apilado_transiciones.json` (unos 15 minutos). Con `APILADO_SMOKE=1` hace una prueba rápida de 3 minutos con 20 árboles y 2 pliegues, que sirve para depurar el código pero no para decidir.
- `/projects/sandbox/.venv_fase5/bin/python replica_confirmatoria_apilado.py` regenera `salidas/confirmacion_v2.json` (unos 7 minutos).
- `/projects/sandbox/.venv_fase5/bin/python control_deriva_maestros.py` regenera `salidas/control_deriva_maestros.json`, que respalda las cifras de deriva de la sección 13.1.
- Los dos módulos de análisis guardan en `salidas/cache_apilado.pkl` y `salidas/cache_confirmacion.pkl` las predicciones ya calculadas para poder reanudar si se interrumpen. Esas cachés no se versionan (están en `.gitignore`); si se modifica el código o los datos hay que borrarlas antes de volver a ejecutar, o el módulo reutilizará resultados antiguos.
- Si faltan los `prev_*.pkl`, `analisis_apilado_transiciones.py` los regenera desde los anexos del SIIS (unos 30 segundos por año).
- Ambos módulos aseveran la integridad del panel (19,937 empresas y 304 positivos, 15,949 y 243 en entrenamiento, 3,988 y 61 en Test), la ausencia de nulos y la ausencia de columnas de t+1 en X. Nada bajo `py/` ni `SIIS/` fue modificado.

## 14. Subir la zona roja con anexos nuevos y una cascada de dos etapas (evaluada y NO aplicada)

Esta sección documenta la última tanda de trabajo: dos palancas pensadas para subir la efectividad de la ZONA ROJA del modelo (el pequeño grupo de empresas marcadas como más peligrosas) y la evaluación honesta que decidió si adoptarlas. El resultado anticipado, para no perder al lector: NINGUNA de las dos palancas pasó la regla de adopción declarada de antemano, de modo que las dos quedan disponibles tras banderas apagadas y los artefactos de producción (`reporte_fase5.json`, `modelo_final_2026.joblib`, `semaforo_2026.csv`) se conservan INTACTOS en el PISO publicado.

Módulos nuevos: `features_anexos.py` (palanca 1), `modelo_cascada.py` (palanca 2) y `evaluacion_zona_roja.py` (banco de pruebas y decisión). Artefactos nuevos en `salidas/`: `evaluacion_zona_roja.json` (veredicto completo), `evaluacion_zona_roja_base.json` (reproducción del PISO del arnés), `run_evaluacion_zona_roja_full.log` y `cache_evaluacion_zona_roja.pkl` (caché de scores OOF y Test para reanudar rápido).

### 14.1 Palanca 1: integrar anexos no explotados (`features_anexos.py`)

El pipeline base usa los anexos 210030 (balance) y 310030 (resultados). La palanca 1 suma información de tres anexos que hasta ahora no entraban en el modelo, todos del corte 2024 (nunca 2025, para no romper el anti-fuga del modelo 2024 -> 2025):

- 520000 (estado de flujos de efectivo, método indirecto): señales granulares de generación o consumo de caja (flujo operativo, de inversión y de financiación, y su relación con pasivos y activos).
- 801500 (deterioro de activos): pérdidas por deterioro reconocidas, un indicador directo de activos que están perdiendo valor.
- 801700 (otras provisiones y pasivos contingentes): provisiones y contingencias que anticipan salidas de caja futuras.

De esos tres anexos se derivaron 16 variables nuevas con el mismo blindaje numérico del resto del pipeline (`division_segura`, `.clip()`, `np.log1p` donde aplica, imputación neutral cuando una empresa no reporta el anexo). Con la palanca activa la matriz `X` pasa de 90 a 106 columnas. La cobertura es parcial por naturaleza: no todas las empresas reportan los tres anexos, por eso las variables se imputan de forma neutral y se marca su ausencia, coherente con el diseño de entrada parcial de la sección 10.

### 14.2 Palanca 2: modelo en cascada de dos etapas (`modelo_cascada.py`)

La idea de la cascada es especializar el modelo en la cola de riesgo. En vez de un único ensamble plano, se encadenan dos etapas con un score único calibrado en [0, 1]:

- Etapa 1 (filtro general): el ensamble del PISO puntúa a toda la población y define una cola de riesgo, las empresas por encima del percentil `PERCENTIL_COLA`.
- Etapa 2 (especialista en la cola): un segundo ensamble, entrenado para separar quiebras dentro de esa cola, re-puntúa a las empresas de la cola. Los scores de ambas etapas se combinan de forma monótona (`combinar_scores`) para que el ranking global siga siendo coherente y el número siga siendo una probabilidad en [0, 1].

El hiperparámetro `PERCENTIL_COLA` (qué tan ancha es la cola que hereda la etapa 2) se trató como un parámetro a barrer. Para no elegir ganador a posteriori, el valor se decide mirando SOLO la OOF o el entrenamiento, nunca el Test reservado.

### 14.3 La regla de adopción, declarada ANTES de ver resultados

Siguiendo el mismo rigor de las secciones 9 y 13, la regla se fijó en el encabezado de `evaluacion_zona_roja.py` antes de correr nada. Una variante se adopta por defecto SOLO si cumple los cuatro gates:

- G1 (no regresión OOF): PR-AUC de la variante >= PR-AUC base menos 0.005, y ROC-AUC de la variante >= ROC-AUC base menos 0.002.
- G2 (mejora de la zona roja OOF): los verdaderos positivos en el top 1% (o el top 0.5%) de la variante son ESTRICTAMENTE más que la base, sin caer en el top 2.5%. Este es el objetivo del usuario: detectar más quiebras en el grupo más pequeño.
- G3 (no regresión en el Test reservado): el F1 máximo no cae más de 0.005 por debajo del PISO y los verdaderos positivos del rojo élite (top 1%, 40 alertas) no caen más de 1 respecto al PISO.
- G4 (respaldo estadístico): en el bootstrap pareado de 500 remuestreos, la probabilidad de mejora en PR-AUC, o en recall del top 2.5%, o en precisión del top 1%, es al menos 0.60.

En empate o ambigüedad NO se adopta. El PISO es un suelo intocable. El arnés primero REPRODUCE el PISO (gate de montaje) y solo entonces compara: OOF ROC 0.8867 / PR 0.2534, top 1% 85/200; Test ROC 0.8782 / PR 0.3115, rojo élite 19/40, F1 máximo 0.4144. La reproducción fue exacta, lo que valida que las comparaciones son parejas.

### 14.4 Evaluación honesta con intervalos de confianza

Protocolo idéntico al de las tandas previas: OOF de 5 pliegues (`StratifiedKFold` con `shuffle=True` y `random_state=42`) sobre las 19,937 empresas, Test reservado de 3,988 empresas (61 positivos) con el mismo `train_test_split` estratificado, y bootstrap pareado de 500 remuestreos. Se midieron explícitamente los cortes top 0.5 / 1.0 / 2.5 por ciento y el F1 máximo. Por la restricción autoritativa de tiempo de reloj del usuario, el barrido de `PERCENTIL_COLA` del arnés completo se acotó al valor base 88 (una prueba de humo previa ya había recorrido las colas 80 a 92 sin encontrar una que subiera el top 1% OOF); se conservaron intactos los cinco pliegues OOF, el Test reservado y los 500 remuestreos del bootstrap, que son lo que da rigor a la decisión.

Resultados sobre el top 1% y el F1 máximo (CON vs el PISO):

| Variante | top 1% OOF (TP/200) | ROC OOF | PR OOF | rojo élite Test (TP/40) | F1 máx. Test |
|----------|----------------------|---------|--------|--------------------------|---------------|
| base (PISO) | 85 | 0.8867 | 0.2534 | 19 | 0.4144 |
| anexos | 82 | 0.8859 | 0.2544 | 19 | 0.4038 |
| cascada (cola 88) | 84 | 0.8862 | 0.2527 | 17 | 0.4107 |
| anexos + cascada (cola 88) | 85 | 0.8858 | 0.2496 | 16 | 0.4000 |

Bootstrap pareado contra la base (probabilidad de mejora, un valor por encima de 0.5 indica que la variante tiende a ganar):

- anexos: PR-AUC 0.548, recall top 2.5% 0.610, precisión top 1% 0.126. La subida de PR-AUC es marginal y el intervalo de confianza del 95% cruza el cero.
- cascada: PR-AUC 0.400, recall top 2.5% 0.410, precisión top 1% 0.366. Ningún KPI supera 0.5.
- anexos + cascada: PR-AUC 0.234, recall top 2.5% 0.218, precisión top 1% 0.558. Solo la precisión del top 1% roza la mitad.

### 14.5 Veredicto de los gates y decisión final

| Variante | G1 | G2 | G3 | G4 | Adoptar |
|----------|----|----|----|----|---------|
| anexos | OK | NO | NO | OK | No |
| cascada | OK | NO | NO | NO | No |
| anexos + cascada | OK | NO | NO | NO | No |

Lectura:

- Ninguna variante mejora el objetivo del usuario (G2): anexos baja el top 1% OOF de 85 a 82, la cascada a 84, y la combinación lo iguala en 85 pero sin superarlo estrictamente, de modo que no gana el gate.
- Ninguna variante preserva el PISO en el Test reservado (G3): el F1 máximo cae por debajo del umbral de ruido (anexos 0.4038, combinación 0.4000) y los verdaderos positivos del rojo élite caen (cascada 17, combinación 16) frente a los 19 del PISO.
- El respaldo estadístico (G4) solo se cumple débilmente para anexos, y aun así anexos falla G2 y G3.

Por tanto la DECISIÓN final, coherente con la regla declarada de antemano y no con la intuición, es NO adoptar ninguna variante. `USAR_FEATURES_ANEXOS` y `USAR_CASCADA` quedan en `False` por defecto, el código y las banderas se conservan para reactivarlos si en el futuro cambia la evidencia, y los tres artefactos de producción NO se tocaron (siguen reproduciendo el PISO: Test ROC 0.8782 / PR 0.3115 / F1 0.4144, rojo élite 40 empresas con 19 quiebras, 47.5 por ciento, lift 31.05). La demostración de entrada parcial (`demo_entrada_parcial.py`) sigue corriendo sin cambios.

Interpretación honesta del resultado, consistente con la sección 13: el límite para subir la zona roja no parece ser la falta de filas ni la falta de un modelo más sofisticado, sino la información disponible. Los anexos añadidos tienen cobertura parcial y aportan señal marginal que no se traduce en más quiebras detectadas en la cola; la cascada reorganiza el ranking de la cola sin aumentar su contenido de verdaderos positivos. El PISO publicado sigue siendo la mejor configuración verificada.
