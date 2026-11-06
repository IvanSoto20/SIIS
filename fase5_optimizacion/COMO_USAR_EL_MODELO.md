# Cómo usar el modelo de predicción de insolvencia (Fase 5, SIIS Colombia)

Este documento explica, paso a paso y en lenguaje claro, cómo usar el modelo ya
entrenado para puntuar una empresa, cómo leer lo que devuelve y qué tan bueno es en
la práctica. Está pensado tanto para una persona técnica que va a ejecutar el código
como para una persona no técnica que necesita interpretar el resultado.

El modelo responde a una pregunta concreta: dada la foto financiera de una empresa en
un año (por ejemplo 2025), ¿qué tan probable es que esa empresa entre en insolvencia
al año siguiente (2026)? El resultado es una probabilidad entre 0 y 1 y una "zona de
semáforo" (VERDE, AMARILLO, NARANJA o ROJO ÉLITE).

Todos los comandos deben ejecutarse con el intérprete del entorno virtual del
proyecto:

```
/projects/sandbox/.venv_fase5/bin/python
```

No use el `python` del sistema: las librerías (pandas, scikit-learn, lightgbm,
xgboost) están instaladas solo en ese entorno.

---

## 1. Qué necesita el modelo como entrada

El modelo de producción ya está entrenado y guardado en:

```
fase5_optimizacion/salidas/modelo_final_2026.joblib
```

Ese archivo contiene cuatro cosas: los cuatro modelos del ensamble, la lista exacta
de columnas (`features`) que el modelo espera, el calibrador que convierte el puntaje
crudo en una probabilidad honesta, y los umbrales que separan las zonas del semáforo.

Para puntuar una empresa el modelo reconstruye internamente sus 90 variables a partir
de los saldos contables del corte. Las cuentas que necesita son las habituales de los
anexos 210030 (estado de situación financiera) y 310030 (estado de resultados):

- Totales del balance: activos, pasivos, patrimonio, activo corriente, pasivo
  corriente, efectivo.
- Núcleo del estado de resultados: ingresos, utilidad (resultado del periodo), EBIT
  (resultado operativo), costos financieros, flujo operativo.
- Cuentas de detalle que afinan la señal: obligaciones vencidas, pasivo con la DIAN,
  gastos de administración y de ventas, entre otras.
- Identificación: NIT, razón social y código CIIU (para el reporte, no para el
  cálculo).

### Qué pasa si faltan datos (entrada parcial)

El requisito central del usuario es que una empresa con datos incompletos igual
reciba una predicción. El modelo está diseñado para eso:

- Las cuentas de detalle que falten se imputan de forma neutral (se tratan como cero o
  como variación nula). No se inventa deterioro ni mejora.
- Si la empresa no tiene historia de varios años, se activan banderas de cobertura
  (por ejemplo `Flag_Sin_Historia_3A`) y las tendencias plurianuales toman valor
  neutro.
- El resultado es que la matriz de entrada nunca queda con huecos, y el modelo
  siempre devuelve una probabilidad válida en el intervalo [0, 1] y su zona de
  semáforo, sin caerse.

Hay un esqueleto contable mínimo recomendable: conviene aportar los totales
estructurales del balance (activos, pasivos, patrimonio y los corrientes) y el núcleo
del estado de resultados (ingresos y utilidad). Entre más completa la foto, más
confiable la predicción; pero aún con datos parciales el modelo entrega un número.

---

## 2. Cómo puntuar una empresa (comando exacto)

La forma más directa y segura de probar el modelo con datos parciales es el script de
demostración ya incluido, que usa EXACTAMENTE la misma ruta de inferencia de
producción (misma construcción de variables, mismas columnas, mismo ensamble, mismo
calibrador y mismos umbrales):

```
cd /projects/sandbox/SIIS/fase5_optimizacion
/projects/sandbox/.venv_fase5/bin/python demo_entrada_parcial.py
```

Ese comando toma empresas reales, las "mutila" de forma controlada para simular datos
faltantes y falta de historia, y confirma que cada una recibe su probabilidad y su
zona. El resultado detallado queda en:

```
fase5_optimizacion/salidas/demo_entrada_parcial.json
```

Para puntuar la población completa del siguiente año (el semáforo 2026 sobre todas las
empresas) se ejecuta el pipeline de producción:

```
cd /projects/sandbox/SIIS/fase5_optimizacion
/projects/sandbox/.venv_fase5/bin/python script_cinco_optimizado.py
```

Al terminar imprime `Proceso completado con éxito` y regenera:

- `salidas/semaforo_2026.csv`: una fila por empresa con su NIT, razón social, CIIU,
  puntaje, probabilidad calibrada, zona de semáforo y detonantes críticos.
- `salidas/reporte_fase5.json`: las métricas de efectividad del modelo.
- `salidas/modelo_final_2026.joblib`: el modelo entrenado.

### Para puntuar UNA empresa nueva con sus propios datos

Use `demo_entrada_parcial.py` como plantilla. Internamente hace tres pasos que usted
puede replicar para su propia fila:

1. Carga el modelo: `modelo = joblib.load("salidas/modelo_final_2026.joblib")`.
2. Construye la matriz de variables con la función de producción
   `construir_matriz_caracteristicas(...)` a partir de un DataFrame con las columnas
   crudas de la empresa, y selecciona `modelo["features"]` (las 90 columnas exactas).
3. Mezcla las probabilidades de los cuatro modelos con los pesos del ensamble, aplica
   el calibrador `modelo["calibrador"]` y asigna la zona con `modelo["umbrales"]`.

La clave es no inventar una ruta paralela: reutilice las funciones del pipeline para
que su empresa pase por el mismo camino que la población de producción. Las columnas
que no tenga, déjelas en blanco (NaN): la ruta las imputa de forma neutral.

---

## 3. Cómo interpretar el output

Para cada empresa el modelo entrega dos números y una etiqueta.

### Score_Ensamble (puntaje crudo)

Es el promedio ponderado de las probabilidades que dan los cuatro modelos del ensamble
(30% LightGBM, 30% XGBoost, 20% HistGradientBoosting, 20% RandomForest). Sirve para
ORDENAR a las empresas de más a menos riesgosa. Es un número relativo: úselo para
rankear, no como "probabilidad real".

### Probabilidad_Calibrada_Quiebra (probabilidad honesta)

Es el `Score_Ensamble` pasado por un calibrador isotónico. La calibración corrige el
puntaje crudo para que se parezca a una probabilidad real: si el modelo asigna 0.40 a
un grupo de empresas, aproximadamente el 40% de ellas deberían quebrar. Este es el
número que debe reportar e interpretar como "probabilidad de insolvencia al año
siguiente". Siempre está entre 0 y 1.

### Zona del semáforo

El semáforo agrupa a las empresas en cuatro zonas según su puntaje, de menor a mayor
peligro. Las tasas de quiebra y los lift que siguen son los OBSERVADOS en el conjunto
de prueba reservado (3,988 empresas, 61 quiebras reales), es decir, empresas que el
modelo nunca vio al entrenarse. La tasa base del panel es de 1.52% (cerca de 1 de cada
66 empresas quiebra).

| Zona | Qué significa | Tasa de quiebra observada | Lift (veces el promedio) |
|------|----------------|---------------------------|---------------------------|
| VERDE (Estable / Sano) | Riesgo muy bajo, operación normal | 0.64% | 0.42x (por debajo del promedio) |
| AMARILLO (Vigilancia / Alerta) | Riesgo moderado, conviene monitorear | 3.68% | 2.41x |
| NARANJA (Riesgo Alto) | Señales claras de deterioro | 13.33% | 8.72x |
| ROJO ÉLITE (Peligro Inminente) | Máximo riesgo, revisión prioritaria | 47.50% | 31.05x |

Lectura rápida: una empresa en ROJO ÉLITE tiene una probabilidad de quebrar al año
siguiente unas 31 veces mayor que una empresa promedio. Una en VERDE está claramente
por debajo del promedio.

---

## 4. Qué tan bueno es el modelo (efectividad en lenguaje natural)

Esta sección traduce las métricas técnicas a frases sencillas. Todos los números son
reales, tomados del reporte vigente (`salidas/reporte_fase5.json`), medidos sobre el
conjunto de prueba reservado que el modelo no vio al entrenarse.

### La idea central: el modelo concentra el peligro en un grupo pequeño

De cada 100 empresas sanas, solo alrededor de 1 o 2 quiebran al año siguiente. El
valor del modelo es que no reparte ese riesgo por igual: lo concentra en un grupo
pequeño que marca como peligroso.

- De las 40 empresas que el modelo marca en ROJO ÉLITE (el 1% más riesgoso), 19
  quiebran de verdad al año siguiente. Es decir, casi la mitad (47.5%). Eso es 31
  veces más que el promedio general.
- De las 20 empresas más riesgosas de todas (el 0.5% superior), 12 quiebran: 6 de cada
  10. Es el corte más quirúrgico, 39 veces el promedio.
- Si se amplía la alerta a 100 empresas (el 2.5% superior), el modelo captura 27 de las
  61 quiebras reales, es decir cerca del 44% de todas las quiebras del año mirando solo
  al 2.5% de la población.

En palabras llanas: si usted solo pudiera visitar o auditar un puñado de empresas, el
modelo le dice a cuáles ir primero, y acierta muchísimo más que elegir al azar.

### ROC-AUC de 0.88: el modelo ordena bien

El ROC-AUC del modelo es 0.8782. Interprételo así: si se toma al azar una empresa que
sí quebró y una que no, hay un 88% de probabilidad de que el modelo le haya dado mayor
puntaje de riesgo a la que efectivamente quebró. Un 0.5 sería lanzar una moneda y un
1.0 sería perfecto. 0.88 es un modelo que ordena muy bien el riesgo.

### PR-AUC de 0.31: por qué un número aparentemente bajo es en realidad fuerte

El PR-AUC (área bajo la curva de precisión y recall) es 0.3115. A primera vista parece
bajo, pero hay que leerlo en contexto de desbalance. Solo el 1.52% de las empresas
quiebra, una relación cercana a 65 a 1 entre sanas y quebradas. En ese escenario, un
modelo que adivinara al azar tendría un PR-AUC cercano a 0.015 (la tasa base). Un
PR-AUC de 0.31 es unas 20 veces mejor que el azar. Cuando las quiebras son tan raras,
este tipo de valor es un resultado fuerte, no débil.

### Resumen para alguien no técnico

- El modelo no predice el futuro con certeza; estima riesgo y ordena empresas.
- Marca a un grupo pequeño como de altísimo riesgo y, dentro de ese grupo, casi la
  mitad efectivamente quiebra.
- Acierta mucho más que el azar, tanto al ordenar (ROC 0.88) como al concentrar
  quiebras raras en pocas alertas (PR 0.31, unas 20 veces el azar).
- La probabilidad calibrada se puede leer directamente como "de cada 100 empresas con
  esta probabilidad, aproximadamente tantas quiebran".

---

## 5. Resumen de archivos

| Archivo | Para qué sirve |
|---------|-----------------|
| `salidas/modelo_final_2026.joblib` | El modelo entrenado (ensamble + calibrador + umbrales + lista de features). |
| `demo_entrada_parcial.py` | Demostración de puntuación con datos parciales por la ruta real. |
| `salidas/demo_entrada_parcial.json` | Resultado de la demo (entrada, probabilidad y zona por caso). |
| `script_cinco_optimizado.py` | Pipeline completo: reentrena y regenera el semáforo 2026. |
| `salidas/semaforo_2026.csv` | Una fila por empresa con probabilidad, zona y detonantes. |
| `salidas/reporte_fase5.json` | Métricas de efectividad (ROC, PR, cortes y semáforo). |
