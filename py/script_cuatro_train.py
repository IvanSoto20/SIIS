import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score
)

# ==============================================================================
# PASO 1: CARGA DE DATASETS (2024 Y 2025) Y EXTRACCIÓN DE TENDENCIAS 2023
# ==============================================================================
df_2024 = pd.read_pickle("./df_maestro_con_ratios_2024.pkl")
df_2025 = pd.read_pickle("./df_maestro_con_ratios.pkl")
print(f"Datasets cargados -> 2024: {df_2024.shape} | 2025: {df_2025.shape}")

# Extraer 'Periodo Anterior' (2023) directamente de los archivos 2 y 3 de 2024 para medir deterioro 2023 -> 2024
print("Extrayendo historia 2023 ('Periodo Anterior') de los archivos 210030 y 310030 de 2024...")
cols_b23 = [
    "NIT", "Periodo", "Fecha de Corte",
    "Total de activos (Assets)", "Total pasivos (Liabilities)", "Patrimonio total (Equity)",
    "Activos corrientes totales (CurrentAssets)", "Pasivos corrientes totales (CurrentLiabilities)",
    "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)"
]
df_b23 = pd.read_excel(
    "../SIIS/pymes_individuales-2024/210030_Estado de situación financiera, corriente_no corriente.xlsx",
    usecols=cols_b23
)
df_b23["NIT"] = df_b23["NIT"].astype(str).str.strip().str.replace(".0", "", regex=False)
df_b23 = (
    df_b23[df_b23["Periodo"] == "Periodo Anterior"]
    .sort_values(by=["NIT", "Fecha de Corte"])
    .drop_duplicates(subset=["NIT"], keep="last")
    .drop(columns=["Periodo", "Fecha de Corte"])
)

cols_r23 = [
    "NIT", "Periodo", "Fecha de Corte",
    "Ingresos de actividades ordinarias (Revenue)",
    "Ganancia (pérdida) (ProfitLoss)",
    "Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)"
]
df_r23 = pd.read_excel(
    "../SIIS/pymes_individuales-2024/310030_Estado de resultado integral, resultado del periodo, por funcion de gasto.xlsx",
    usecols=cols_r23
)
df_r23["NIT"] = df_r23["NIT"].astype(str).str.strip().str.replace(".0", "", regex=False)
df_r23 = (
    df_r23[df_r23["Periodo"] == "Periodo Anterior"]
    .sort_values(by=["NIT", "Fecha de Corte"])
    .drop_duplicates(subset=["NIT"], keep="last")
    .drop(columns=["Periodo", "Fecha de Corte"])
)

df_2023 = pd.merge(df_b23, df_r23, on="NIT", how="outer").add_suffix("_2023")
df_2023 = df_2023.rename(columns={"NIT_2023": "NIT"})

# Filtrar población sana en 2024 y unir con 2025 y con historia 2023
mascara_sanas_2024 = (
    (df_2024["Estado actual"].isin(["ACTIVA", "EN ETAPA PREOPERATIVA"])) &
    (df_2024["Flag_Quiebra_Patrimonial"] == 0)
)
df_2024_sanas = df_2024[mascara_sanas_2024].copy()

df_maestro = pd.merge(
    df_2024_sanas,
    df_2025,
    on="NIT",
    how="inner",
    suffixes=("_2024", "_2025")
)
df_maestro = pd.merge(df_maestro, df_2023, on="NIT", how="left")
print(f"Total empresas sanas en 2024 con seguimiento 2023->2024->2025: {len(df_maestro)}")

# ==============================================================================
# PASO 2: INGENIERÍA DE CARACTERÍSTICAS (ESTÁTICAS 2024 + TENDENCIAS 2023->2024)
# ==============================================================================
columnas_base_2024 = (
    list(df_maestro.loc[:, "Flag_Reexpresion_Contable_2024":"Flag_Desespero_Gerencial_2024"].columns) +
    ["Edad_Empresa_Anios_2024"]
)
columnas_base_2024 = [c for c in columnas_base_2024 if c != "Flag_Quiebra_Patrimonial_2024"]

X = df_maestro[columnas_base_2024].copy()

# Saldos limpios de 2024
activos_24 = df_maestro["Total de activos (Assets)_2024"].clip(lower=0)
pasivos_24 = df_maestro["Total pasivos (Liabilities)_2024"].clip(lower=0)
patrimonio_24 = df_maestro["Patrimonio total (Equity)_2024"].clip(lower=0)
act_corr_24 = df_maestro["Activos corrientes totales (CurrentAssets)_2024"].clip(lower=0)
pas_corr_24 = df_maestro["Pasivos corrientes totales (CurrentLiabilities)_2024"].clip(lower=0)
efectivo_24 = df_maestro["Efectivo y equivalentes al efectivo (CashAndCashEquivalents)_2024"].clip(lower=0)
ingresos_24 = df_maestro["Ingresos de actividades ordinarias (Revenue)_2024"].clip(lower=0)
utilidad_24 = df_maestro["Ganancia (pérdida) (ProfitLoss)_2024"]
ebit_24 = df_maestro["Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)_2024"]
flujo_op_24 = df_maestro["Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)_2024"]
costos_fin_24 = df_maestro["Costos financieros (FinanceCosts)_2024"].clip(lower=0)

# --- A. VARIABLES DE AGOTAMIENTO PATRIMONIAL Y ESCALA EN 2024 ---
perdida_neta_24 = np.maximum(0.0, -utilidad_24)
X["Tasa_Consumo_Patrimonio_2024"] = np.where(
    patrimonio_24 > 0, perdida_neta_24 / patrimonio_24, np.where(perdida_neta_24 > 0, 5.0, 0.0)
).clip(0.0, 5.0)

perdida_ebit_24 = np.maximum(0.0, -ebit_24)
X["Tasa_Sangria_EBIT_Patrimonio_2024"] = np.where(
    patrimonio_24 > 0, perdida_ebit_24 / patrimonio_24, np.where(perdida_ebit_24 > 0, 5.0, 0.0)
).clip(0.0, 5.0)

patrimonio_proyectado_inercial = patrimonio_24 + utilidad_24
X["Ratio_Patrimonio_Inercial_Sobre_Activos_2024"] = np.where(
    activos_24 > 0, patrimonio_proyectado_inercial / activos_24, -1.0
).clip(-1.0, 1.0)
X["Flag_Colapso_Patrimonial_Inercial_2024"] = np.where(patrimonio_proyectado_inercial <= 0, 1, 0)

X["ROA_Neto_2024"] = np.where(activos_24 > 0, utilidad_24 / activos_24, -1.0).clip(-1.0, 1.0)
X["ROE_2024"] = np.where(patrimonio_24 > 0, utilidad_24 / patrimonio_24, -2.0).clip(-2.0, 2.0)

X["Colchon_Patrimonio_Sobre_Pasivos_2024"] = np.where(pasivos_24 > 0, patrimonio_24 / pasivos_24, 5.0).clip(0.0, 5.0)
X["Impacto_Intereses_Sobre_Patrimonio_2024"] = np.where(
    patrimonio_24 > 0, costos_fin_24 / patrimonio_24, np.where(costos_fin_24 > 0, 2.0, 0.0)
).clip(0.0, 2.0)

quema_caja_op_24 = np.maximum(0.0, -flujo_op_24)
X["Tasa_Agotamiento_Efectivo_2024"] = np.where(
    efectivo_24 > 0, quema_caja_op_24 / efectivo_24, np.where(quema_caja_op_24 > 0, 5.0, 0.0)
).clip(0.0, 5.0)

deficit_corriente_24 = np.maximum(0.0, pas_corr_24 - act_corr_24)
X["Deficit_Corriente_Sobre_Patrimonio_2024"] = np.where(
    patrimonio_24 > 0, deficit_corriente_24 / patrimonio_24, np.where(deficit_corriente_24 > 0, 5.0, 0.0)
).clip(0.0, 5.0)

X["Log_Activos_2024"] = np.log1p(activos_24)
X["Log_Ingresos_2024"] = np.log1p(ingresos_24)
X["Log_Patrimonio_2024"] = np.log1p(patrimonio_24)
X["Rotacion_Activos_Totales_2024"] = np.where(activos_24 > 0, ingresos_24 / activos_24, 0.0).clip(0.0, 10.0)
X["Concentracion_Deuda_Corto_Plazo_2024"] = np.where(pasivos_24 > 0, pas_corr_24 / pasivos_24, 0.0).clip(0.0, 1.0)
X["Flag_NIIF_Plenas_2024"] = np.where(
    df_maestro["Punto de Entrada_2024"].astype(str).str.upper().str.contains("PLENAS"), 1, 0
)

# --- B. NUEVAS VARIABLES DE VELOCIDAD DE DETERIORO Y TENDENCIA (2023 -> 2024) ---
activos_23 = df_maestro["Total de activos (Assets)_2023"].fillna(activos_24).clip(lower=0)
pasivos_23 = df_maestro["Total pasivos (Liabilities)_2023"].fillna(pasivos_24).clip(lower=0)
patrimonio_23 = df_maestro["Patrimonio total (Equity)_2023"].fillna(patrimonio_24)
act_corr_23 = df_maestro["Activos corrientes totales (CurrentAssets)_2023"].fillna(act_corr_24).clip(lower=0)
pas_corr_23 = df_maestro["Pasivos corrientes totales (CurrentLiabilities)_2023"].fillna(pas_corr_24).clip(lower=0)
efectivo_23 = df_maestro["Efectivo y equivalentes al efectivo (CashAndCashEquivalents)_2023"].fillna(efectivo_24).clip(lower=0)
ingresos_23 = df_maestro["Ingresos de actividades ordinarias (Revenue)_2023"].fillna(ingresos_24).clip(lower=0)
utilidad_23 = df_maestro["Ganancia (pérdida) (ProfitLoss)_2023"].fillna(utilidad_24)
ebit_23 = df_maestro["Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)_2023"].fillna(ebit_24)

endeudamiento_23 = np.where(activos_23 > 0, pasivos_23 / activos_23, X["Nivel_Endeudamiento_2024"]).clip(0.0, 2.0)
roa_neto_23 = np.where(activos_23 > 0, utilidad_23 / activos_23, X["ROA_Neto_2024"]).clip(-1.0, 1.0)
razon_corr_23 = np.where(pas_corr_23 > 0, act_corr_23 / pas_corr_23, X["Razon_Corriente_2024"]).clip(0.0, 10.0)

# 1. Salto en Nivel de Endeudamiento de 2023 a 2024 (> 0 significa que se endeudó más en 2024)
X["Delta_Endeudamiento_23_24"] = (X["Nivel_Endeudamiento_2024"] - endeudamiento_23).clip(-1.0, 1.0)

# 2. Variación Porcentual del Patrimonio de 2023 a 2024 (< 0 significa destrucción patrimonial en el último año)
X["Var_Pct_Patrimonio_23_24"] = np.where(
    patrimonio_23 > 0,
    (patrimonio_24 - patrimonio_23) / patrimonio_23,
    0.0
).clip(-1.0, 2.0)

# 3. Variación Porcentual de los Activos de 2023 a 2024 (< 0 significa contracción / desinversión)
X["Var_Pct_Activos_23_24"] = np.where(
    activos_23 > 0,
    (activos_24 - activos_23) / activos_23,
    0.0
).clip(-1.0, 2.0)

# 4. Variación Porcentual de los Ingresos de 2023 a 2024 (< 0 significa caída de ventas)
X["Var_Pct_Ingresos_23_24"] = np.where(
    ingresos_23 > 0,
    (ingresos_24 - ingresos_23) / ingresos_23,
    np.where(ingresos_24 > 0, 1.0, 0.0)
).clip(-1.0, 2.0)

# 5. Variación Porcentual del Efectivo en Caja de 2023 a 2024
X["Var_Pct_Efectivo_23_24"] = np.where(
    efectivo_23 > 0,
    (efectivo_24 - efectivo_23) / efectivo_23,
    0.0
).clip(-1.0, 2.0)

# 6. Deterioro de Rentabilidad (Delta ROA Neto) y Deterioro de Liquidez (Delta Razón Corriente)
X["Delta_ROA_Neto_23_24"] = (X["ROA_Neto_2024"] - roa_neto_23).clip(-1.0, 1.0)
X["Delta_Razon_Corriente_23_24"] = (X["Razon_Corriente_2024"] - razon_corr_23).clip(-5.0, 5.0)

# 7. Banderas de Historia Crítica (2 años seguidos perdiendo dinero o colapso de ventas > 30%)
X["Flag_Dos_Anios_Perdida_23_24"] = np.where((utilidad_23 < 0) & (utilidad_24 < 0), 1, 0)
X["Flag_Dos_Anios_EBIT_Negativo_23_24"] = np.where((ebit_23 < 0) & (ebit_24 < 0), 1, 0)
X["Flag_Desplome_Ventas_23_24"] = np.where(X["Var_Pct_Ingresos_23_24"] < -0.30, 1, 0)
X["Flag_Destruccion_Patrimonio_Severa_23_24"] = np.where(X["Var_Pct_Patrimonio_23_24"] < -0.30, 1, 0)

# --- C. ÍNDICES SINTÉTICOS Y NORMALIZACIÓN SECTORIAL CIIU ---
X["Altman_Z_Emergente_2024"] = (
    1.2 * X["Capital_Trabajo_Sobre_Activos_2024"]
    + 3.3 * X["ROA_Operativo_2024"]
    + 1.0 * X["ROA_Neto_2024"]
    + 0.6 * X["Colchon_Patrimonio_Sobre_Pasivos_2024"]
    + 1.0 * X["Rotacion_Activos_Totales_2024"]
)

X["Score_Alerta_Temprana_2024"] = (
    X["Nivel_Endeudamiento_2024"] * 2.5
    + X["Tasa_Consumo_Patrimonio_2024"] * 2.0
    + X["Delta_Endeudamiento_23_24"] * 2.0
    - X["Var_Pct_Patrimonio_23_24"] * 1.5
    + X["Deficit_Corriente_Sobre_Patrimonio_2024"] * 1.2
    - X["Capital_Trabajo_Sobre_Activos_2024"]
    - X["ROA_Neto_2024"] * 2.0
    + X["Ratio_Morosidad_Severa_2024"]
    + X["Ratio_Financiacion_DIAN_2024"]
)

X["Indice_Asfixia_Operativa_2024"] = (
    X["Nivel_Endeudamiento_2024"] * (1.0 - X["Razon_Corriente_2024"].clip(0.0, 2.0) / 2.0)
)

columnas_flags_2024 = [c for c in X.columns if c.startswith("Flag_")]
X["Total_Banderas_Rojas_2024"] = X[columnas_flags_2024].sum(axis=1)

ciiu_div = df_maestro["Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)_2024"].astype(str).str[:2]
for col_rel in [
    "Nivel_Endeudamiento_2024",
    "ROA_Neto_2024",
    "Razon_Corriente_2024",
    "Tasa_Consumo_Patrimonio_2024",
    "Altman_Z_Emergente_2024",
    "Var_Pct_Patrimonio_23_24",
    "Delta_Endeudamiento_23_24"
]:
    media_sector = X[col_rel].groupby(ciiu_div).transform("mean")
    X[f"{col_rel}_vs_Sector"] = X[col_rel] - media_sector

assert X.isnull().sum().sum() == 0, "ERROR: Hay nulos en X."
print(f"Matriz X con Historia 2023->2024 lista: {X.shape[0]} filas x {X.shape[1]} características.")

# ==============================================================================
# PASO 3: DEFINICIÓN DEL VECTOR OBJETIVO (y) EN 2025 Y PARTICIÓN TRAIN / TEST
# ==============================================================================
colapso_legal_2025 = ~df_maestro["Estado actual_2025"].isin(["ACTIVA", "EN ETAPA PREOPERATIVA"])
colapso_patrimonial_2025 = df_maestro["Flag_Quiebra_Patrimonial_2025"] == 1
colapso_cesacion_pagos_2025 = (
    (df_maestro["Flag_Cobro_Juridico_Proveedores_2024"] == 0) &
    (df_maestro["Flag_Cobro_Juridico_Proveedores_2025"] == 1) &
    (df_maestro["Flag_Default_Sin_Efectivo_2025"] == 1) &
    (df_maestro["Flag_Doble_Sangria_Caja_2025"] == 1)
)

y_estricto = (colapso_legal_2025 | colapso_patrimonial_2025).astype(int)
y_integral = (colapso_legal_2025 | colapso_patrimonial_2025 | colapso_cesacion_pagos_2025).astype(int)

y = y_estricto.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==============================================================================
# PASO 4: ESCUADRÓN ÉLITE DE 5 MODELOS (CENTRADO EN HISTGRADIENTBOOSTING Y RF)
# ==============================================================================
print("\nEntrenando Escuadrón Élite de 5 Modelos con Tendencias 2023->2024...")

# 1. HistGradientBoosting Controlado (1:8) - El campeón en PR-AUC de la corrida anterior
hgb_1 = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.03,
    max_depth=4,
    min_samples_leaf=25,
    l2_regularization=5.0,
    class_weight={0: 1, 1: 8},
    random_state=42
)

# 2. HistGradientBoosting Profundo (1:12) con mayor regularización L2 para interacciones 2023->2024
hgb_2 = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.025,
    max_depth=6,
    min_samples_leaf=30,
    l2_regularization=8.0,
    class_weight={0: 1, 1: 12},
    random_state=100
)

# 3. Random Forest Peso Controlado (1:10)
rf_1 = RandomForestClassifier(
    n_estimators=500,
    max_depth=9,
    min_samples_leaf=8,
    max_features="sqrt",
    class_weight={0: 1, 1: 10},
    random_state=42,
    n_jobs=-1
)

# 4. Random Forest Balanced Subsample
rf_2 = RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_leaf=16,
    class_weight="balanced_subsample",
    random_state=100,
    n_jobs=-1
)

# 5. GradientBoosting Estocástico Puro (Probabilidades bien calibradas en el extremo superior)
gb_puro = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.035,
    max_depth=3,
    subsample=0.80,
    min_samples_leaf=15,
    random_state=42
)

hgb_1.fit(X_train, y_train)
hgb_2.fit(X_train, y_train)
rf_1.fit(X_train, y_train)
rf_2.fit(X_train, y_train)
gb_puro.fit(X_train, y_train)

p1 = hgb_1.predict_proba(X_test)[:, 1]
p2 = hgb_2.predict_proba(X_test)[:, 1]
p3 = rf_1.predict_proba(X_test)[:, 1]
p4 = rf_2.predict_proba(X_test)[:, 1]
p5 = gb_puro.predict_proba(X_test)[:, 1]

# Ensamble Probabilístico Élite (50% HistGradientBoosting + 30% RF + 20% GB Puro)
prob_ensamble = 0.30 * p1 + 0.20 * p2 + 0.20 * p3 + 0.10 * p4 + 0.20 * p5

# Ensamble por Ranking Percentil
rank_ensamble = (
    0.30 * rankdata(p1) +
    0.20 * rankdata(p2) +
    0.20 * rankdata(p3) +
    0.10 * rankdata(p4) +
    0.20 * rankdata(p5)
) / len(y_test)

# ==============================================================================
# PASO 5: AUDITORÍA DE MÉTRICAS CON TENDENCIAS 2023->2024
# ==============================================================================
print("\n======================================================================")
print("COMPARATIVA DE PODER DISCRIMINATORIO CON HISTORIA 2023->2024 (TEST)")
print("======================================================================")
for nombre, probs in [
    ("1. HistGradientBoosting Controlado (1:8)", p1),
    ("2. HistGradientBoosting Profundo (1:12)", p2),
    ("3. RF Peso Controlado (1:10)", p3),
    ("4. RF Balanced Subsample", p4),
    ("5. GradientBoosting Estocástico Puro", p5),
    ("6. SUPER-ENSAMBLE PROBABILÍSTICO ÉLITE", prob_ensamble),
    ("7. SUPER-ENSAMBLE RANKING (Percentil)", rank_ensamble)
]:
    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    print(f"{nombre:45s} -> ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

# Usamos prob_ensamble para tener probabilidades reales interpretables (0% a 100%)
score_final = prob_ensamble

precisions, recalls, thresholds = precision_recall_curve(y_test, score_final)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
idx_mejor = np.argmax(f1_scores)
umbral_optimo_f1 = thresholds[idx_mejor]

print("\n======================================================================")
print("DIAGNÓSTICO POR PUNTOS DE OPERACIÓN (SOBRE X_TEST = 3,988 EMPRESAS)")
print("======================================================================")
for nombre_u, u in [
    (f"1. Umbral Óptimo Máximo F1 (Prob >= {umbral_optimo_f1:.3f})", umbral_optimo_f1),
    ("2. Corte Quirúrgico Alta Precisión (Top 1.2% más riesgoso)", np.percentile(score_final, 98.8)),
    ("3. Corte Equilibrado (Top 2.5% más riesgoso)", np.percentile(score_final, 97.5)),
    ("4. Corte Cobertura Media (Top 5.0% más riesgoso)", np.percentile(score_final, 95.0)),
    ("5. Corte Auditoría Preventiva (Top 10.0% más riesgoso)", np.percentile(score_final, 90.0))
]:
    preds_u = (score_final >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_u).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_test, preds_u)
    lift = prec / y_test.mean()
    print(f"\n---> {nombre_u}:")
    print(f"     Quebradas detectadas (TP): {tp} de {tp+fn} ({rec*100:.1f}% Recall)")
    print(f"     Falsas alarmas (FP): {fp} de {tn+fp} sanas | Precisión: {prec*100:.1f}% | F1-Score: {f1:.4f} | Lift: {lift:.1f}x")

# ==============================================================================
# PASO 6: VALIDACIÓN DEL SEMÁFORO PREDICTIVO DE 3 ZONAS EN TEST (2024 -> 2025)
# ==============================================================================
print("\n======================================================================")
print("RENDIMIENTO DEL SEMÁFORO PREDICTIVO DE 3 ZONAS EN TEST (2024 -> 2025)")
print("======================================================================")
corte_amarillo = np.percentile(score_final, 90.0)
corte_rojo = umbral_optimo_f1

zona_semaforo = np.where(
    score_final >= corte_rojo, "3. ROJO (Peligro Inminente)",
    np.where(score_final >= corte_amarillo, "2. AMARILLO (Vigilancia / Alerta)", "1. VERDE (Estable / Sano)")
)

df_eval_semaforo = pd.DataFrame({"Zona_Semaforo_2024": zona_semaforo, "Quebro_en_2025": y_test.values})
resumen_semaforo = df_eval_semaforo.groupby("Zona_Semaforo_2024").agg(
    Total_Empresas=("Quebro_en_2025", "count"),
    Empresas_Quebradas_2025=("Quebro_en_2025", "sum"),
    Tasa_Real_Quiebra_Pct=("Quebro_en_2025", lambda s: s.mean() * 100)
)
resumen_semaforo["Pct_Del_Total_Quebradas"] = (
    resumen_semaforo["Empresas_Quebradas_2025"] / y_test.sum() * 100
)
resumen_semaforo["Lift_vs_Promedio"] = resumen_semaforo["Tasa_Real_Quiebra_Pct"] / (y_test.mean() * 100)
print(resumen_semaforo.to_string())

# Top 20 Variables más importantes (incluyendo Tendencias 2023->2024)
importancias_final = pd.Series(rf_1.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n======================================================================")
print("TOP 20 INDICADORES (ESTÁTICOS 2024 + TENDENCIAS 2023->2024):")
print("======================================================================")
print(importancias_final.head(20).to_string())