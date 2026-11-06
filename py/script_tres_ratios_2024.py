import pandas as pd
import numpy as np

# CARGAR EL DATASET LIMPIO GENERADO EN LA FASE 3
df_maestro = pd.read_pickle("./df_maestro_limpio_2024.pkl")
print(f"Dataset limpio cargado: {df_maestro.shape[0]} filas y {df_maestro.shape[1]} columnas.")

# ==============================================================================
# FASE 4: INGENIERÍA DE CARACTERÍSTICAS (CREACIÓN DE RATIOS Y BANDERAS BINARIAS)
# ==============================================================================

# ==========================
# PASO 1: VARIABLES DEL ARCHIVO 1 (10000_CARÁTULA - GOBERNANZA Y CONFIABILIDAD)
# ==========================

# 1. BANDERA DE REEXPRESIÓN CONTABLE (1 = REHICIERON CONTABILIDAD POR ERRORES/FRAUDE, 0 = NO)
df_maestro['Flag_Reexpresion_Contable'] = np.where(
    df_maestro['Estos estados financieros presentan información reexpresada?'].astype(str).str.strip().str.upper() == 'SI',
    1, 0
)

# 2. BANDERA DE DISTRIBUCIÓN DE UTILIDADES (1 = REPARTIERON DIVIDENDOS / CONFIANZA DE SOCIOS, 0 = NO)
df_maestro['Flag_Distribucion_Utilidades'] = np.where(
    df_maestro['El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior'].astype(str).str.strip().str.upper() == 'SI',
    1, 0
)

# 3. BANDERA DE ALERTA DEL REVISOR FISCAL (1 = CON SALVEDAD, ABSTENCIÓN O DICTAMEN NEGATIVO, 0 = LIMPIO / NO APLICA)
dictamenes_alerta = [
    '02. CON SALVEDAD',
    '01. ABSTENCIÓN DE OPINIÓN',
    '04. NEGATIVO'
]
df_maestro['Flag_Alerta_Revisor_Fiscal'] = np.where(
    df_maestro['Concepto del Revisor fiscal en su informe'].isin(dictamenes_alerta),
    1, 0
)

# 4. VERIFICACIÓN DEL PASO 1 (CARÁTULA)
variables_paso_uno = [
    'Edad_Empresa_Anios',
    'Flag_Reexpresion_Contable',
    'Flag_Distribucion_Utilidades',
    'Flag_Alerta_Revisor_Fiscal'
]

print("\n--- VERIFICACIÓN PASO 1 (VARIABLES DE CARÁTULA) ---")
print("Nulos en variables creadas:")
print(df_maestro[variables_paso_uno].isnull().sum())
print(f"Empresas con Reexpresión Contable (== 1): {(df_maestro['Flag_Reexpresion_Contable'] == 1).sum()}")
print(f"Empresas que Distribuyeron Utilidades (== 1): {(df_maestro['Flag_Distribucion_Utilidades'] == 1).sum()}")
print(f"Empresas con Alerta del Revisor Fiscal (== 1): {(df_maestro['Flag_Alerta_Revisor_Fiscal'] == 1).sum()}")

# ==========================
# PASO 2: RATIOS DEL ARCHIVO 2 (210030_ESTADO DE SITUACIÓN FINANCIERA - LIQUIDEZ Y ENDEUDAMIENTO)
# ==========================

activos_totales = df_maestro['Total de activos (Assets)']
pasivos_totales = df_maestro['Total pasivos (Liabilities)']
patrimonio_total = df_maestro['Patrimonio total (Equity)']
activos_corrientes = df_maestro['Activos corrientes totales (CurrentAssets)']
pasivos_corrientes = df_maestro['Pasivos corrientes totales (CurrentLiabilities)']
efectivo = df_maestro['Efectivo y equivalentes al efectivo (CashAndCashEquivalents)']
inventarios = df_maestro['Inventarios corrientes (Inventories)']
intangibles = df_maestro['Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)']
impuestos_corrientes = df_maestro['Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)']
otros_pasivos_fin_corr = df_maestro['Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)']

# 1. RAZÓN CORRIENTE (LIQUIDEZ GENERAL A CORTO PLAZO - ACOTADA EN [0, 10])
df_maestro['Razon_Corriente'] = np.where(
    pasivos_corrientes > 0,
    activos_corrientes / pasivos_corrientes,
    np.where(activos_corrientes > 0, 10.0, 0.0)
).clip(0.0, 10.0)

# 2. PRUEBA ÁCIDA (LIQUIDEZ SIN INVENTARIOS - ACOTADA EN [0, 10])
df_maestro['Prueba_Acida'] = np.where(
    pasivos_corrientes > 0,
    (activos_corrientes - inventarios) / pasivos_corrientes,
    np.where((activos_corrientes - inventarios) > 0, 10.0, 0.0)
).clip(0.0, 10.0)

# 3. RATIO DE CAJA INMEDIATA / CASH RATIO (EFECTIVO REAL EN BANCO FRENTE A DEUDA CORRIENTE - ACOTADO EN [0, 5])
df_maestro['Ratio_Caja_Inmediata'] = np.where(
    pasivos_corrientes > 0,
    efectivo / pasivos_corrientes,
    np.where(efectivo > 0, 5.0, 0.0)
).clip(0.0, 5.0)

# 4. CAPITAL DE TRABAJO SOBRE ACTIVOS TOTALES (COLCHÓN OPERATIVO NETO - ACOTADO EN [-1, 1])
df_maestro['Capital_Trabajo_Sobre_Activos'] = np.where(
    activos_totales > 0,
    (activos_corrientes - pasivos_corrientes) / activos_totales,
    -1.0
).clip(-1.0, 1.0)

# 5. NIVEL DE ENDEUDAMIENTO TOTAL (APALANCAMIENTO - ACOTADO EN [0, 2])
df_maestro['Nivel_Endeudamiento'] = np.where(
    activos_totales > 0,
    pasivos_totales / activos_totales,
    2.0
).clip(0.0, 2.0)

# 6. BANDERA DE QUIEBRA PATRIMONIAL TÉCNICA (1 SI PATRIMONIO < 0, 0 EN CASO CONTRARIO)
df_maestro['Flag_Quiebra_Patrimonial'] = np.where(patrimonio_total < 0, 1, 0)

# 7. RATIO DE ACTIVOS INTANGIBLES "HUMO" (ILUSIÓN CONTABLE EN ACTIVOS - ACOTADO EN [0, 1])
df_maestro['Ratio_Activos_Intangibles_Humo'] = np.where(
    activos_totales > 0,
    intangibles / activos_totales,
    0.0
).clip(0.0, 1.0)

# 8. RATIO DE FINANCIACIÓN CON LA DIAN (IMPUESTOS CORRIENTES SOBRE ACTIVOS TOTALES - ACOTADO EN [0, 1])
df_maestro['Ratio_Financiacion_DIAN'] = np.where(
    activos_totales > 0,
    impuestos_corrientes / activos_totales,
    0.0
).clip(0.0, 1.0)

# 9. RATIO DE DEUDA BANCARIA DE RESCATE A CORTO PLAZO (SOBRE ACTIVOS TOTALES - ACOTADO EN [0, 1])
df_maestro['Ratio_Deuda_Rescate_Corto_Plazo'] = np.where(
    activos_totales > 0,
    otros_pasivos_fin_corr / activos_totales,
    0.0
).clip(0.0, 1.0)

# 10. VERIFICACIÓN DEL PASO 2 (ARCHIVO 2 - BALANCE)
variables_paso_dos = [
    'Razon_Corriente',
    'Prueba_Acida',
    'Ratio_Caja_Inmediata',
    'Capital_Trabajo_Sobre_Activos',
    'Nivel_Endeudamiento',
    'Flag_Quiebra_Patrimonial',
    'Ratio_Activos_Intangibles_Humo',
    'Ratio_Financiacion_DIAN',
    'Ratio_Deuda_Rescate_Corto_Plazo'
]

print("\n--- VERIFICACIÓN PASO 2 (RATIOS DE BALANCE GENERAL) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_dos].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_dos]).sum().sum())
print(f"Empresas en Iliquidez Severa de Caja (Ratio_Caja_Inmediata < 0.05): {(df_maestro['Ratio_Caja_Inmediata'] < 0.05).sum()}")
print(f"Empresas con Endeudamiento Crítico (> 80%): {(df_maestro['Nivel_Endeudamiento'] > 0.80).sum()}")
print(f"Empresas con Deuda DIAN Crítica (> 8% del Activo): {(df_maestro['Ratio_Financiacion_DIAN'] > 0.08).sum()}")

# ==========================
# PASO 3: RATIOS DEL ARCHIVO 3 (310030_ESTADO DE RESULTADOS - RENTABILIDAD Y EFICIENCIA)
# ==========================

ingresos = df_maestro['Ingresos de actividades ordinarias (Revenue)']
ganancia_bruta = df_maestro['Ganancia bruta (GrossProfit)']
costo_ventas = df_maestro['Costo de ventas (CostOfSales)']
ebit = df_maestro['Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)']
utilidad_neta = df_maestro['Ganancia (pérdida) (ProfitLoss)']
costos_financieros = df_maestro['Costos financieros (FinanceCosts)']
gastos_admin = df_maestro['Gastos de administración (AdministrativeExpense)']
gastos_ventas = df_maestro['Gastos de ventas (DistributionCosts)']
impuestos = df_maestro['Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)']
otros_ingresos = df_maestro['Otros ingresos (OtherIncome)']

# 1. MARGEN BRuto (SI INGRESOS == 0 PERO GANANCIA BRUTA < 0 POR COSTOS SIN VENTAS, CASTIGA CON -1.0)
df_maestro['Margen_Bruto'] = np.where(
    ingresos > 0,
    ganancia_bruta / ingresos,
    np.where(ganancia_bruta < 0, -1.0, 0.0)
).clip(-1.0, 1.0)

# 2. MARGEN OPERACIONAL EBIT (ACOTADO EN [-1, 1])
df_maestro['Margen_Operacional_EBIT'] = np.where(
    ingresos > 0,
    ebit / ingresos,
    np.where(ebit < 0, -1.0, np.where(ebit > 0, 1.0, 0.0))
).clip(-1.0, 1.0)

# 3. MARGEN NETO FINAL (ACOTADO EN [-1, 1])
df_maestro['Margen_Neto'] = np.where(
    ingresos > 0,
    utilidad_neta / ingresos,
    np.where(utilidad_neta < 0, -1.0, np.where(utilidad_neta > 0, 1.0, 0.0))
).clip(-1.0, 1.0)

# 4. ROA OPERATIVO (PRODUCTIVIDAD DEL ACTIVO TOTAL INCLUSO SI INGRESOS == 0 - ACOTADO EN [-1, 1])
df_maestro['ROA_Operativo'] = np.where(
    activos_totales > 0,
    ebit / activos_totales,
    -1.0
).clip(-1.0, 1.0)

# 5. COBERTURA DE INTERESES (SI COSTOS FINANCIEROS == 0: +10 SI EBIT > 0, -5 SI EBIT < 0, 0 SI EBIT == 0)
df_maestro['Cobertura_Intereses'] = np.where(
    costos_financieros > 0,
    ebit / costos_financieros,
    np.where(ebit > 0, 10.0, np.where(ebit < 0, -5.0, 0.0))
).clip(-5.0, 10.0)

# 6. ROTACIÓN DE INVENTARIO (COSTO DE VENTAS / INVENTARIOS - ACOTADO EN [0, 50])
df_maestro['Rotacion_Inventario'] = np.where(
    inventarios > 0,
    costo_ventas / inventarios,
    0.0
).clip(0.0, 50.0)

# 7. PESO DE LA BUROCRACIA ADMINISTRATIVA SOBRE LA GANANCIA BRUTA (ACOTADO EN [0, 2])
df_maestro['Peso_Burocracia_Admin'] = np.where(
    ganancia_bruta > 0,
    gastos_admin / ganancia_bruta,
    np.where(gastos_admin > 0, 2.0, 0.0)
).clip(0.0, 2.0)

# 8. PESO DEL GASTO DE VENTAS SOBRE LA GANANCIA BRUTA (ACOTADO EN [0, 2])
df_maestro['Peso_Gasto_Ventas'] = np.where(
    ganancia_bruta > 0,
    gastos_ventas / ganancia_bruta,
    np.where(gastos_ventas > 0, 2.0, 0.0)
).clip(0.0, 2.0)

# 9. CARGA IMPOSITIVA DIAN (EVITA EL DOBLE NEGATIVO CUANDO EBIT < 0 E IMPUESTOS < 0,
# Y CASTIGA CON 1.0 CUANDO EBIT <= 0 PERO AÚN ASÍ LA DIAN COBRA IMPUESTOS > 0)
df_maestro['Carga_Impositiva_DIAN'] = np.where(
    impuestos <= 0,
    0.0,
    np.where(ebit <= 0, 1.0, impuestos / ebit)
).clip(0.0, 1.0)

# 10. BANDERA DE EMPRESAS QUE CAEN EN NÚMEROS ROJOS POR CULPA DEL IMPUESTO DE LA DIAN
df_maestro['Flag_Rojos_Por_Impuestos'] = np.where(
    (utilidad_neta < 0) & ((utilidad_neta + impuestos) > 0) & (impuestos > 0),
    1, 0
)

# 11. RATIO DE DEPENDENCIA DE OTROS INGRESOS (SIN DIVIDIR ENTRE UTILIDAD NEGATIVA) Y BANDERA (> 30% DE UTILIDAD)
ingresos_totales = ingresos + otros_ingresos
df_maestro['Ratio_Dependencia_Otros_Ingresos'] = np.where(
    ingresos_totales > 0,
    otros_ingresos / ingresos_totales,
    0.0
).clip(0.0, 1.0)

utilidad_neta_segura = np.where(utilidad_neta > 0, utilidad_neta, 1.0)
df_maestro['Flag_Sobrevive_Por_Otros_Ingresos'] = np.where(
    ((utilidad_neta > 0) & ((otros_ingresos / utilidad_neta_segura) > 0.30)) | ((ebit <= 0) & (otros_ingresos > 0)),
    1, 0
)

# 12. BANDERA DE EMPRESA SIN INGRESOS OPERACIONALES (INGRESOS == 0)
df_maestro['Flag_Sin_Ingresos_Operacionales'] = np.where(ingresos == 0, 1, 0)

# 13. VERIFICACIÓN DEL PASO 3 (ARCHIVO 3 - RESULTADOS)
variables_paso_tres = [
    'Margen_Bruto',
    'Margen_Operacional_EBIT',
    'Margen_Neto',
    'ROA_Operativo',
    'Cobertura_Intereses',
    'Rotacion_Inventario',
    'Peso_Burocracia_Admin',
    'Peso_Gasto_Ventas',
    'Carga_Impositiva_DIAN',
    'Flag_Rojos_Por_Impuestos',
    'Ratio_Dependencia_Otros_Ingresos',
    'Flag_Sobrevive_Por_Otros_Ingresos',
    'Flag_Sin_Ingresos_Operacionales'
]

print("\n--- VERIFICACIÓN PASO 3 (RATIOS DE ESTADO DE RESULTADOS) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_tres].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_tres]).sum().sum())
print(f"Empresas con Cobertura de Intereses Crítica (< 1.0): {(df_maestro['Cobertura_Intereses'] < 1.0).sum()}")
print(f"Empresas con Burocracia Excesiva (> 80% de Ganancia Bruta): {(df_maestro['Peso_Burocracia_Admin'] > 0.80).sum()}")
print(f"Empresas en Números Rojos por golpe de Impuestos: {(df_maestro['Flag_Rojos_Por_Impuestos'] == 1).sum()}")

# ==========================
# PASO 4: RATIOS DEL ARCHIVO 4 (410030_ESTADO DEL RESULTADO INTEGRAL - ORI Y EMPRESAS ZOMBI)
# ==========================

resultado_integral = df_maestro['Resultado integral total (ComprehensiveIncome)']

# VALOR PURO DEL ORI (DIFERENCIA ENTRE EL RESULTADO INTEGRAL Y LA UTILIDAD NETA REAL)
ori_puro = resultado_integral - utilidad_neta

# 1. PROPORCIÓN DEL ORI SOBRE ACTIVOS TOTALES (PESO DE LA GANANCIA O PÉRDIDA FANTASMA - ACOTADO EN [-1, 1])
df_maestro['Proporcion_ORI_Sobre_Activos'] = np.where(
    activos_totales > 0,
    ori_puro / activos_totales,
    0.0
).clip(-1.0, 1.0)

# 2. BANDERA ZOMBI ORI NETO (1 SI PERDIÓ DINERO REAL < 0 PERO EL ORI LA HIZO VER POSITIVA > 0)
df_maestro['Flag_Zombi_ORI_Neto'] = np.where(
    (utilidad_neta < 0) & (resultado_integral > 0),
    1, 0
)

# 3. BANDERA DE MAQUILLAJE PATRIMONIAL CON ORI (1 SI TIENE PÉRDIDA OPERATIVA O NETA < 0
# Y AL MISMO TIEMPO REGISTRA VALORIZACIONES POSITIVAS EN EL ORI > 0 PARA AMORTIGUAR EL GOLPE)
df_maestro['Flag_Maquillaje_Patrimonial_ORI'] = np.where(
    ((ebit < 0) | (utilidad_neta < 0)) & (ori_puro > 0),
    1, 0
)

# 4. VERIFICACIÓN DEL PASO 4 (ARCHIVO 4 - ORI)
variables_paso_cuatro = [
    'Proporcion_ORI_Sobre_Activos',
    'Flag_Zombi_ORI_Neto',
    'Flag_Maquillaje_Patrimonial_ORI'
]

print("\n--- VERIFICACIÓN PASO 4 (RATIOS DE ORI Y EMPRESAS ZOMBI) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_cuatro].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_cuatro]).sum().sum())
print(f"Empresas Zombi ORI Neto (Voltearon pérdida neta a positivo): {(df_maestro['Flag_Zombi_ORI_Neto'] == 1).sum()}")
print(f"Empresas con Maquillaje Patrimonial ORI (Pérdida + ORI positivo): {(df_maestro['Flag_Maquillaje_Patrimonial_ORI'] == 1).sum()}")

# ==========================
# PASO 5: RATIOS DEL ARCHIVO 5 (520000_ESTADO DE FLUJOS DE EFECTIVO - VERDAD BANCARIA)
# ==========================

flujo_operativo = df_maestro['Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)']
flujo_inversion = df_maestro['Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)']
flujo_financiacion = df_maestro['Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)']

# 1. RATIO DE CALIDAD DE LAS GANANCIAS (PROTEGIDO CONTRA EL DOBLE NEGATIVO EN LAS 2,620 EMPRESAS:
# SI UTILIDAD NETA <= 0 Y FLUJO OPERATIVO < 0, RECIBE CASTIGO DIRECTO DE -1.0 EN VEZ DE DAR POSITIVO)
df_maestro['Ratio_Calidad_Ganancias'] = np.where(
    utilidad_neta > 0,
    flujo_operativo / utilidad_neta,
    np.where(flujo_operativo < 0, -1.0, np.where(flujo_operativo > 0, 0.5, 0.0))
).clip(-1.0, 2.0)

# 2. BRECHA DE CAJA VS UTILIDAD SOBRE ACTIVOS TOTALES (ESTANDARIZADA SIN CAMBIO DE SIGNO EN DENOMINADOR)
df_maestro['Brecha_Caja_vs_Utilidad'] = np.where(
    activos_totales > 0,
    (flujo_operativo - utilidad_neta) / activos_totales,
    0.0
).clip(-1.0, 1.0)

# 3. COBERTURA DE EFECTIVO INMEDIATA (FLUJO OPERATIVO / PASIVOS CORRIENTES - ACOTADA EN [-1, 2])
df_maestro['Ratio_Cobertura_Efectivo_Inmediata'] = np.where(
    pasivos_corrientes > 0,
    flujo_operativo / pasivos_corrientes,
    np.where(flujo_operativo > 0, 1.0, np.where(flujo_operativo < 0, -1.0, 0.0))
).clip(-1.0, 2.0)

# 4. BANDERA ZOMBI DE CAJA (1 SI REPORTA UTILIDAD NETA > 0 EN PAPEL PERO QUEMA EFECTIVO OPERATIVO < 0)
df_maestro['Flag_Zombi_Caja'] = np.where(
    (utilidad_neta > 0) & (flujo_operativo < 0),
    1, 0
)

# 5. BANDERA DE DOBLE SANGRÍA (1 SI PIERDE EN PAPEL <= 0 Y ADEMÁS QUEMA EFECTIVO OPERATIVO < 0)
df_maestro['Flag_Doble_Sangria_Caja'] = np.where(
    (utilidad_neta <= 0) & (flujo_operativo < 0),
    1, 0
)

# 6. RATIOS DE FLUJO DE INVERSIÓN Y FINANCIACIÓN SOBRE ACTIVOS TOTALES (ACOTADOS EN [-1, 1])
df_maestro['Ratio_Flujo_Inversion_Activos'] = np.where(
    activos_totales > 0,
    flujo_inversion / activos_totales,
    0.0
).clip(-1.0, 1.0)

df_maestro['Ratio_Flujo_Financiacion_Activos'] = np.where(
    activos_totales > 0,
    flujo_financiacion / activos_totales,
    0.0
).clip(-1.0, 1.0)

# 7. VERIFICACIÓN DEL PASO 5 (ARCHIVO 5 - FLUJOS DE EFECTIVO)
variables_paso_cinco = [
    'Ratio_Calidad_Ganancias',
    'Brecha_Caja_vs_Utilidad',
    'Ratio_Cobertura_Efectivo_Inmediata',
    'Flag_Zombi_Caja',
    'Flag_Doble_Sangria_Caja',
    'Ratio_Flujo_Inversion_Activos',
    'Ratio_Flujo_Financiacion_Activos'
]

print("\n--- VERIFICACIÓN PASO 5 (RATIOS DE FLUJOS DE EFECTIVO) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_cinco].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_cinco]).sum().sum())
print(f"Empresas respirando artificialmente (Cobertura Efectivo < 0.20): {(df_maestro['Ratio_Cobertura_Efectivo_Inmediata'] < 0.20).sum()}")
print(f"Empresas Zombi de Caja (Utilidad > 0 pero Flujo Op < 0): {(df_maestro['Flag_Zombi_Caja'] == 1).sum()}")
print(f"Empresas en Doble Sangría (Utilidad <= 0 y Flujo Op < 0): {(df_maestro['Flag_Doble_Sangria_Caja'] == 1).sum()}")

# ==========================
# PASO 6: RATIOS DEL ARCHIVO 6 (610010_ESTADO DE CAMBIOS EN EL PATRIMONIO - SALVAVIDAS DE SOCIOS)
# ==========================

capital_emitido = df_maestro['Capital emitido [miembro] (CapitalEmitidoMiembro)']

# 1. PROPORCIÓN DE INYECCIÓN DE CAPITAL FRENTE AL TAMAÑO DE LA EMPRESA (ACOTADO EN [0, 1])
df_maestro['Ratio_Inyeccion_Capital_Activos'] = np.where(
    activos_totales > 0,
    capital_emitido / activos_totales,
    0.0
).clip(0.0, 1.0)

# 2. BANDERA DE INYECCIÓN DE CAPITAL DE LOS SOCIOS (1 SI CAPITAL EMITIDO > 0, 0 EN CASO CONTRARIO)
df_maestro['Flag_Inyeccion_Capital_Socios'] = np.where(capital_emitido > 0, 1, 0)

# 3. BANDERA DE RESCATE PATRIMONIAL EN NÚMEROS ROJOS (HIPÓTESIS RESUMEN.DOCX:
# 1 SI LA EMPRESA TIENE PÉRDIDA NETA < 0, RAZÓN CORRIENTE < 1.0 O ENDEUDAMIENTO > 0.70,
# PERO LOS DUEÑOS INYECTARON CAPITAL FRESCO > 0 PARA ESTABILIZARLA)
en_numeros_rojos = (
    (utilidad_neta < 0) | 
    (df_maestro['Razon_Corriente'] < 1.0) | 
    (df_maestro['Nivel_Endeudamiento'] > 0.70)
)
df_maestro['Flag_Rescate_Socios_En_Rojos'] = np.where(
    en_numeros_rojos & (capital_emitido > 0),
    1, 0
)

# 4. VERIFICACIÓN DEL PASO 6 (ARCHIVO 6 - PATRIMONIO)
variables_paso_seis = [
    'Ratio_Inyeccion_Capital_Activos',
    'Flag_Inyeccion_Capital_Socios',
    'Flag_Rescate_Socios_En_Rojos'
]

print("\n--- VERIFICACIÓN PASO 6 (RATIOS DE INYECCIÓN DE CAPITAL) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_seis].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_seis]).sum().sum())
print(f"Empresas con Inyección de Capital de Socios (== 1): {(df_maestro['Flag_Inyeccion_Capital_Socios'] == 1).sum()}")
print(f"Empresas en Números Rojos Rescatadas por Socios (== 1): {(df_maestro['Flag_Rescate_Socios_En_Rojos'] == 1).sum()}")

# ==========================
# PASO 7: RATIOS DEL ARCHIVO 8 (801000_NOTAS - CUENTAS COMERCIALES POR COBRAR / CARTERA TÓXICA)
# ==========================

cxc_saldo_total = df_maestro['Total saldo del ejercicio [miembro] (CxCTotalMiembro)']
cxc_vencidas = df_maestro['Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)']
cxc_mas_360 = df_maestro['Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)']

# 1. RATIO DE CARTERA TÓXICA (PROPORCIÓN DE CUENTAS POR COBRAR VENCIDAS - ACOTADO EN [0, 1])
df_maestro['Ratio_Cartera_Toxica'] = np.where(
    cxc_saldo_total > 0,
    cxc_vencidas / cxc_saldo_total,
    0.0
).clip(0.0, 1.0)

# 2. RATIO DE CARTERA PERDIDA > 360 DÍAS (FACTURAS CON MÁS DE 1 AÑO DE MORA - ACOTADO EN [0, 1])
df_maestro['Ratio_Cartera_Perdida_360d'] = np.where(
    cxc_saldo_total > 0,
    cxc_mas_360 / cxc_saldo_total,
    0.0
).clip(0.0, 1.0)

# 3. BANDERA DE CARTERA INCOBRABLE > 360 DÍAS (1 SI TIENE FACTURAS VENCIDAS DE MÁS DE UN AÑO)
df_maestro['Flag_Cartera_Perdida_360d'] = np.where(cxc_mas_360 > 0, 1, 0)

# 4. BANDERA DE FALSA LIQUIDEZ / ESPEJISMO DE CARTERA (HIPÓTESIS RESUMEN.DOCX:
# 1 SI LA RAZÓN CORRIENTE CLÁSICA PARECE SANA >= 1.0, PERO TIENE CARTERA TÓXICA > 30%
# O AL RESTAR LA CARTERA VENCIDA DE SUS ACTIVOS CORRIENTES CAE EN ILIQUIDEZ < 1.0)
activos_corrientes_sin_mora = np.maximum(0.0, activos_corrientes - cxc_vencidas)
razon_corriente_depurada = np.where(
    pasivos_corrientes > 0,
    activos_corrientes_sin_mora / pasivos_corrientes,
    np.where(activos_corrientes_sin_mora > 0, 10.0, 0.0)
)

df_maestro['Flag_Falsa_Liquidez_Cartera'] = np.where(
    (df_maestro['Razon_Corriente'] >= 1.0) & 
    ((df_maestro['Ratio_Cartera_Toxica'] > 0.30) | (razon_corriente_depurada < 1.0)),
    1, 0
)

# 5. ELIMINAR LA COLUMNA REDUNDANTE 'Total saldo del ejercicio [miembro] (CxCTotalMiembro)'
# (COMO SE DEFINIÓ EN CONSIDERACIONES, EL SALDO DE CXC CORRIENTES YA ESTÁ EN EL ARCHIVO 2)
df_maestro = df_maestro.drop(columns=['Total saldo del ejercicio [miembro] (CxCTotalMiembro)'])

# 6. VERIFICACIÓN DEL PASO 7 (ARCHIVO 8 - CARTERA)
variables_paso_siete = [
    'Ratio_Cartera_Toxica',
    'Ratio_Cartera_Perdida_360d',
    'Flag_Cartera_Perdida_360d',
    'Flag_Falsa_Liquidez_Cartera'
]

print("\n--- VERIFICACIÓN PASO 7 (RATIOS DE CARTERA TÓXICA) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_siete].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_siete]).sum().sum())
print(f"Empresas con Cartera Tóxica Crítica (> 30%): {(df_maestro['Ratio_Cartera_Toxica'] > 0.30).sum()}")
print(f"Empresas con Cartera Perdida > 360 días (== 1): {(df_maestro['Flag_Cartera_Perdida_360d'] == 1).sum()}")
print(f"Empresas con Falsa Liquidez / Espejismo de Cartera (== 1): {(df_maestro['Flag_Falsa_Liquidez_Cartera'] == 1).sum()}")

# ==========================
# PASO 8: RATIOS DEL ARCHIVO 14 (801600_NOTAS - CUENTAS COMERCIALES POR PAGAR / MOROSIDAD SEVERA)
# ==========================

cxp_vencidas = df_maestro['Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)']
cxp_mas_360 = df_maestro['Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)']

# 1. RATIO DE MOROSIDAD SEVERA (OBLIGACIONES VENCIDAS / PASIVOS CORRIENTES - ACOTADO EN [0, 1])
df_maestro['Ratio_Morosidad_Severa'] = np.where(
    pasivos_corrientes > 0,
    cxp_vencidas / pasivos_corrientes,
    0.0
).clip(0.0, 1.0)

# 2. RATIO DE DEUDA VENCIDA > 360 DÍAS SOBRE PASIVOS CORRIENTES (ACOTADO EN [0, 1])
df_maestro['Ratio_Deuda_Vencida_Mas_360d'] = np.where(
    pasivos_corrientes > 0,
    cxp_mas_360 / pasivos_corrientes,
    0.0
).clip(0.0, 1.0)

# 3. BANDERA DE COBRO JURÍDICO CON PROVEEDORES (PENALIZADOR DE CONFIABILIDAD: MORA > 360 DÍAS > 0)
df_maestro['Flag_Cobro_Juridico_Proveedores'] = np.where(cxp_mas_360 > 0, 1, 0)

# 4. BANDERA DE DEFAULT SIN EFECTIVO (1 SI LAS OBLIGACIONES YA VENCIDAS SUPERAN TODO EL EFECTIVO EN BANCO)
df_maestro['Flag_Default_Sin_Efectivo'] = np.where(
    (cxp_vencidas > 0) & (cxp_vencidas > efectivo),
    1, 0
)

# 5. VERIFICACIÓN DEL PASO 8 (ARCHIVO 14 - PROVEEDORES)
variables_paso_ocho = [
    'Ratio_Morosidad_Severa',
    'Ratio_Deuda_Vencida_Mas_360d',
    'Flag_Cobro_Juridico_Proveedores',
    'Flag_Default_Sin_Efectivo'
]

print("\n--- VERIFICACIÓN PASO 8 (RATIOS DE MOROSIDAD CON PROVEEDORES) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_ocho].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_ocho]).sum().sum())
print(f"Empresas en Default Comercial (Morosidad Severa > 30%): {(df_maestro['Ratio_Morosidad_Severa'] > 0.30).sum()}")
print(f"Empresas en Cobro Jurídico / Mora > 360 días (== 1): {(df_maestro['Flag_Cobro_Juridico_Proveedores'] == 1).sum()}")
print(f"Empresas en Default sin Efectivo para cubrir la mora (== 1): {(df_maestro['Flag_Default_Sin_Efectivo'] == 1).sum()}")

# ==========================
# PASO 9: RATIOS DEL ARCHIVO 17 (801800b_NOTAS - DESESPERO GERENCIAL Y NÓMINA)
# ==========================

ganancias_venta_activos = df_maestro['Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)']
perdidas_venta_activos = df_maestro['Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)']
gastos_despidos = df_maestro['Gastos por beneficios de terminación (TerminationBenefitsExpense)']
total_gastos_empleados = df_maestro['Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)']

# 1. RATIO DE DESPIDOS / TERMINACIÓN DE CONTRATOS SOBRE EL GASTO TOTAL DE NÓMINA (ACOTADO EN [0, 1])
df_maestro['Ratio_Despidos_Terminacion'] = np.where(
    total_gastos_empleados > 0,
    gastos_despidos / total_gastos_empleados,
    0.0
).clip(0.0, 1.0)

# 2. RATIO DE SUPERVIVENCIA POR VENTA DE ACTIVOS FIJOS SOBRE LA UTILIDAD NETA
# (SE CALCULA SOLO SI UTILIDAD NETA > 0 PARA EVITAR DIVIDIR ENTRE PÉRDIDAS NEGATIVAS - ACOTADO EN [0, 1])
df_maestro['Ratio_Supervivencia_Venta_Activos'] = np.where(
    utilidad_neta > 0,
    ganancias_venta_activos / utilidad_neta,
    0.0
).clip(0.0, 1.0)

# 3. RATIO DE LIQUIDACIÓN DE ACTIVOS FIJOS SOBRE ACTIVOS TOTALES (CAPTURA TANTO GANANCIAS COMO PÉRDIDAS EN VENTA)
df_maestro['Ratio_Liquidacion_Activos_Fijos'] = np.where(
    activos_totales > 0,
    (ganancias_venta_activos + perdidas_venta_activos) / activos_totales,
    0.0
).clip(0.0, 1.0)

# 4. RATIO DE PESO DE LA NÓMINA SOBRE INGRESOS OPERACIONALES (ACOTADO EN [0, 1])
df_maestro['Ratio_Peso_Nomina_Ingresos'] = np.where(
    ingresos > 0,
    total_gastos_empleados / ingresos,
    np.where(total_gastos_empleados > 0, 1.0, 0.0)
).clip(0.0, 1.0)

# 5. BANDERA COMPUESTA DE DESESPERO GERENCIAL (1 SI VENDIÓ ACTIVOS FIJOS O PAGÓ INDEMNIZACIONES POR DESPIDOS)
df_maestro['Flag_Desespero_Gerencial'] = np.where(
    (ganancias_venta_activos > 0) | (perdidas_venta_activos > 0) | (gastos_despidos > 0),
    1, 0
)

# 6. VERIFICACIÓN DEL PASO 9 (ARCHIVO 17 - DESESPERO Y NÓMINA)
variables_paso_nueve = [
    'Ratio_Despidos_Terminacion',
    'Ratio_Supervivencia_Venta_Activos',
    'Ratio_Liquidacion_Activos_Fijos',
    'Ratio_Peso_Nomina_Ingresos',
    'Flag_Desespero_Gerencial'
]

print("\n--- VERIFICACIÓN PASO 9 (RATIOS DE DESESPERO GERENCIAL Y NÓMINA) ---")
print("Nulos en ratios creados:")
print(df_maestro[variables_paso_nueve].isnull().sum())
print("Infinitos en ratios creados:", np.isinf(df_maestro[variables_paso_nueve]).sum().sum())
print(f"Empresas con Señal de Desespero Gerencial (== 1): {(df_maestro['Flag_Desespero_Gerencial'] == 1).sum()}")
print(f"Empresas que sobreviven vendiendo activos (> 30% de su Utilidad Neta): {(df_maestro['Ratio_Supervivencia_Venta_Activos'] > 0.30).sum()}")

# ==============================================================================
# AUDITORÍA FINAL GLOBAL DE LA FASE 4 Y EXPORTACIÓN PARA EL MODELO DE IA
# ==============================================================================

todas_las_variables_modelo = (
    variables_paso_uno +
    variables_paso_dos +
    variables_paso_tres +
    variables_paso_cuatro +
    variables_paso_cinco +
    variables_paso_seis +
    variables_paso_siete +
    variables_paso_ocho +
    variables_paso_nueve
)

nulos_totales_fase4 = df_maestro[todas_las_variables_modelo].isnull().sum().sum()
infinitos_totales_fase4 = np.isinf(df_maestro[todas_las_variables_modelo]).sum().sum()

print("\n==========================================")
print(f"RESUMEN FINAL FASE 4 (INGENIERÍA DE CARACTERÍSTICAS):")
print(f" - Filas (Empresas únicas): {df_maestro.shape[0]}")
print(f" - Columnas totales en df_maestro: {df_maestro.shape[1]}")
print(f" - Total de Ratios y Banderas Forenses listos para ML: {len(todas_las_variables_modelo)}")
print(f" - Total NULOS en los {len(todas_las_variables_modelo)} indicadores: {nulos_totales_fase4}")
print(f" - Total INFINITOS en los {len(todas_las_variables_modelo)} indicadores: {infinitos_totales_fase4}")
print("==========================================")

assert nulos_totales_fase4 == 0, "ERROR: Hay valores nulos en los ratios creados."
assert infinitos_totales_fase4 == 0, "ERROR: Hay valores infinitos en los ratios creados."

df_maestro.to_pickle("./df_maestro_con_ratios_2024.pkl")
print("Archivo 'df_maestro_con_ratios_2024.pkl' guardado con éxito.")