import pandas as pd
import numpy as np

# CARGAR EL df_maestro
df_maestro = pd.read_pickle("./df_maestro_crudo_2024.pkl")
print(f"Dataset cargado: {df_maestro.shape[0]} filas y {df_maestro.shape[1]} columnas.")

# ==========================
# PASO 1: LIMPIEZA ARCHIVO 1 (10000_CARÁTULA)
# ==========================

# 1. REPARAR ERRORES DE DIGITACIÓN EN FECHA DE CONSTITUCIÓN
fechas_texto = df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'].astype(str).str.strip()

correcciones_fechas = {
    r'^119': '199',
    r'^020': '200',
    r'^021': '201',
    r'^0189': '1989',
    r'^0199': '1999',
    r'^0997': '1997',
    r'^0015': '2015',
    r'^0986': '1986'
}

for patron, reemplazo in correcciones_fechas.items():
    fechas_texto = fechas_texto.str.replace(patron, reemplazo, regex=True)

# 2. CONVERTIR A FORMATO FECHA Y FILTRAR AÑOS ILÓGICOS
df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'] = pd.to_datetime(fechas_texto, errors='coerce')
df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'] = df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'].where(
    (df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'].dt.year >= 1900) & 
    (df_maestro['Fecha de constitución (Aaaa-Mm-Dd)'].dt.year <= 2025)
)
df_maestro['Fecha de Corte'] = pd.to_datetime(df_maestro['Fecha de Corte'], errors='coerce')

# 3. CREAR LA VARIABLE 'Edad_Empresa_Anios' E IMPUTAR LOS 7 CASOS RESTANTES CON LA MEDIANA
df_maestro['Edad_Empresa_Anios'] = (df_maestro['Fecha de Corte'] - df_maestro['Fecha de constitución (Aaaa-Mm-Dd)']).dt.days / 365.25
mediana_edad = df_maestro['Edad_Empresa_Anios'].median()
df_maestro['Edad_Empresa_Anios'] = df_maestro['Edad_Empresa_Anios'].fillna(mediana_edad)

# ELIMINAR LAS FECHAS CRUDAS
df_maestro = df_maestro.drop(columns=['Fecha de constitución (Aaaa-Mm-Dd)', 'Fecha de Corte'])

# 4. IMPUTAR NULOS EN LAS VARIABLES CATEGÓRICAS DE LA CARÁTULA
df_maestro['Concepto del Revisor fiscal en su informe'] = df_maestro['Concepto del Revisor fiscal en su informe'].fillna('00. NO APLICA / SIN OBLIGACIÓN')
df_maestro['Estos estados financieros presentan información reexpresada?'] = df_maestro['Estos estados financieros presentan información reexpresada?'].fillna('NO')
df_maestro['Estado actual'] = df_maestro['Estado actual'].fillna('ACTIVA')
df_maestro['Razón social de la sociedad'] = df_maestro['Razón social de la sociedad'].fillna('SIN RAZÓN SOCIAL')
# Agregar '^0986': '1986' en el diccionario correcciones_fechas, y en el punto 4 agregar:
df_maestro['El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior'] = (df_maestro['El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior'].fillna('NO')
)

# VERIFICACIÓN DE NULOS DEL PASO 1
columnas_paso_uno = [
    'Punto de Entrada',
    'NIT',
    'Razón social de la sociedad',
    'Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)',
    'Estado actual',
    'Concepto del Revisor fiscal en su informe',
    'Estos estados financieros presentan información reexpresada?',
    'El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior',
    'Edad_Empresa_Anios'
]

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 1 (CARÁTULA) ---")
print(df_maestro[columnas_paso_uno].isnull().sum())
print(f"Mediana de edad aplicada a los 7 casos sin fecha válida: {mediana_edad:.2f} años")

# ==========================
# PASO 2: LIMPIEZA ARCHIVO 2 (210030_ESTADO DE SITUACIÓN FINANCIERA)
# ==========================

# 0. ELIMINAR EMPRESAS CON FILAS PRINCIPALES VACÍAS EN PERIODO ACTUAL (BALANCE O RESULTADOS EN 2024)
columnas_estructurales_obligatorias = [
    'Total de activos (Assets)',
    'Total pasivos (Liabilities)',
    'Patrimonio total (Equity)',
    'Activos corrientes totales (CurrentAssets)',
    'Efectivo y equivalentes al efectivo (CashAndCashEquivalents)',
    'Ingresos de actividades ordinarias (Revenue)',
    'Ganancia (pérdida) (ProfitLoss)',
    'Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)',
    'Ganancia bruta (GrossProfit)'
]
filas_antes = len(df_maestro)
df_maestro = df_maestro.dropna(subset=columnas_estructurales_obligatorias).reset_index(drop=True)
print(f"Empresas eliminadas por tener Balance o Resultados en blanco en Periodo Actual: {filas_antes - len(df_maestro)}")

# 1. IMPUTAR CON $0 LAS SUBCUENTAS DONDE NaN SIGNIFICA AUSENCIA DE SALDO O DEUDA A CORTO PLAZO
columnas_imputar_cero_archivo_dos = [
    'Pasivos corrientes totales (CurrentLiabilities)',
    'Inventarios corrientes (Inventories)',
    'Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes (TradeAndOtherCurrentReceivables)',
    'Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)',
    'Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)',
    'Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)'
]

df_maestro[columnas_imputar_cero_archivo_dos] = df_maestro[columnas_imputar_cero_archivo_dos].fillna(0)

# 2. VERIFICACIÓN DE NULOS Y DENOMINADORES EN CERO DEL ARCHIVO 2
columnas_paso_dos = [
    'Total de activos (Assets)',
    'Total pasivos (Liabilities)',
    'Patrimonio total (Equity)',
    'Activos corrientes totales (CurrentAssets)',
    'Pasivos corrientes totales (CurrentLiabilities)',
    'Efectivo y equivalentes al efectivo (CashAndCashEquivalents)',
    'Inventarios corrientes (Inventories)',
    'Cuentas comerciales por cobrar y otras cuentas por cobrar corrientes (TradeAndOtherCurrentReceivables)',
    'Activos intangibles distintos de la plusvalía (IntangibleAssetsOtherThanGoodwill)',
    'Pasivos por impuestos corrientes, corriente (CurrentTaxLiabilitiesCurrent)',
    'Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)'
]

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 2 (ARCHIVO 2 - BALANCE) ---")
print(df_maestro[columnas_paso_dos].isnull().sum())
print(f"Empresas con Total de activos == 0 (Liquidadas): {(df_maestro['Total de activos (Assets)'] == 0).sum()}")
print(f"Empresas con Pasivos corrientes == 0 (Sin deuda a corto plazo): {(df_maestro['Pasivos corrientes totales (CurrentLiabilities)'] == 0).sum()}")
print(f"Empresas en Quiebra Patrimonial Técnica (Patrimonio < 0): {(df_maestro['Patrimonio total (Equity)'] < 0).sum()}")

# ==========================
# PASO 3: LIMPIEZA ARCHIVO 3 (310030_ESTADO DE RESULTADOS)
# ==========================

# 1. IMPUTAR CON $0 LAS SUBCUENTAS DE COSTOS, GASTOS E IMPUESTOS DONDE NaN SIGNIFICA $0
columnas_imputar_cero_archivo_tres = [
    'Costo de ventas (CostOfSales)',
    'Costos financieros (FinanceCosts)',
    'Gastos de administración (AdministrativeExpense)',
    'Gastos de ventas (DistributionCosts)',
    'Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)'
]

# Incluir Otras ganancias (pérdidas) y/o Otros ingresos según esté en df_maestro
if 'Otras ganancias (pérdidas) (OtherGainsLosses)' in df_maestro.columns:
    columnas_imputar_cero_archivo_tres.append('Otras ganancias (pérdidas) (OtherGainsLosses)')
if 'Otros ingresos (OtherIncome)' in df_maestro.columns:
    columnas_imputar_cero_archivo_tres.append('Otros ingresos (OtherIncome)')

df_maestro[columnas_imputar_cero_archivo_tres] = df_maestro[columnas_imputar_cero_archivo_tres].fillna(0)

# 2. VERIFICACIÓN DE NULOS Y DENOMINADORES CRÍTICOS DEL ARCHIVO 3
columnas_paso_tres = [
    'Ingresos de actividades ordinarias (Revenue)',
    'Ganancia (pérdida) (ProfitLoss)',
    'Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)',
    'Ganancia bruta (GrossProfit)'
] + columnas_imputar_cero_archivo_tres

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 3 (ARCHIVO 3 - RESULTADOS) ---")
print(df_maestro[columnas_paso_tres].isnull().sum())
print(f"Empresas con Ingresos == 0 (Peligro Margen Bruto): {(df_maestro['Ingresos de actividades ordinarias (Revenue)'] == 0).sum()}")
print(f"Empresas sin Costos Financieros == 0 (Peligro Cobertura Intereses): {(df_maestro['Costos financieros (FinanceCosts)'] == 0).sum()}")
print(f"Empresas con Pérdida Operativa (EBIT < 0): {(df_maestro['Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)'] < 0).sum()}")
print(f"Empresas con Pérdida Neta (ProfitLoss < 0): {(df_maestro['Ganancia (pérdida) (ProfitLoss)'] < 0).sum()}")

# ==========================
# PASO 4: LIMPIEZA ARCHIVO 4 (410030_ESTADO DEL RESULTADO INTEGRAL - ORI)
# ==========================

# 1. IMPUTACIÓN: SI EL RESULTADO INTEGRAL ES NaN (NO TIENE ORI),
# SU VALOR REAL ES EXACTAMENTE IGUAL A LA GANANCIA (PÉRDIDA) NETA DEL ARCHIVO 3
df_maestro['Resultado integral total (ComprehensiveIncome)'] = df_maestro[
    'Resultado integral total (ComprehensiveIncome)'
].fillna(df_maestro['Ganancia (pérdida) (ProfitLoss)'])

# 2. VERIFICACIÓN DE NULOS Y DETECCIÓN DE EMPRESAS ZOMBI POR ORI
columnas_paso_cuatro = [
    'Resultado integral total (ComprehensiveIncome)'
]

zombis_ori_neto = (
    (df_maestro['Ganancia (pérdida) (ProfitLoss)'] < 0) & 
    (df_maestro['Resultado integral total (ComprehensiveIncome)'] > 0)
).sum()

zombis_ori_operativo = (
    (df_maestro['Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)'] < 0) & 
    (df_maestro['Resultado integral total (ComprehensiveIncome)'] > 0)
).sum()

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 4 (ARCHIVO 4 - ORI) ---")
print(df_maestro[columnas_paso_cuatro].isnull().sum())
print(f"Empresas Zombi ORI (Pérdida Neta < 0 pero Resultado Integral > 0): {zombis_ori_neto}")
print(f"Empresas Zombi Operativo (EBIT < 0 pero Resultado Integral > 0): {zombis_ori_operativo}")

# ==========================
# PASO 5: LIMPIEZA ARCHIVO 5 (520000_ESTADO DE FLUJOS DE EFECTIVO)
# ==========================

# 1. IMPUTAR CON $0 LOS NULOS EN LAS 3 ACTIVIDADES DE FLUJO DE EFECTIVO
# (NO SE APLICA .abs() PORQUE LOS FLUJOS NEGATIVOS SON SALIDAS REALES DE DINERO)
columnas_paso_cinco = [
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)'
]

df_maestro[columnas_paso_cinco] = df_maestro[columnas_paso_cinco].fillna(0)

# 2. VERIFICACIÓN DE NULOS Y CASOS CRÍTICOS PARA LOS RATIOS DE FLUJO DE CAJA
flujo_op = df_maestro['Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)']
utilidad_neta = df_maestro['Ganancia (pérdida) (ProfitLoss)']

zombis_caja = ((utilidad_neta > 0) & (flujo_op < 0)).sum()
trampa_doble_negativo = ((utilidad_neta < 0) & (flujo_op < 0)).sum()

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 5 (ARCHIVO 5 - FLUJOS DE EFECTIVO) ---")
print(df_maestro[columnas_paso_cinco].isnull().sum())
print(f"Empresas Zombi de Caja (Utilidad Neta > 0 pero Flujo Operativo < 0): {zombis_caja}")
print(f"Empresas con Doble Negativo (Utilidad < 0 y Flujo Op < 0 - Cuidar en Ratio 1): {trampa_doble_negativo}")

# ==========================
# PASO 6: LIMPIEZA ARCHIVO 6 (610010_ESTADO DE CAMBIOS EN EL PATRIMONIO)
# ==========================

# 1. IMPUTAR CON $0 LAS EMPRESAS QUE NO EMITIERON PATRIMONIO (21,534 NULOS)
# Y CONVERTIR A $0 LOS 6 CASOS NEGATIVOS (REDUCCIONES DE CAPITAL) PARA MEDIR SOLO INYECCIÓN FRESCA
columna_paso_seis = 'Capital emitido [miembro] (CapitalEmitidoMiembro)'

df_maestro[columna_paso_seis] = df_maestro[columna_paso_seis].fillna(0).clip(lower=0)

# 2. VERIFICACIÓN DE NULOS Y EMPRESAS CON INYECCIÓN DE CAPITAL REAL (> 0)
columnas_paso_seis_verificar = [columna_paso_seis]
empresas_con_inyeccion = (df_maestro[columna_paso_seis] > 0).sum()

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 6 (ARCHIVO 6 - PATRIMONIO) ---")
print(df_maestro[columnas_paso_seis_verificar].isnull().sum())
print(f"Empresas con Inyección Real de Capital (> 0): {empresas_con_inyeccion}")
print(f"Empresas con valores negativos restantes (< 0): {(df_maestro[columna_paso_seis] < 0).sum()}")

# ==========================
# PASO 7: LIMPIEZA ARCHIVO 8 (801000_NOTAS - CUENTAS COMERCIALES POR COBRAR)
# ==========================

# 1. IMPUTAR CON $0 LOS NULOS EN SALDO TOTAL Y TRAMOS DE CARTERA VENCIDA
# (YA SE VERIFICÓ QUE HAY 0 VALORES NEGATIVOS Y 0 DESBORDES DONDE VENCIDAS > SALDO TOTAL)
columnas_paso_siete = [
    'Total saldo del ejercicio [miembro] (CxCTotalMiembro)',
    'Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)',
    'Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)'
]

df_maestro[columnas_paso_siete] = df_maestro[columnas_paso_siete].fillna(0)

# 2. VERIFICACIÓN DE NULOS Y MÉTRICAS DE CARTERA TÓXICA
cxc_total = df_maestro['Total saldo del ejercicio [miembro] (CxCTotalMiembro)']
cxc_vencida = df_maestro['Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)']
cxc_mas_360 = df_maestro['Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)']

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 7 (ARCHIVO 8 - CARTERA) ---")
print(df_maestro[columnas_paso_siete].isnull().sum())
print(f"Empresas sin Cuentas por Cobrar (Saldo Total == 0): {(cxc_total == 0).sum()}")
print(f"Empresas con Cartera Vencida (> 0): {(cxc_vencida > 0).sum()}")
print(f"Empresas con Cartera Vencida Crítica > 360 días (> 0): {(cxc_mas_360 > 0).sum()}")
print(f"Desbordes (Cartera Vencida > Saldo Total): {(cxc_vencida > cxc_total).sum()}")

# ==========================
# PASO 8: LIMPIEZA ARCHIVO 14 (801600_NOTAS - CUENTAS COMERCIALES POR PAGAR)
# ==========================

col_cxp_vencidas = 'Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)'
col_cxp_mas_360 = 'Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)'
col_pasivo_corriente = 'Pasivos corrientes totales (CurrentLiabilities)'

# 1. IMPUTAR CON $0 LOS NULOS EN OBLIGACIONES VENCIDAS (EMPRESAS AL DÍA CON PROVEEDORES)
columnas_paso_ocho = [col_cxp_vencidas, col_cxp_mas_360]
df_maestro[columnas_paso_ocho] = df_maestro[columnas_paso_ocho].fillna(0)

# 2. CORREGIR EL ERROR DE LOS "TRES CEROS DE MÁS" (CONTADORES QUE DIGITARON EN PESOS EN VEZ DE MILES)
# Si las obligaciones vencidas superan en más de 50 veces al Pasivo Corriente Total, se divide entre 1000
mascara_tres_ceros_cxp = (
    (df_maestro[col_pasivo_corriente] > 0) & 
    (df_maestro[col_cxp_vencidas] > df_maestro[col_pasivo_corriente] * 50)
)
casos_tres_ceros_cxp = mascara_tres_ceros_cxp.sum()

df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_vencidas] = df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_vencidas] / 1000.0
df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_mas_360] = df_maestro.loc[mascara_tres_ceros_cxp, col_cxp_mas_360] / 1000.0

# 3. APLICAR TOPE LÓGICO CONTABLE: LA DEUDA CORRIENTE VENCIDA NO PUEDE SUPERAR AL PASIVO CORRIENTE TOTAL,
# NI LA MORA > 360 DÍAS PUEDE SUPERAR AL TOTAL DE OBLIGACIONES VENCIDAS
df_maestro[col_cxp_vencidas] = np.minimum(df_maestro[col_cxp_vencidas], df_maestro[col_pasivo_corriente])
df_maestro[col_cxp_mas_360] = np.minimum(df_maestro[col_cxp_mas_360], df_maestro[col_cxp_vencidas])

# 4. VERIFICACIÓN DE NULOS Y MÉTRICAS DE MOROSIDAD CON PROVEEDORES
print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 8 (ARCHIVO 14 - PROVEEDORES) ---")
print(df_maestro[columnas_paso_ocho].isnull().sum())
print(f"Empresas corregidas por error de escala (3 ceros de más): {casos_tres_ceros_cxp}")
print(f"Empresas con Obligaciones Vencidas (> 0): {(df_maestro[col_cxp_vencidas] > 0).sum()}")
print(f"Empresas en Cobro Jurídico / Mora > 360 días (> 0): {(df_maestro[col_cxp_mas_360] > 0).sum()}")
print(f"Desbordes restantes (Vencidas > Pasivo Corriente): {(df_maestro[col_cxp_vencidas] > df_maestro[col_pasivo_corriente]).sum()}")

# ==========================
# PASO 9: LIMPIEZA ARCHIVO 17 (801800b_NOTAS - DESESPERO Y NÓMINA)
# ==========================

col_gan_activos = 'Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)'
col_per_activos = 'Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)'
col_despidos = 'Gastos por beneficios de terminación (TerminationBenefitsExpense)'
col_total_empleados = 'Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)'
col_activos_totales = 'Total de activos (Assets)'

columnas_paso_nueve = [col_gan_activos, col_per_activos, col_despidos, col_total_empleados]

# 1. IMPUTAR CON $0 LOS NULOS EN LAS 4 COLUMNAS DEL ARCHIVO 17
df_maestro[columnas_paso_nueve] = df_maestro[columnas_paso_nueve].fillna(0)

# 2. RECLASIFICAR LAS 4 EMPRESAS CON SIGNO NEGATIVO EN 'GANANCIAS POR DISPOSICIÓN DE ACTIVOS'
# (SON PÉRDIDAS EN VENTA DE ACTIVOS DIGITADAS CON SIGNO MENOS EN LA CASILLA DE GANANCIAS)
mascara_gan_negativa = df_maestro[col_gan_activos] < 0
df_maestro.loc[mascara_gan_negativa, col_per_activos] = (
    df_maestro.loc[mascara_gan_negativa, col_per_activos].abs() + 
    df_maestro.loc[mascara_gan_negativa, col_gan_activos].abs()
)
df_maestro[col_gan_activos] = df_maestro[col_gan_activos].clip(lower=0)
df_maestro[col_per_activos] = df_maestro[col_per_activos].abs()
df_maestro[col_total_empleados] = df_maestro[col_total_empleados].abs()
df_maestro[col_despidos] = df_maestro[col_despidos].abs()

# 3. CORREGIR EL ERROR DE LOS "TRES CEROS DE MÁS" EN VENTA DE ACTIVOS (CONTRA TOTAL DE ACTIVOS)
mascara_tres_ceros_activos = (
    (df_maestro[col_activos_totales] > 0) & 
    (df_maestro[col_gan_activos] > df_maestro[col_activos_totales] * 10)
)
df_maestro.loc[mascara_tres_ceros_activos, col_gan_activos] /= 1000.0
df_maestro[col_gan_activos] = np.minimum(df_maestro[col_gan_activos], df_maestro[col_activos_totales])
df_maestro[col_per_activos] = np.minimum(df_maestro[col_per_activos], df_maestro[col_activos_totales])

# 4. CORREGIR EL ERROR DE LOS "TRES CEROS DE MÁS" EN NÓMINA (CONTRA GASTOS Y COSTOS DEL ARCHIVO 3)
gastos_totales_archivo_tres = (
    df_maestro['Costo de ventas (CostOfSales)'] + 
    df_maestro['Gastos de administración (AdministrativeExpense)'] + 
    df_maestro['Gastos de ventas (DistributionCosts)']
)
mascara_tres_ceros_nomina = (
    (gastos_totales_archivo_tres > 0) & 
    (df_maestro[col_total_empleados] > gastos_totales_archivo_tres * 10)
)
df_maestro.loc[mascara_tres_ceros_nomina, col_total_empleados] /= 1000.0
df_maestro.loc[mascara_tres_ceros_nomina, col_despidos] /= 1000.0

# 5. GARANTIZAR COHERENCIA INTERNA EN NÓMINA (QUE DESPIDOS NUNCA SUPERE AL TOTAL DE EMPLEADOS)
df_maestro[col_total_empleados] = np.maximum(df_maestro[col_total_empleados], df_maestro[col_despidos])

print("\n--- INFORME DE NULOS DESPUÉS DEL PASO 9 (ARCHIVO 17 - DESESPERO) ---")
print(df_maestro[columnas_paso_nueve].isnull().sum())
print(f"Empresas con Ganancia en Venta de Activos (> 0): {(df_maestro[col_gan_activos] > 0).sum()}")
print(f"Empresas con Pérdida en Venta de Activos (> 0): {(df_maestro[col_per_activos] > 0).sum()}")
print(f"Empresas con Pagos por Despidos / Terminación (> 0): {(df_maestro[col_despidos] > 0).sum()}")

# ==========================
# AUDITORÍA FINAL GLOBAL DE LA FASE 3 Y EXPORTACIÓN
# ==========================
total_nulos_dataset = df_maestro.isnull().sum().sum()
print("\n==========================================")
print(f"TOTAL DE NULOS EN TODO EL DF_MAESTRO ({df_maestro.shape[0]} filas x {df_maestro.shape[1]} columnas): {total_nulos_dataset}")
print("==========================================")

assert total_nulos_dataset == 0, "ERROR: Todavía quedan valores nulos en el dataset."

df_maestro.to_pickle("../py/df_maestro_limpio_2024.pkl")
print("Archivo 'df_maestro_limpio_2024.pkl' guardado con éxito.")