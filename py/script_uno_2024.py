import pandas as pd

# ============================
# FUNCIONES GENERALES
# ============================
def limpiar_nit(empresa):
    empresa['NIT'] = empresa['NIT'].astype(str).str.strip().str.replace('.0', '', regex=False)

def verificar_duplicados(df, nombre_archivo=""):
    duplicados = df[df['NIT'].duplicated(keep=False)]
    if not duplicados.empty:
        print(f"\nALERTA: Se encontraron NITs duplicados en [{nombre_archivo}]:")
        print(duplicados['NIT'].value_counts())
        raise ValueError(f"Ejecución detenida: [{nombre_archivo}] tiene empresas clonadas.")

def verificar_sinNIT(df, nombre_archivo=""):
    assert df['NIT'].isnull().sum() == 0, f"ERROR CRÍTICO. [{nombre_archivo}] trae filas sin NIT."



# ==========================
# ARCHIVO 1: 10000_CARÁTULA
#===========================

columnas_caratula = [
    'Punto de Entrada',
    'NIT',
    'Razón social de la sociedad',
    'Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)',
    'Fecha de constitución (Aaaa-Mm-Dd)',
    'Estado actual', 
    'Fecha de Corte',
    'Concepto del Revisor fiscal en su informe',
    'Estos estados financieros presentan información reexpresada?',
    'El Máximo Órgano Social aprobó distribuir utilidades del ejercicio inmediatamente anterior'
]

df_caratula = pd.read_excel("../SIIS/pymes_individuales-2024/10000_Carátula.xlsx", usecols=columnas_caratula)
limpiar_nit(df_caratula)

df_caratula = df_caratula.sort_values(by=['NIT', 'Fecha de Corte'])
df_caratula = df_caratula.drop_duplicates(subset=['NIT'], keep='last')

verificar_duplicados(df_caratula, "Archivo 1 - Carátula")

verificar_sinNIT(df_caratula, "Archivo 1 - Carátula")

print(f"Empresas cargadas: {len(df_caratula)}")
print(df_caratula.head())

# ==========================
# MERGE ARCHIVO 2:  210030_Estado de situación financiera, corriente_no corriente
#===========================

columnas_archivo_dos = [
    'NIT',
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
    'Otros pasivos financieros corrientes (OtherCurrentFinancialLiabilities)',
    'Periodo',
    'Fecha de Corte'
]

df_archivo_dos = pd.read_excel("../SIIS/pymes_individuales-2024/210030_Estado de situación financiera, corriente_no corriente.xlsx", usecols=columnas_archivo_dos)
limpiar_nit(df_archivo_dos)

# ESCUDO PRE-MERGE

# TOMAR SOLO REGISTROS MÁS RECIENTES Y FECHA DE CORTE MÁS RECIENTES
df_archivo_dos = df_archivo_dos[df_archivo_dos['Periodo'] == 'Periodo Actual'] # TOMAR SOLO REGISTROS ACTUALES
df_archivo_dos = df_archivo_dos.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_dos = df_archivo_dos.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_dos = df_archivo_dos.drop(columns=['Periodo', 'Fecha de Corte'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_dos_actualizar = [
    'NIT',
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

#VERIFICAR DUPLICADOS
duplicados = df_archivo_dos[df_archivo_dos['NIT'].duplicated(keep=False)]
if not duplicados.empty:
    print("Se encontraron NITs duplicados en el Archivo 2:")
    print(duplicados['NIT'].value_counts())
    raise ValueError("Ejecución detenida: El archivo 2 tiene empresas clonadas.")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
assert df_archivo_dos['NIT'].isnull().sum() == 0, "ERROR. El archivo 2 trae filas sin NIT."

# MERGE

df_maestro = pd.merge(df_caratula, df_archivo_dos, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 2) ---")
print(df_maestro[columnas_archivo_dos_actualizar].isnull().sum())

del df_archivo_dos

# ==========================
# MERGE ARCHIVO 3: 310030_Estado de resultado integral, resultado del periodo, por funcion de gasto
#===========================

columnas_archivo_tres = [
    'NIT',
    'Ingresos de actividades ordinarias (Revenue)',
    'Ganancia (pérdida) (ProfitLoss)',
    'Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)',
    'Costo de ventas (CostOfSales)',
    'Costos financieros (FinanceCosts)',
    'Otros ingresos (OtherIncome)',
    'Ganancia bruta (GrossProfit)',
    'Gastos de administración (AdministrativeExpense)',
    'Gastos de ventas (DistributionCosts)',
    'Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)',
    'Periodo',
    'Fecha de Corte'
]

df_archivo_tres = pd.read_excel("../SIIS/pymes_individuales-2024/310030_Estado de resultado integral, resultado del periodo, por funcion de gasto.xlsx", usecols=columnas_archivo_tres)
limpiar_nit(df_archivo_tres)

# ESCUDO PRE-MERGE

# TOMAR SOLO REGISTROS MÁS RECIENTES Y FECHA DE CORTE MÁS RECIENTES
df_archivo_tres = df_archivo_tres[df_archivo_tres['Periodo'] == 'Periodo Actual'] # TOMAR SOLO REGISTROS ACTUALES
df_archivo_tres = df_archivo_tres.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_tres = df_archivo_tres.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_tres = df_archivo_tres.drop(columns=['Periodo', 'Fecha de Corte'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_tres_actualizar = [
    'NIT',
    'Ingresos de actividades ordinarias (Revenue)',
    'Ganancia (pérdida) (ProfitLoss)',
    'Ganancia (pérdida) por actividades de operación (GananciaPerdidaPorActividadesDeOperacion)',
    'Costo de ventas (CostOfSales)',
    'Costos financieros (FinanceCosts)',
    'Otros ingresos (OtherIncome)',
    'Ganancia bruta (GrossProfit)',
    'Gastos de administración (AdministrativeExpense)',
    'Gastos de ventas (DistributionCosts)',
    'Ingreso (gasto) por impuestos (IncomeTaxExpenseContinuingOperations)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_tres, "Archivo 3")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_tres, "Archivo 3")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_tres, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 3) ---")
print(df_maestro[columnas_archivo_tres_actualizar].isnull().sum())

del df_archivo_tres

# ==========================
# MERGE ARCHIVO 4: 410030_Estado del resultado integral, componentes ORI presentados netos de impuestos
#===========================

columnas_archivo_cuatro = [
    'NIT',
    'Resultado integral total (ComprehensiveIncome)',
    'Periodo',
    'Fecha de Corte'
]

df_archivo_cuatro = pd.read_excel("../SIIS/pymes_individuales-2024/410030_Estado del resultado integral, componentes ORI presentados netos de impuestos.xlsx", usecols=columnas_archivo_cuatro)
limpiar_nit(df_archivo_cuatro)

# ESCUDO PRE-MERGE

# TOMAR SOLO REGISTROS MÁS RECIENTES Y FECHA DE CORTE MÁS RECIENTES
df_archivo_cuatro = df_archivo_cuatro[df_archivo_cuatro['Periodo'] == 'Periodo Actual'] # TOMAR SOLO REGISTROS ACTUALES
df_archivo_cuatro = df_archivo_cuatro.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_cuatro = df_archivo_cuatro.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_cuatro = df_archivo_cuatro.drop(columns=['Periodo', 'Fecha de Corte'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_cuatro_actualizar = [
    'NIT',
    'Resultado integral total (ComprehensiveIncome)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_cuatro, "Archivo 4")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_cuatro, "Archivo 4")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_cuatro, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 4) ---")
print(df_maestro[columnas_archivo_cuatro_actualizar].isnull().sum())

del df_archivo_cuatro

# ==========================
# MERGE ARCHIVO 5: 520000_Estado de flujos de efectivo - método indirecto
#===========================

columnas_archivo_cinco = [
    'NIT',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)',
    'Fecha de Corte'
]

df_archivo_cinco = pd.read_excel("../SIIS/pymes_individuales-2024/520000_Estado de flujos de efectivo - método indirecto.xlsx", usecols=columnas_archivo_cinco)
limpiar_nit(df_archivo_cinco)

# ESCUDO PRE-MERGE

# TOMAR SOLO REGISTROS MÁS RECIENTES Y FECHA DE CORTE MÁS RECIENTES
df_archivo_cinco = df_archivo_cinco.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_cinco = df_archivo_cinco.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_cinco = df_archivo_cinco.drop(columns=['Fecha de Corte'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_cinco_actualizar = [
    'NIT',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de operación (CashFlowsFromUsedInOperatingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de inversión (CashFlowsFromUsedInInvestingActivities)',
    'Flujos de efectivo netos procedentes de (utilizados en) actividades de financiación (CashFlowsFromUsedInFinancingActivities)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_cinco, "Archivo 5")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_cinco, "Archivo 5")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_cinco, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 5) ---")
print(df_maestro[columnas_archivo_cinco_actualizar].isnull().sum())

del df_archivo_cinco

# ==========================
# MERGE ARCHIVO 6: 610010_Estado de cambios en el patrimonio (NOTA: SE CAMBIÓ EL FILTRO PERIODO POR EL FILTRO CONCEPTO, BUSCANDO EL ÚLTIMO REGISTRO)
#===========================

columnas_archivo_seis = [
    'NIT',
    'Concepto',
    'Capital emitido [miembro] (CapitalEmitidoMiembro)',
    'Fecha de Corte'
]

df_archivo_seis = pd.read_excel("../SIIS/pymes_individuales-2024/610010_Estado de cambios en el patrimonio.xlsx", usecols=columnas_archivo_seis)
limpiar_nit(df_archivo_seis)

# ESCUDO PRE-MERGE

# 1. TRAMPA SUPERSOCIEDADES: FILTRAR SOLO EL SALDO FINAL DEL AÑO
df_archivo_seis = df_archivo_seis[df_archivo_seis['Concepto'] == 'Emisión de patrimonio']

# 2. ORDENAR POR FECHA DE CORTE PARA DEJAR LA MÁS RECIENTE DE ÚLTIMA Y BORRAR EL RESTO
df_archivo_seis = df_archivo_seis.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_seis = df_archivo_seis.drop_duplicates(subset=['NIT'], keep='last')

# ELIMINAR LOS FILTROS DEL DATASET
df_archivo_seis = df_archivo_seis.drop(columns=['Fecha de Corte', 'Concepto'])

# ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_seis_actualizar = [
    'NIT',
    'Capital emitido [miembro] (CapitalEmitidoMiembro)'
]

# VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_seis, "Archivo 6")

# VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_seis, "Archivo 6")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_seis, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 6) ---")
print(df_maestro[columnas_archivo_seis_actualizar].isnull().sum())

del df_archivo_seis

# ==========================
# MERGE ARCHIVO 7: 800010_Notas - Subclasificaciones de efectivo y equivalentes al efectivo
#===========================

# ==========================
# MERGE ARCHIVO 8: 801000_Notas - Cuentas comerciales por cobrar y otras cuentas por cobrar
#===========================

columnas_archivo_ocho = [
    'NIT',
    'Concepto',
    'Total saldo del ejercicio [miembro] (CxCTotalMiembro)',
    'Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)',
    'Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)',
    'Fecha de Corte'
]

df_archivo_ocho = pd.read_excel("../SIIS/pymes_individuales-2024/801000_Notas - Cuentas comerciales por cobrar y otras cuentas por cobrar.xlsx", usecols=columnas_archivo_ocho)
limpiar_nit(df_archivo_ocho)

# ESCUDO PRE-MERGE

# TOMAR SOLO EL TOTAL DE CUENTAS POR COBRAR CORRIENTES Y FECHA DE CORTE MÁS RECIENTE
df_archivo_ocho = df_archivo_ocho[df_archivo_ocho['Concepto'] == 'Total cuentas por cobrar y otras cuentas por cobrar corrientes']
df_archivo_ocho = df_archivo_ocho.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_ocho = df_archivo_ocho.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_ocho = df_archivo_ocho.drop(columns=['Fecha de Corte', 'Concepto'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_ocho_actualizar = [
    'NIT',
    'Total saldo del ejercicio [miembro] (CxCTotalMiembro)',
    'Total cuentas por cobrar vencidas [miembro] (CxCTotalVencidasMiembro)',
    'Valor vencidas de mas de 360 días [miembro] (CxCVencidasMasDeTrescientossesentaDiasMiembro)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_ocho, "Archivo 8")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_ocho, "Archivo 8")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_ocho, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 8) ---")
print(df_maestro[columnas_archivo_ocho_actualizar].isnull().sum())

del df_archivo_ocho

# ==========================
# MERGE ARCHIVO 9: 801110_Notas - Información a revelar sobre activos biológicos (NOTA: ARCHIVO OMITIDO)
#===========================

# ==========================
# MERGE ARCHIVO 10: 801150_Notas - Información a revelar sobre propiedades de inversión (NOTA: ARCHIVO OMITIDO)
#===========================

# ==========================
# MERGE ARCHIVO 11: 801200_Notas - Propiedades, planta y equipo (NOTA: ARCHIVO OMITIDO)
#===========================

# ==========================
# MERGE ARCHIVO 12: 801300_Notas - Activos intangibles distintos de la plusvalía (NOTA: ARCHIVO OMITIDO)
#===========================

# ==========================
# MERGE ARCHIVO 13: 801300_Notas - 801500_Notas - Deterioro de Activosa (NOTA: ARCHIVO OMITIDO)
#===========================

# ==========================
# MERGE ARCHIVO 14: 801600_Notas - Cuentas comerciales por pagar, otras cuentas por pagar y otros pasivos
#===========================

columnas_archivo_catorce = [
    'NIT',
    'Concepto',
    'Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)',
    'Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)',
    'Fecha de Corte'
]

df_archivo_catorce = pd.read_excel("../SIIS/pymes_individuales-2024/801600_Notas - Cuentas comerciales por pagar, otras cuentas por pagar y otros pasivos.xlsx", usecols=columnas_archivo_catorce)
limpiar_nit(df_archivo_catorce)

# ESCUDO PRE-MERGE

# TOMAR SOLO EL TOTAL DE PASIVOS Y CUENTAS POR PAGAR CORRIENTES Y FECHA DE CORTE MÁS RECIENTE
df_archivo_catorce = df_archivo_catorce[df_archivo_catorce['Concepto'] == 'Total cuentas comerciales por pagar, otras cuentas por pagar y otros pasivos corrientes']
df_archivo_catorce = df_archivo_catorce.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_catorce = df_archivo_catorce.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_catorce = df_archivo_catorce.drop(columns=['Fecha de Corte', 'Concepto'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_catorce_actualizar = [
    'NIT',
    'Total obligaciones vencidas [miembro] (CxPTotalObligacionesVencidasMiembro)',
    'Valor obligaciones vencidas de mas de 360 días [miembro] (CxPObligacionesVencidasMasDeTrescientossesentaDiasMiembro)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_catorce, "Archivo 14")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_catorce, "Archivo 14")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_catorce, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 14) ---")
print(df_maestro[columnas_archivo_catorce_actualizar].isnull().sum())

del df_archivo_catorce

# ==========================
# MERGE ARCHIVO 17: 801800b_Notas - Analisis de otras ganancias (pérdidas) y otros gastos
#===========================

columnas_archivo_diecisiete = [
    'NIT',
    'Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)',
    'Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)',
    'Gastos por beneficios de terminación (TerminationBenefitsExpense)',
    'Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)',
    'Fecha de Corte'
]

df_archivo_diecisiete = pd.read_excel("../SIIS/pymes_individuales-2024/801800b_Notas - Analisis de otras ganancias (pérdidas) y otros gastos.xlsx", usecols=columnas_archivo_diecisiete)
limpiar_nit(df_archivo_diecisiete)

# ESCUDO PRE-MERGE

# TOMAR SOLO REGISTROS MÁS RECIENTES Y FECHA DE CORTE MÁS RECIENTES
df_archivo_diecisiete = df_archivo_diecisiete.sort_values(by=['NIT', 'Fecha de Corte'])
df_archivo_diecisiete = df_archivo_diecisiete.drop_duplicates(subset=['NIT'], keep='last')

#ELIMINAR LOS FILTROS DEL DATASET
df_archivo_diecisiete = df_archivo_diecisiete.drop(columns=['Fecha de Corte'])

#ACTUALIZAR LISTA PARA EL CONTADOR DE NULOS
columnas_archivo_diecisiete_actualizar = [
    'NIT',
    'Ganancias por la disposicion de activos (Ganancias_por_la_disposicion_de_activos)',
    'Pérdidas por disposición de activos (Perdidas_por_disposicion_de_activos)',
    'Gastos por beneficios de terminación (TerminationBenefitsExpense)',
    'Total de gastos por beneficios a los empleados (EmployeeBenefitsExpense)'
]

#VERIFICAR DUPLICADOS
verificar_duplicados(df_archivo_diecisiete, "Archivo 17")

#VERIFICAR QUE NO HAYAN EMPRESAS SIN NIT
verificar_sinNIT(df_archivo_diecisiete, "Archivo 17")

# MERGE

df_maestro = pd.merge(df_maestro, df_archivo_diecisiete, on='NIT', how='left')

# ESCUDO POST-MERGE
assert len(df_maestro) == len(df_caratula), "ERROR. El cruce alteró las filas maestras."

print(f"\n¡Cruce exitoso! Columnas actuales: {len(df_maestro.columns)}")
print("\n--- INFORME DE NULOS (Vacíos que dejó el Archivo 17) ---")
print(df_maestro[columnas_archivo_diecisiete_actualizar].isnull().sum())

del df_archivo_diecisiete

# ==========================
# EXPORTAR CONSOLIDADO CRUDO PARA FASE DE LIMPIEZA
# ==========================
df_maestro.to_pickle("./df_maestro_crudo_2024.pkl")
print("\nArchivo 'df_maestro_crudo_2024.pkl' guardado con éxito Listo para el Script 2.")