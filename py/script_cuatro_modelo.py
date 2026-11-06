import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tr
from sklearn.ensemble import RandomForestClassifier as rd

df_2024 = pd.read_pickle("./df_maestro_con_ratios_2024.pkl")
df_2025 = pd.read_pickle("./df_maestro_con_ratios.pkl")
print("Datasets de ratios cargados")

#LIMPIAR 2024 CON LAS EMPRESAS QUE NO ESTABAN EN QUIEBRA
df_2024 = df_2024[(df_2024["Estado actual"].isin(['ACTIVA', 'EN ETAPA PREOPERATIVA'])) & (df_2024['Flag_Quiebra_Patrimonial'] == 0)]


df_maestro = pd.merge(df_2024, df_2025, on='NIT', how='inner', suffixes=('_2024', '_2025'))

print(pd.crosstab(df_maestro['Estado actual_2024'], df_maestro['Estado actual_2025'],margins=True).to_string())

y = (~df_maestro['Estado actual_2025'].isin(['ACTIVA', 'EN ETAPA PREOPERATIVA'])) | (df_maestro['Flag_Quiebra_Patrimonial_2025'] == 1)
y = y.astype(int)
print(y.value_counts(normalize=True))

columnas_x = (list(df_maestro.loc[:, 'Flag_Reexpresion_Contable_2024':'Flag_Desespero_Gerencial_2024'].columns) + ['Edad_Empresa_Anios_2024'])
x = df_maestro[columnas_x]
x = x.drop(columns=['Flag_Quiebra_Patrimonial_2024'])

print(x.shape)

x_train, x_test, y_train, y_test = tr(x, y, test_size=0.20, random_state=42, stratify=y)
print(x_train.shape, x_test.shape)
print(y_train.value_counts())
print(y_test.value_counts())

modelo = rd(n_estimators=200, class_weight='balanced', max_depth=10, random_state=42)
modelo.fit(x_train,y_train)

test = modelo.predict(x_test)
print("Examen")
print(pd.Series(test).value_counts())


