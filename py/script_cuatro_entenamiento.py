"""Entrenamiento sin fuga temporal para riesgo empresarial 2024 -> 2025."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_2024 = ROOT / "df_maestro_con_ratios_2024.pkl"
DATA_2025 = ROOT / "df_maestro_con_ratios.pkl"
HISTORY_DIR = ROOT.parent / "SIIS" / "pymes_individuales-2024"
MODEL_OUTPUT = ROOT / "modelo_riesgo_2024_2025.joblib"
RANDOM_STATE = 42
SECTOR_COLUMN = "__sector_ciiu"
SECTOR_FEATURES = (
    "Nivel_Endeudamiento_2024",
    "ROA_Neto_2024",
    "Razon_Corriente_2024",
    "Tasa_Consumo_Patrimonio_2024",
    "Altman_Z_Emergente_2024",
    "Var_Pct_Patrimonio_23_24",
    "Delta_Endeudamiento_23_24",
)


class SectorDeviationTransformer(BaseEstimator, TransformerMixin):
    """Calcula desviaciones sectoriales usando exclusivamente datos de fit."""

    def __init__(self, relative_features=SECTOR_FEATURES, sector_column=SECTOR_COLUMN):
        self.relative_features = relative_features
        self.sector_column = sector_column

    def fit(self, X, y=None):
        X = X.copy()
        missing = set(self.relative_features + (self.sector_column,)) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan columnas para el ajuste sectorial: {sorted(missing)}")

        sectors = X[self.sector_column].fillna("__SIN_SECTOR__").astype(str)
        self.global_means_ = X.loc[:, self.relative_features].mean()
        self.sector_means_ = X.loc[:, self.relative_features].groupby(sectors).mean()
        return self

    def transform(self, X):
        X = X.copy()
        missing = set(self.relative_features + (self.sector_column,)) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan columnas para transformar: {sorted(missing)}")

        sectors = X[self.sector_column].fillna("__SIN_SECTOR__").astype(str)
        for column in self.relative_features:
            sector_mean = sectors.map(self.sector_means_[column])
            sector_mean = sector_mean.fillna(self.global_means_[column])
            X[column] = pd.to_numeric(X[column], errors="coerce") - sector_mean

        return X.drop(columns=[self.sector_column])


sys.modules.setdefault("script_cuatro_entenamiento", sys.modules[__name__])
SectorDeviationTransformer.__module__ = "script_cuatro_entenamiento"


def normalize_nit(values):
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def load_previous_year():
    if not HISTORY_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de SIIS 2024: {HISTORY_DIR}"
        )

    balance_files = sorted(HISTORY_DIR.glob("210030*.xlsx"))
    income_files = sorted(HISTORY_DIR.glob("310030*.xlsx"))
    if not balance_files or not income_files:
        raise FileNotFoundError(
            "Se requieren los Excel SIIS 210030 y 310030 de 2024 en "
            f"{HISTORY_DIR} para construir tendencias 2023->2024."
        )

    balance_columns = [
        "NIT", "Periodo", "Fecha de Corte",
        "Total de activos (Assets)", "Total pasivos (Liabilities)",
        "Patrimonio total (Equity)", "Activos corrientes totales (CurrentAssets)",
        "Pasivos corrientes totales (CurrentLiabilities)",
        "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)",
    ]
    income_columns = [
        "NIT", "Periodo", "Fecha de Corte",
        "Ingresos de actividades ordinarias (Revenue)",
        "Ganancia (pérdida) (ProfitLoss)",
        "Ganancia (pérdida) por actividades de operación "
        "(GananciaPerdidaPorActividadesDeOperacion)",
    ]

    balance = pd.read_excel(balance_files[0], usecols=balance_columns)
    income = pd.read_excel(income_files[0], usecols=income_columns)

    for frame in (balance, income):
        frame["NIT"] = normalize_nit(frame["NIT"])
        frame.dropna(subset=["NIT"], inplace=True)

    balance = (
        balance[balance["Periodo"] == "Periodo Anterior"]
        .sort_values(["NIT", "Fecha de Corte"])
        .drop_duplicates("NIT", keep="last")
        .drop(columns=["Periodo", "Fecha de Corte"])
    )
    income = (
        income[income["Periodo"] == "Periodo Anterior"]
        .sort_values(["NIT", "Fecha de Corte"])
        .drop_duplicates("NIT", keep="last")
        .drop(columns=["Periodo", "Fecha de Corte"])
    )
    previous = balance.merge(income, on="NIT", how="outer", validate="one_to_one")
    return previous.add_suffix("_2023").rename(columns={"NIT_2023": "NIT"})


def build_training_data():
    if not DATA_2024.is_file() or not DATA_2025.is_file():
        raise FileNotFoundError(
            f"Se requieren {DATA_2024.name} y {DATA_2025.name} junto al script."
        )

    df_2024 = pd.read_pickle(DATA_2024)
    df_2025 = pd.read_pickle(DATA_2025)
    for name, frame in (("2024", df_2024), ("2025", df_2025)):
        if "NIT" not in frame or frame["NIT"].isna().any():
            raise ValueError(f"El dataset {name} tiene NIT ausentes o no tiene la columna.")
        frame["NIT"] = normalize_nit(frame["NIT"])
        if frame["NIT"].duplicated().any():
            raise ValueError(f"El dataset {name} contiene NIT duplicados.")

    required_2024 = {"Estado actual", "Flag_Quiebra_Patrimonial"}
    required_2025 = {"Estado actual", "Flag_Quiebra_Patrimonial"}
    if not required_2024.issubset(df_2024.columns):
        raise ValueError(f"Faltan columnas en 2024: {sorted(required_2024 - set(df_2024.columns))}")
    if not required_2025.issubset(df_2025.columns):
        raise ValueError(f"Faltan columnas en 2025: {sorted(required_2025 - set(df_2025.columns))}")

    healthy_2024 = df_2024[
        df_2024["Estado actual"].isin(["ACTIVA", "EN ETAPA PREOPERATIVA"])
        & (df_2024["Flag_Quiebra_Patrimonial"] == 0)
    ].copy()
    cohort = healthy_2024.merge(
        df_2025,
        on="NIT",
        how="inner",
        suffixes=("_2024", "_2025"),
        validate="one_to_one",
    )
    cohort = cohort.merge(
        load_previous_year(),
        on="NIT",
        how="left",
        validate="one_to_one",
    )

    target = (
        ~cohort["Estado actual_2025"].isin(["ACTIVA", "EN ETAPA PREOPERATIVA"])
    ) | (cohort["Flag_Quiebra_Patrimonial_2025"] == 1)
    y = target.astype("int8")

    start = "Flag_Reexpresion_Contable_2024"
    end = "Flag_Desespero_Gerencial_2024"
    if start not in cohort.columns or end not in cohort.columns:
        raise ValueError("No se encontró el rango esperado de indicadores 2024.")
    start_idx = cohort.columns.get_loc(start)
    end_idx = cohort.columns.get_loc(end)
    if start_idx > end_idx:
        raise ValueError("El orden de las columnas de indicadores 2024 no es válido.")

    X = cohort.iloc[:, start_idx : end_idx + 1].copy()
    X.drop(columns=["Flag_Quiebra_Patrimonial_2024"], errors="ignore", inplace=True)
    X["Edad_Empresa_Anios_2024"] = cohort["Edad_Empresa_Anios_2024"]

    def numeric(column):
        return pd.to_numeric(cohort[column], errors="coerce")

    assets_24 = numeric("Total de activos (Assets)_2024").clip(lower=0)
    liabilities_24 = numeric("Total pasivos (Liabilities)_2024").clip(lower=0)
    equity_24 = numeric("Patrimonio total (Equity)_2024").clip(lower=0)
    current_assets_24 = numeric("Activos corrientes totales (CurrentAssets)_2024").clip(lower=0)
    current_liabilities_24 = numeric("Pasivos corrientes totales (CurrentLiabilities)_2024").clip(lower=0)
    cash_24 = numeric("Efectivo y equivalentes al efectivo (CashAndCashEquivalents)_2024").clip(lower=0)
    revenue_24 = numeric("Ingresos de actividades ordinarias (Revenue)_2024").clip(lower=0)
    profit_24 = numeric("Ganancia (pérdida) (ProfitLoss)_2024")
    ebit_24 = numeric(
        "Ganancia (pérdida) por actividades de operación "
        "(GananciaPerdidaPorActividadesDeOperacion)_2024"
    )
    operating_cash_24 = numeric(
        "Flujos de efectivo netos procedentes de (utilizados en) actividades de "
        "operación (CashFlowsFromUsedInOperatingActivities)_2024"
    )
    finance_costs_24 = numeric("Costos financieros (FinanceCosts)_2024").clip(lower=0)

    loss_24 = np.maximum(0.0, -profit_24)
    X["Tasa_Consumo_Patrimonio_2024"] = np.where(
        equity_24 > 0, loss_24 / equity_24, np.where(loss_24 > 0, 5.0, 0.0)
    ).clip(0.0, 5.0)
    ebit_loss_24 = np.maximum(0.0, -ebit_24)
    X["Tasa_Sangria_EBIT_Patrimonio_2024"] = np.where(
        equity_24 > 0, ebit_loss_24 / equity_24,
        np.where(ebit_loss_24 > 0, 5.0, 0.0),
    ).clip(0.0, 5.0)

    projected_equity_24 = equity_24 + profit_24
    X["Ratio_Patrimonio_Inercial_Sobre_Activos_2024"] = np.where(
        assets_24 > 0, projected_equity_24 / assets_24, -1.0
    ).clip(-1.0, 1.0)
    X["Flag_Colapso_Patrimonial_Inercial_2024"] = (projected_equity_24 <= 0).astype("int8")
    X["ROA_Neto_2024"] = np.where(assets_24 > 0, profit_24 / assets_24, -1.0).clip(-1.0, 1.0)
    X["ROE_2024"] = np.where(equity_24 > 0, profit_24 / equity_24, -2.0).clip(-2.0, 2.0)
    X["Colchon_Patrimonio_Sobre_Pasivos_2024"] = np.where(
        liabilities_24 > 0, equity_24 / liabilities_24, 5.0
    ).clip(0.0, 5.0)
    X["Impacto_Intereses_Sobre_Patrimonio_2024"] = np.where(
        equity_24 > 0, finance_costs_24 / equity_24,
        np.where(finance_costs_24 > 0, 2.0, 0.0),
    ).clip(0.0, 2.0)

    cash_burn_24 = np.maximum(0.0, -operating_cash_24)
    X["Tasa_Agotamiento_Efectivo_2024"] = np.where(
        cash_24 > 0, cash_burn_24 / cash_24,
        np.where(cash_burn_24 > 0, 5.0, 0.0),
    ).clip(0.0, 5.0)
    current_deficit_24 = np.maximum(0.0, current_liabilities_24 - current_assets_24)
    X["Deficit_Corriente_Sobre_Patrimonio_2024"] = np.where(
        equity_24 > 0, current_deficit_24 / equity_24,
        np.where(current_deficit_24 > 0, 5.0, 0.0),
    ).clip(0.0, 5.0)

    X["Log_Activos_2024"] = np.log1p(assets_24)
    X["Log_Ingresos_2024"] = np.log1p(revenue_24)
    X["Log_Patrimonio_2024"] = np.log1p(equity_24)
    X["Rotacion_Activos_Totales_2024"] = np.where(
        assets_24 > 0, revenue_24 / assets_24, 0.0
    ).clip(0.0, 10.0)
    X["Concentracion_Deuda_Corto_Plazo_2024"] = np.where(
        liabilities_24 > 0, current_liabilities_24 / liabilities_24, 0.0
    ).clip(0.0, 1.0)
    X["Flag_NIIF_Plenas_2024"] = (
        cohort["Punto de Entrada_2024"].astype("string").str.upper().str.contains(
            "PLENAS", na=False
        )
    ).astype("int8")

    previous_columns = [
        "Total de activos (Assets)_2023",
        "Total pasivos (Liabilities)_2023",
        "Patrimonio total (Equity)_2023",
        "Activos corrientes totales (CurrentAssets)_2023",
        "Pasivos corrientes totales (CurrentLiabilities)_2023",
        "Efectivo y equivalentes al efectivo (CashAndCashEquivalents)_2023",
        "Ingresos de actividades ordinarias (Revenue)_2023",
        "Ganancia (pérdida) (ProfitLoss)_2023",
        "Ganancia (pérdida) por actividades de operación "
        "(GananciaPerdidaPorActividadesDeOperacion)_2023",
    ]
    historical_missing = cohort[previous_columns].isna().any(axis=1)
    X["Flag_Datos_Historicos_2023_Faltantes"] = historical_missing.astype("int8")
    assets_23 = numeric(previous_columns[0]).fillna(assets_24).clip(lower=0)
    liabilities_23 = numeric(previous_columns[1]).fillna(liabilities_24).clip(lower=0)
    equity_23 = numeric(previous_columns[2]).fillna(equity_24)
    current_assets_23 = numeric(previous_columns[3]).fillna(current_assets_24).clip(lower=0)
    current_liabilities_23 = numeric(previous_columns[4]).fillna(current_liabilities_24).clip(lower=0)
    cash_23 = numeric(previous_columns[5]).fillna(cash_24).clip(lower=0)
    revenue_23 = numeric(previous_columns[6]).fillna(revenue_24).clip(lower=0)
    profit_23 = numeric(previous_columns[7]).fillna(profit_24)
    ebit_23 = numeric(previous_columns[8]).fillna(ebit_24)

    debt_ratio_23 = np.where(
        assets_23 > 0, liabilities_23 / assets_23, X["Nivel_Endeudamiento_2024"]
    ).clip(0.0, 2.0)
    roa_23 = np.where(assets_23 > 0, profit_23 / assets_23, X["ROA_Neto_2024"]).clip(-1.0, 1.0)
    current_ratio_23 = np.where(
        current_liabilities_23 > 0,
        current_assets_23 / current_liabilities_23,
        X["Razon_Corriente_2024"],
    ).clip(0.0, 10.0)

    X["Delta_Endeudamiento_23_24"] = (
        X["Nivel_Endeudamiento_2024"] - debt_ratio_23
    ).clip(-1.0, 1.0)
    X["Var_Pct_Patrimonio_23_24"] = np.where(
        equity_23 > 0, (equity_24 - equity_23) / equity_23, 0.0
    ).clip(-1.0, 2.0)
    X["Var_Pct_Activos_23_24"] = np.where(
        assets_23 > 0, (assets_24 - assets_23) / assets_23, 0.0
    ).clip(-1.0, 2.0)
    X["Var_Pct_Ingresos_23_24"] = np.where(
        revenue_23 > 0, (revenue_24 - revenue_23) / revenue_23,
        np.where(revenue_24 > 0, 1.0, 0.0),
    ).clip(-1.0, 2.0)
    X["Var_Pct_Efectivo_23_24"] = np.where(
        cash_23 > 0, (cash_24 - cash_23) / cash_23, 0.0
    ).clip(-1.0, 2.0)
    X["Delta_ROA_Neto_23_24"] = (X["ROA_Neto_2024"] - roa_23).clip(-1.0, 1.0)
    X["Delta_Razon_Corriente_23_24"] = (
        X["Razon_Corriente_2024"] - current_ratio_23
    ).clip(-5.0, 5.0)
    X["Flag_Dos_Anios_Perdida_23_24"] = ((profit_23 < 0) & (profit_24 < 0)).astype("int8")
    X["Flag_Dos_Anios_EBIT_Negativo_23_24"] = ((ebit_23 < 0) & (ebit_24 < 0)).astype("int8")
    X["Flag_Desplome_Ventas_23_24"] = (X["Var_Pct_Ingresos_23_24"] < -0.30).astype("int8")
    X["Flag_Destruccion_Patrimonio_Severa_23_24"] = (
        X["Var_Pct_Patrimonio_23_24"] < -0.30
    ).astype("int8")

    X["Altman_Z_Emergente_2024"] = (
        1.2 * X["Capital_Trabajo_Sobre_Activos_2024"]
        + 3.3 * X["ROA_Operativo_2024"]
        + X["ROA_Neto_2024"]
        + 0.6 * X["Colchon_Patrimonio_Sobre_Pasivos_2024"]
        + X["Rotacion_Activos_Totales_2024"]
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
    X["Indice_Asfixia_Operativa_2024"] = X["Nivel_Endeudamiento_2024"] * (
        1.0 - X["Razon_Corriente_2024"].clip(0.0, 2.0) / 2.0
    )
    flag_columns = [column for column in X.columns if column.startswith("Flag_")]
    X["Total_Banderas_Rojas_2024"] = X[flag_columns].sum(axis=1, min_count=1)
    X[SECTOR_COLUMN] = (
        cohort[
            "Clasificación Industrial Internacional Uniforme Versión 4 A.C (CIIU)_2024"
        ].astype("string").str[:2]
    )

    if not X.index.equals(y.index):
        raise ValueError("X e y quedaron desalineados durante la ingeniería de variables.")
    if cohort["NIT"].duplicated().any():
        raise ValueError("La cohorte de modelado contiene más de una fila por empresa.")
    return X, y, cohort["NIT"].reset_index(drop=True)


def make_pipeline(classifier, scale=False):
    steps = [
        ("sector", SectorDeviationTransformer()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def f_beta_scores(y_true, probabilities, beta):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = beta**2
    scores = (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall + 1e-12
    )
    return scores, thresholds


def choose_traffic_light_thresholds(y_true, oof_probabilities):
    yellow_scores, thresholds = f_beta_scores(y_true, oof_probabilities, beta=2.0)
    if not len(thresholds):
        raise ValueError("No fue posible calcular umbrales con las predicciones OOF.")

    yellow_index = int(np.argmax(yellow_scores))
    yellow_threshold = float(thresholds[yellow_index])

    red_scores, red_candidates = f_beta_scores(y_true, oof_probabilities, beta=0.5)
    eligible = np.flatnonzero(red_candidates >= yellow_threshold)
    if not len(eligible):
        raise ValueError("No se encontró un corte rojo igual o superior al amarillo.")
    red_index = int(eligible[np.argmax(red_scores[eligible])])
    return yellow_threshold, float(red_candidates[red_index])


def main():
    X, y, company_ids = build_training_data()
    if y.nunique() != 2 or int(y.sum()) < 5:
        raise ValueError("Se requieren ambas clases y al menos cinco casos positivos.")
    print(
        f"Cohorte: {len(X):,} empresas | variables: {X.shape[1]} | "
        f"positivos: {int(y.sum()):,} ({y.mean():.2%})"
    )

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        company_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    if set(ids_train).intersection(ids_test):
        raise ValueError("Fuga de NIT entre entrenamiento y prueba.")
    print(
        f"Partición independiente por NIT: train={len(X_train):,}, "
        f"test={len(X_test):,}; positivos test={int(y_test.sum()):,}"
    )

    candidates = {
        "HistGradientBoosting_balanced": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.04,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=8.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "RandomForest_balanced": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=9,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "ExtraTrees_balanced": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=9,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "LogisticRegression_balanced": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = {}
    print("\nComparación de candidatos con predicciones out-of-fold (solo train):")
    for name, pipeline in candidates.items():
        probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        oof_predictions[name] = probabilities
        score = average_precision_score(y_train, probabilities)
        print(f"  {name:34s} PR-AUC OOF={score:.4f}")

    best_name = max(
        oof_predictions,
        key=lambda name: average_precision_score(y_train, oof_predictions[name]),
    )
    best_pipeline = candidates[best_name]
    yellow_threshold, red_threshold = choose_traffic_light_thresholds(
        y_train, oof_predictions[best_name]
    )
    best_pipeline.fit(X_train, y_train)

    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= red_threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
    print("\nEvaluación final (test reservado; no intervino en selección ni umbrales):")
    print(f"  Modelo: {best_name}")
    print(f"  PR-AUC: {average_precision_score(y_test, test_probabilities):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, test_probabilities):.4f}")
    print(f"  Umbral amarillo OOF (F2): {yellow_threshold:.5f}")
    print(f"  Umbral rojo OOF (F0.5): {red_threshold:.5f}")
    print(
        f"  Rojo: precision={tp / (tp + fp) if tp + fp else 0:.3f}, "
        f"recall={tp / (tp + fn) if tp + fn else 0:.3f}, "
        f"F1={f1_score(y_test, test_predictions, zero_division=0):.3f}, "
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
    )

    traffic_light = np.where(
        test_probabilities >= red_threshold,
        "ROJO",
        np.where(test_probabilities >= yellow_threshold, "AMARILLO", "VERDE"),
    )
    test_summary = pd.DataFrame(
        {"Semaforo": traffic_light, "Evento_2025": y_test.to_numpy()}
    )
    summary = test_summary.groupby("Semaforo", sort=False).agg(
        Empresas=("Evento_2025", "size"),
        Eventos=("Evento_2025", "sum"),
        Tasa_Evento=("Evento_2025", "mean"),
    )
    print("\nRendimiento por zona en el test:")
    print(summary.to_string())

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "yellow_threshold": yellow_threshold,
        "red_threshold": red_threshold,
        "target_definition": (
            "Estado actual 2025 distinto de ACTIVA/EN ETAPA PREOPERATIVA "
            "o Flag_Quiebra_Patrimonial_2025 == 1"
        ),
        "training_period": "2024",
        "outcome_period": "2025",
        "notes": (
            "El score del clasificador no es una probabilidad calibrada. "
            "La imputación y las medias sectoriales se ajustan dentro de cada fold."
        ),
    }
    joblib.dump(artifact, MODEL_OUTPUT)
    print(f"\nModelo y cortes guardados en: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
