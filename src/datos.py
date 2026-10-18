"""Carga y preparación de los datos, sin fuga de información."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
ETFS = ["SPY", "QQQ", "XLE", "XLF"]


def cargar_datos():
    df = pd.read_excel(RAIZ / "data" / "BaseCompleta.xlsx")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.drop(columns="index")
    df = df.dropna()
    return df

def construir_variables(df):
    """Devuelve las variables que verá el agente, todas estacionarias."""
    X = df.copy()

    for etf in ETFS:
        # Distancia del precio a su media móvil de 20 días (ejemplo hecho):
        X[f"{etf}_dist_sma_20"] = X[etf] / X[f"{etf}_sma_20"] - 1
        # Distancia del precio a su media móvil de 50 días (ejemplo hecho):
        X[f"{etf}_dist_sma_50"] = X[etf] / X[f"{etf}_sma_50"] - 1

    medias = [f"{etf}_sma_{v}" for etf in ETFS for v in (20, 50)]
    X = X.drop(columns=ETFS + medias + ["VIX_y", "FED_RATE_y"])
    return X


def dividir(df):
    train_data = df.loc[:'2019-12-31'].copy()
    test_data = df.loc['2020-01-01':'2022-12-31'].copy()
    holdout_data = df.loc['2023-01-01':].copy()
    return train_data, test_data, holdout_data

            
def escalar(train, test, holdout):
    scaler = StandardScaler()
    scaler.fit(train)          # aprende media y std SOLO de train

    train_esc = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test_esc = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    holdout_esc = pd.DataFrame(scaler.transform(holdout), index=holdout.index, columns=holdout.columns)

    return train_esc, test_esc, holdout_esc, scaler


if __name__ == "__main__":
    df = cargar_datos()
    X = construir_variables(df)
    train, test, holdout = dividir(X)
    train_esc, test_esc, holdout_esc, _ = escalar(train, test, holdout)
    print(X.shape)
    print(holdout_esc.describe().loc[["min", "max"]].T)