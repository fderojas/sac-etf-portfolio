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
    train, test, holdout = dividir(df)
    train_esc, test_esc, holdout_esc, _ = escalar(train, test, holdout)
    print(df.shape)                          # debería tener 40 columnas, no 41
    print(train_esc.abs().max().max())       # ya no debería salir 5237
    print(holdout_esc[["SPY", "QQQ"]].describe())