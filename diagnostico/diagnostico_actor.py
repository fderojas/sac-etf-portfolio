"""
Diagnóstico del agente SAC del TFM
==================================

Pregunta: ¿el agente decide los pesos mirando el mercado, o devuelve siempre lo mismo?

El TFM concluyó que SAC "converge a equal-weight". Este script comprueba qué hace
realmente la red del actor con los modelos finales entrenados.

Uso (desde la raíz del repositorio):
    python diagnostico/diagnostico_actor.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import SAC

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "BaseCompleta.xlsx"
MODELOS = [RAIZ / "models" / "final" / "sac_3a_final.zip",
           RAIZ / "models" / "final" / "sac_3c_final.zip"]
ETFS = ["SPY", "QQQ", "XLE", "XLF"]


# ---------------------------------------------------------------------------
# 1. Reproducir exactamente la preparación de datos del notebook original
# ---------------------------------------------------------------------------
def preparar_datos():
    df = pd.read_excel(DATOS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").dropna()

    tramos = {
        "train": df.loc[:"2019-12-31"],
        "test": df.loc["2020-01-01":"2022-12-31"],
        "holdout": df.loc["2023-01-01":],
    }
    # Igual que DataManager: el scaler SOLO se aplica a los 4 precios.
    scaler = StandardScaler().fit(tramos["train"][ETFS])
    escalados = {}
    for nombre, t in tramos.items():
        e = t.copy()
        e[ETFS] = scaler.transform(t[ETFS])
        escalados[nombre] = e
    return tramos, escalados


def acciones_a_pesos(a):
    """Misma transformación que step_fixed(): (a+1)/2, recortar y normalizar."""
    w = np.clip((a + 1) / 2, 0, 1)
    return w / (w.sum(axis=-1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# 2. Problema de entrada: variables sin escalar
# ---------------------------------------------------------------------------
def revisar_entradas(escalados):
    train = escalados["train"]
    grandes = {c: round(float(train[c].abs().max()), 1)
               for c in train.columns if train[c].abs().max() > 100}
    print("\n[ENTRADAS] Variables con valores > 100 que llegan a la red sin escalar:")
    for c, v in grandes.items():
        print(f"   {c:<12} máx = {v}")
    if "index" in train.columns:
        print("   ⚠ 'index' es el número de fila del Excel: no es una variable de mercado.")


# ---------------------------------------------------------------------------
# 3. Qué hace la red del actor
# ---------------------------------------------------------------------------
def revisar_modelo(ruta, tramos, escalados):
    modelo = SAC.load(ruta, device="cpu")
    print(f"\n[MODELO] {ruta.name}  (ent_coef = {modelo.ent_coef})")

    for nombre in ["train", "test", "holdout"]:
        obs = torch.as_tensor(escalados[nombre].values.astype(np.float32)[60:-1])

        with torch.no_grad():
            # Salida de la última capa oculta (después de la ReLU)
            rasgos = modelo.actor.extract_features(obs, modelo.actor.features_extractor)
            oculta = modelo.actor.latent_pi(rasgos)
        neuronas_activas = (oculta > 0).float().mean().item()

        acciones, _ = modelo.predict(obs.numpy(), deterministic=True)
        pesos = acciones_a_pesos(acciones)

        # Rendimientos diarios del modelo frente a equal-weight (1/N)
        precios = tramos[nombre][ETFS].values
        r = np.clip(np.log(precios[61:] / precios[60:-1]), -0.15, 0.15)
        r_modelo, r_ew = (pesos * r).sum(axis=1), r.mean(axis=1)

        print(f"   {nombre:<8} neuronas activas: {neuronas_activas:6.1%} | "
              f"variación de la acción en el tiempo: {acciones.std(axis=0).max():.4f} | "
              f"desviación máx. respecto a 25%: {np.abs(pesos - 0.25).max():.3f} | "
              f"corr. con equal-weight: {np.corrcoef(r_modelo, r_ew)[0, 1]:.5f}")


if __name__ == "__main__":
    tramos, escalados = preparar_datos()
    revisar_entradas(escalados)
    for ruta in MODELOS:
        revisar_modelo(ruta, tramos, escalados)
    print("\nConclusión: con 0% de neuronas activas, la salida del actor es solo el sesgo "
          "de la última capa (≈0), que la transformación convierte en pesos de 1/N.")
