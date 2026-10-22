"""Evalúa el agente corregido frente a equal-weight en train, test y holdout."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from datos import cargar_datos, construir_variables, dividir, escalar, ETFS
from entorno import EntornoCartera
from config import COSTE, nombre_modelo

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "corregido" / f"{nombre_modelo()}.zip"

def recorrer(modelo, env):
    """Juega un episodio completo con la política determinista y guarda todo."""
    obs, _ = env.reset()
    observaciones, pesos, rend = [], [], []
    terminado = False
    while not terminado:
        observaciones.append(obs)
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, recompensa, terminado, _, info = env.step(accion)
        pesos.append(info["pesos"])
        rend.append(recompensa / 100)          # rendimiento neto de costes
    return np.array(observaciones), np.array(pesos), np.array(rend)


def sharpe_anual(r):
    """Sharpe anualizado de rendimientos diarios (sin tipo libre de riesgo)."""
    return r.mean() / r.std() * np.sqrt(252)


def max_drawdown(r):
    valor = np.exp(np.cumsum(r))                 # valor de la cartera (rend. logarítmicos)
    return (valor / np.maximum.accumulate(valor) - 1).min()


def neuronas_activas(modelo, observaciones):
    with torch.no_grad():
        x = torch.as_tensor(observaciones, dtype=torch.float32)
        oculta = modelo.actor.latent_pi(modelo.actor.extract_features(x, modelo.actor.features_extractor))
    return (oculta > 0).float().mean().item()


def main():
    df = cargar_datos()
    X = construir_variables(df)
    tramos = dict(zip(["train", "test", "holdout"], escalar(*dividir(X))[:3]))
    modelo = SAC.load(MODELO, device="cpu")

    filas = []
    for nombre, X_esc in tramos.items():
        env = EntornoCartera(X_esc, df.loc[X_esc.index, ETFS])
        observaciones, pesos, r_agente = recorrer(modelo, env)
        r_ew = env.rend.mean(axis=1)           # equal-weight rebalanceado a diario

        filas.append({
            "tramo": nombre,
            "Sharpe agente": sharpe_anual(r_agente),
            "Sharpe EW": sharpe_anual(r_ew),
            "Rend. total agente": np.exp(r_agente.sum()) - 1,
            "Rend. total EW": np.exp(r_ew.sum()) - 1,
            "MaxDD agente": max_drawdown(r_agente),
            "MaxDD EW": max_drawdown(r_ew),
            "Giro medio diario": np.abs(np.diff(pesos, axis=0)).sum(axis=1).mean(),
            "Neuronas activas": neuronas_activas(modelo, observaciones),
        })
        print(f"{nombre:<8} pesos medios: " +
              ", ".join(f"{e} {w:.0%}" for e, w in zip(ETFS, pesos.mean(axis=0))))

    tabla = pd.DataFrame(filas).set_index("tramo").T
    print("\n", tabla.round(3).to_string())
    (RAIZ / "results").mkdir(exist_ok=True)
    tabla.to_csv(RAIZ / "results" / f"evaluacion_{nombre_modelo()}.csv")

if __name__ == "__main__":
    main()
