"""Entrena el agente SAC con los datos y el entorno corregidos."""
from pathlib import Path
from stable_baselines3 import SAC
from datos import cargar_datos, construir_variables, dividir, escalar, ETFS
from entorno import EntornoCartera
from config import COSTE, SEMILLA, PASOS, nombre_modelo

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELOS = RAIZ / "models" / "corregido"


def main():
    # Datos: mismo proceso que en entorno.py
    df = cargar_datos()
    X = construir_variables(df)
    
    train, test, holdout = dividir(X)
    train_esc, _, _, _ = escalar(train, test, holdout)
    env = EntornoCartera(train_esc, df.loc[train_esc.index, ETFS], coste = COSTE)

    modelo = SAC(
        "MlpPolicy",
        env,
        ent_coef= 'auto',          
        learning_rate=3e-4,
        batch_size=256,
        learning_starts=1000,
        seed=SEMILLA,
        verbose=1,
    )

    modelo.learn(total_timesteps=PASOS)

    CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
    modelo.save(CARPETA_MODELOS / nombre_modelo())
    print("Modelo guardado en", CARPETA_MODELOS)


if __name__ == "__main__":
    main()