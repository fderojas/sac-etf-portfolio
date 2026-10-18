"""Entorno de trading: el agente reparte el capital entre 4 ETFs cada día."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EntornoCartera(gym.Env):

    def __init__(self, X, precios, coste=0.0001):
        super().__init__()
        self.X = X.values.astype(np.float32)          # variables escaladas (lo que ve el agente)
        self.rend = np.log(precios.values[1:] / precios.values[:-1])  # rendimientos diarios
        self.n = precios.shape[1]                      # número de ETFs (4)
        self.coste = coste                             # 0,01% por unidad de peso movida

        # La acción son 4 números entre -3 y 3; softmax los convierte en pesos
        self.action_space = spaces.Box(-3, 3, shape=(self.n,), dtype=np.float32)
        # Observación = variables del día + pesos actuales
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.X.shape[1] + self.n,), dtype=np.float32)

    def _obs(self):
        t = min(self.t, len(self.X) - 1)
        return np.concatenate([self.X[t], self.pesos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pesos = np.ones(self.n) / self.n          # empezamos en equal-weight
        return self._obs(), {}

    def step(self, accion):
        # TODO 1: softmax → nuevos = exp(accion) / suma de exp(accion)
        nuevos = np.exp(accion) / sum(np.exp(accion))

        # TODO 2: coste = self.coste × suma de |nuevos − self.pesos|
        coste = self.coste * sum(np.abs(nuevos - self.pesos))

        # TODO 3: rendimiento de la cartera hoy = producto escalar de nuevos y self.rend[self.t]
        r_cartera = np.dot(nuevos, self.rend[self.t])

        recompensa = 100 * (r_cartera - coste)         # ×100: en puntos porcentuales

        self.pesos = nuevos
        self.t += 1
        terminado = self.t >= len(self.rend)
        return self._obs(), float(recompensa), terminado, False, {"pesos": nuevos, "r_cartera": r_cartera}


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from datos import cargar_datos, construir_variables, dividir, escalar, ETFS

    df = cargar_datos()
    X = construir_variables(df)
    train, test, holdout = dividir(X)
    train_esc, _, _, _ = escalar(train, test, holdout)

    env = EntornoCartera(train_esc, df.loc[train_esc.index, ETFS])
    check_env(env)                                     # comprueba que cumple el formato de Gymnasium
    obs, _ = env.reset()
    obs, r, fin, _, info = env.step(env.action_space.sample())   # un paso con una acción aleatoria
    print("Pesos:", info["pesos"].round(3), "| Recompensa:", round(r, 4))