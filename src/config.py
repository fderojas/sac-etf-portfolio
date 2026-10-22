"""Parámetros del experimento, en un solo sitio."""
COSTE = 0.001      # 0,1% por unidad de peso movida
SEMILLA = 42
PASOS = 50_000


def nombre_modelo():
    pb = round(COSTE * 10_000)          # 0.001 → 10 pb
    return f"sac_coste{pb}pb_seed{SEMILLA}"