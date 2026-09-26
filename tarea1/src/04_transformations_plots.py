"""
Paso 3: Transformaciones y gráficos.

Grafica la serie en niveles, en logaritmos, y en diferencias de logaritmos
(retornos). Diferencia visual esperada:
- Niveles: tendencia y cambios de escala marcados, no estacionaria a simple
  vista (la varianza y el nivel promedio cambian con el tiempo).
- Log-niveles: misma tendencia, pero los movimientos porcentuales grandes
  y pequeños quedan en una escala comparable (el logaritmo comprime los
  saltos de precio absolutos altos).
- Log-retornos: oscilan alrededor de 0 sin tendencia visible, con
  agrupamiento de volatilidad (períodos de oscilación fuerte, como 2022,
  seguidos de calma) -- la marca visual típica de una serie estacionaria
  en media pero con varianza que cambia en el tiempo.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["close"], linewidth=0.8)
    ax.set_title("Algodón (CT=F) -- precio de cierre en niveles")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("USD / libra (centavos)")
    fig.tight_layout()
    fig.savefig("figures/01_levels.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["log_close"], linewidth=0.8, color="tab:orange")
    ax.set_title("Algodón (CT=F) -- log(precio de cierre)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("log(precio)")
    fig.tight_layout()
    fig.savefig("figures/02_log_levels.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["log_return"], linewidth=0.6, color="tab:green")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Algodón (CT=F) -- retornos (diferencia de log-precios)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("log-retorno diario")
    fig.tight_layout()
    fig.savefig("figures/03_log_returns.png", dpi=150)
    plt.close(fig)

    print("Guardado: figures/01_levels.png, figures/02_log_levels.png, figures/03_log_returns.png")


if __name__ == "__main__":
    main()
