"""
Paso 5: Identificación de modelos candidatos vía ACF/PACF.

Se grafican ACF y PACF sobre la serie de RETORNOS (log-diferencias), que
es la que el test ADF confirmó como estacionaria -- no tiene sentido
identificar un modelo AR/MA/ARMA sobre una serie no estacionaria (precios
en niveles).

Este script solo describe la forma de ACF/PACF y sugiere qué familia de
modelo(s) parece más consistente con la regla práctica:
- PACF se corta bruscamente y ACF decae lento  -> sugiere AR(p)
- ACF se corta bruscamente y PACF decae lento  -> sugiere MA(q)
- Ambas decaen gradualmente, sin corte claro   -> sugiere ARMA(p,q)
- (la serie ya viene diferenciada una vez respecto al precio, por lo que
  cualquier modelo estimado sobre retornos es, en términos de la serie de
  precios original, un ARIMA(p,1,q))

La decisión final de qué orden(es) probar la toma el usuario después de
revisar los gráficos generados aquí.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
N_LAGS = 20
SIGNIFICANCE_BAND = 1.96  # aprox. IC 95% bajo H0 de no autocorrelación


def first_cutoff_lag(values, conf_radius):
    """Primer lag (>=1) donde el valor cae dentro de la banda de no significancia
    y se mantiene así en los lags siguientes -- proxy simple de 'corte' visual."""
    for lag in range(1, len(values)):
        if all(abs(v) <= conf_radius for v in values[lag:]):
            return lag
    return None


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
    returns = df["log_return"].dropna()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(returns, lags=N_LAGS, ax=axes[0])
    axes[0].set_title("ACF -- retornos (log-diferencias) de algodón (CT=F)")
    plot_pacf(returns, lags=N_LAGS, ax=axes[1], method="ywm")
    axes[1].set_title("PACF -- retornos (log-diferencias) de algodón (CT=F)")
    fig.tight_layout()
    fig.savefig("figures/04_acf_pacf_returns.png", dpi=150)
    plt.close(fig)
    print("Guardado: figures/04_acf_pacf_returns.png")

    acf_vals = acf(returns, nlags=N_LAGS)[1:]
    pacf_vals = pacf(returns, nlags=N_LAGS, method="ywm")[1:]
    conf_radius = SIGNIFICANCE_BAND / np.sqrt(len(returns))

    acf_cut = first_cutoff_lag(acf_vals, conf_radius)
    pacf_cut = first_cutoff_lag(pacf_vals, conf_radius)

    print(f"\nBanda de no significancia aprox. (95%): +/-{conf_radius:.4f}")
    print(f"ACF  -- lags 1..{N_LAGS}: {np.round(acf_vals, 4).tolist()}")
    print(f"PACF -- lags 1..{N_LAGS}: {np.round(pacf_vals, 4).tolist()}")
    print(f"\nPrimer lag desde el cual ACF  se mantiene dentro de la banda: {acf_cut}")
    print(f"Primer lag desde el cual PACF se mantiene dentro de la banda: {pacf_cut}")

    print(
        "\nSugerencia automática (a confirmar por el usuario viendo el gráfico):\n"
        "- Si PACF se corta claro y ACF decae lento -> candidato AR(p), p = lag de corte de PACF.\n"
        "- Si ACF se corta claro y PACF decae lento -> candidato MA(q), q = lag de corte de ACF.\n"
        "- Si ambas decaen gradualmente / hay varios lags significativos -> candidato ARMA(p,q).\n"
        "- En términos de la serie de precios original: cualquier modelo sobre retornos\n"
        "  equivale a ARIMA(p,1,q) sobre 'close'."
    )


if __name__ == "__main__":
    main()
