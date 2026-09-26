"""
Paso 7: Validación de residuos del modelo ganador (AR(1) con constante).

El modelo ganador fue elegido por el usuario en el paso 6 (mejor AIC/BIC
que ARIMA(0,0,0), y coeficiente ar.L1 significativo). Este script reestima
el mismo AR(1) sobre la serie de retornos (para que corra de forma
independiente) y valida si sus residuos se comportan como ruido blanco:

1. Gráfico de residuos en el tiempo.
2. ACF/PACF de los residuos (mismo formato y bandas de significancia
   usados en 06_acf_pacf_identification.py).
3. Test de Ljung-Box en varios rezagos (10 y 20), ajustando los grados
   de libertad por el número de parámetros ARMA estimados (model_df=1,
   por el único coeficiente ar.L1; la constante no cuenta).

No concluye si el modelo es "bueno" o "el mejor" -- eso lo interpreta
el usuario a partir de estos resultados.
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore", message=".*no associated frequency.*")

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
ORDER = (1, 0, 0)
N_LAGS_ACF = 20
LB_LAGS = [10, 20]
MODEL_DF = 1  # un parámetro ARMA estimado: ar.L1 (la constante no cuenta)

RESID_PLOT_PATH = "figures/05_residuals_ar1.png"
ACF_PACF_PLOT_PATH = "figures/06_acf_pacf_residuals_ar1.png"
LJUNG_BOX_PATH = "results/ljung_box_results.csv"


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
    returns = df["log_return"].dropna()

    res = ARIMA(returns, order=ORDER, trend="c").fit()
    resid = res.resid

    # 1. Residuos en el tiempo
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resid.index, resid.values, linewidth=0.6, color="tab:red")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Residuos del AR(1) con constante -- retornos de algodón (CT=F)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Residuo")
    fig.tight_layout()
    fig.savefig(RESID_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Guardado: {RESID_PLOT_PATH}")

    # 2. ACF/PACF de los residuos
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(resid, lags=N_LAGS_ACF, ax=axes[0])
    axes[0].set_title("ACF -- residuos del AR(1)")
    plot_pacf(resid, lags=N_LAGS_ACF, ax=axes[1], method="ywm")
    axes[1].set_title("PACF -- residuos del AR(1)")
    fig.tight_layout()
    fig.savefig(ACF_PACF_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ACF_PACF_PLOT_PATH}")

    # 3. Ljung-Box en varios rezagos
    lb = acorr_ljungbox(resid, lags=LB_LAGS, model_df=MODEL_DF, return_df=True)
    lb = lb.rename(columns={"lb_stat": "estadistico", "lb_pvalue": "p_value"})
    lb.index.name = "lags"
    lb.to_csv(LJUNG_BOX_PATH)
    print(f"\nGuardado: {LJUNG_BOX_PATH}")
    print(lb)

    # 4. Resumen breve (sin conclusión final)
    resid_mean = resid.mean()
    resid_std = resid.std()
    print("\n=== Resumen de la validación (sin conclusión final) ===")
    print(
        f"- Media de los residuos: {resid_mean:.6f} (std: {resid_std:.6f}). "
        "El gráfico en el tiempo muestra si oscilan sin patrón visible alrededor de 0 "
        "o si hay tramos de mayor/menor volatilidad agrupada."
    )
    print(
        "- ACF/PACF de residuos: revisar si quedan rezagos fuera de la banda de "
        "significancia (95%) -- si prácticamente todos los rezagos caen dentro de la "
        "banda, es consistente con ausencia de autocorrelación remanente."
    )
    for lag, row in lb.iterrows():
        veredicto = "p < 0.05 (se rechaza H0 de no-autocorrelación)" if row["p_value"] < 0.05 else "p >= 0.05 (no se rechaza H0 de no-autocorrelación)"
        print(f"- Ljung-Box lag {lag}: estadístico={row['estadistico']:.3f}, p-valor={row['p_value']:.4f} -> {veredicto}")


if __name__ == "__main__":
    main()
