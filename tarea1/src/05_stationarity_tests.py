"""
Paso 4: Test de estacionariedad (ADF).

H0 del ADF: la serie tiene raíz unitaria (no estacionaria).
H1: la serie es estacionaria.
Si p-valor < 0.05 se rechaza H0 y se concluye estacionariedad.

Se aplica tanto a precios en niveles como a retornos (log-diferencias),
tal como pide el enunciado.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
OUTPUT_PATH = "results/adf_results.csv"
ALPHA = 0.05


def run_adf(series, label):
    stat, pvalue, used_lag, nobs, crit_values, _ = adfuller(
        series, autolag="AIC", result_object=False
    )
    conclusion = "Estacionaria" if pvalue < ALPHA else "No estacionaria"
    return {
        "serie": label,
        "adf_statistic": stat,
        "p_value": pvalue,
        "used_lag": used_lag,
        "n_obs": nobs,
        "crit_1%": crit_values["1%"],
        "crit_5%": crit_values["5%"],
        "crit_10%": crit_values["10%"],
        "conclusion": conclusion,
    }


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)

    results = [
        run_adf(df["close"].dropna(), "Precios en niveles (close)"),
        run_adf(df["log_return"].dropna(), "Retornos (diff log-close)"),
    ]

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    print(out.to_string(index=False))
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
