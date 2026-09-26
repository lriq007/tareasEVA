"""
Paso 2: Análisis exploratorio liviano.

Resumen estadístico básico de precios y retornos, y detección (sin
eliminación) de outliers evidentes en los retornos diarios.

Los outliers NO se eliminan: en una serie de precios de commodity, los
saltos grandes suelen ser eventos económicos reales (shocks de oferta o
demanda -- p.ej. la corrida del precio del algodón en 2010-2011 -- o
crisis globales como marzo 2020), no errores de captura de datos.
Eliminarlos sesgaría el análisis de estacionariedad y de residuos que
viene después, así que solo se listan para discutirlos en el informe.
"""

import numpy as np
import pandas as pd

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
SUMMARY_PATH = "results/summary_stats.csv"
OUTLIERS_PATH = "results/outliers_flagged.csv"
Z_THRESHOLD = 3


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)

    summary = pd.DataFrame(
        {
            "close": df["close"].describe(),
            "log_return": df["log_return"].describe(),
        }
    )
    summary.loc["date_min"] = [df.index.min().date(), df.index.min().date()]
    summary.loc["date_max"] = [df.index.max().date(), df.index.max().date()]
    summary.to_csv(SUMMARY_PATH)
    print("Resumen estadístico:")
    print(summary)

    returns = df["log_return"].dropna()
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = df.loc[z_scores[z_scores.abs() > Z_THRESHOLD].index, ["close", "log_return"]]
    outliers = outliers.assign(z_score=z_scores.loc[outliers.index])
    outliers.to_csv(OUTLIERS_PATH)

    print(f"\nOutliers evidentes (|z-score| > {Z_THRESHOLD}), flageados sin eliminar: {len(outliers)}")
    if not outliers.empty:
        print(outliers)
    print(f"\nGuardado: {SUMMARY_PATH}, {OUTLIERS_PATH}")


if __name__ == "__main__":
    main()
