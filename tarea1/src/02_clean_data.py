"""
Paso 1 (cont.): Limpieza y estructuración de datos.

Toma el CSV crudo de yfinance (con encabezado MultiIndex: Price/Ticker),
se queda con fecha + precio de cierre, ordena cronológicamente y deja
la serie lista para el análisis (niveles, log-niveles, log-retornos).

Manejo de huecos: NO se rellenan ni se reindexa a un calendario continuo.
Un futuro cotiza solo en días hábiles de mercado, así que fines de semana
y feriados no son "datos faltantes" sino ausencia esperada de cotización.
Solo se reportan huecos "anormales" (> 5 días corridos entre observaciones
consecutivas), que podrían indicar un feriado largo o un corte real del
proveedor de datos, para mencionarlos como limitación en el informe.
"""

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/cotton_ct_f_raw.csv"
CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
GAP_THRESHOLD_DAYS = 5


def main():
    raw = pd.read_csv(RAW_PATH, header=[0, 1], index_col=0, parse_dates=True)
    raw.index.name = "date"

    close = raw[("Close", "CT=F")].rename("close")
    df = close.to_frame().sort_index()

    # elimina filas sin precio (no debería haber, pero se valida igual)
    n_before = len(df)
    df = df.dropna(subset=["close"])
    if len(df) < n_before:
        print(f"Se eliminaron {n_before - len(df)} filas sin precio de cierre.")

    # revisión de huecos anormales entre días de cotización consecutivos
    gaps = df.index.to_series().diff().dt.days
    abnormal = gaps[gaps > GAP_THRESHOLD_DAYS]
    if not abnormal.empty:
        print(f"Huecos > {GAP_THRESHOLD_DAYS} días corridos detectados ({len(abnormal)}):")
        for dt, gap in abnormal.items():
            print(f"  {dt.date()}: {int(gap)} días desde la observación anterior")
    else:
        print(f"Sin huecos mayores a {GAP_THRESHOLD_DAYS} días corridos.")

    # columnas derivadas usadas en pasos posteriores
    df["log_close"] = np.log(df["close"])
    df["log_return"] = df["log_close"].diff()

    df.to_csv(CLEAN_PATH)
    print(f"Guardado: {CLEAN_PATH} ({len(df)} filas, {df.index.min().date()} a {df.index.max().date()})")


if __name__ == "__main__":
    main()
