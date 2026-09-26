"""
Paso 1: Extracción de datos.

Descarga el histórico diario del futuro de algodón #2 (CT=F, Nueva York)
desde Yahoo Finance, para los últimos 15 años, y lo guarda sin modificar
en data/raw/. La limpieza (orden cronológico, columnas finales, manejo
de huecos) se hace en 02_clean_data.py, separada de la descarga para que
siempre quede una copia cruda intacta.
"""

from datetime import date, timedelta

import yfinance as yf

TICKER = "CT=F"
YEARS = 15
OUTPUT_PATH = "data/raw/cotton_ct_f_raw.csv"


def main():
    end = date.today()
    start = end - timedelta(days=365 * YEARS)

    print(f"Descargando {TICKER} desde {start} hasta {end}...")
    df = yf.download(TICKER, start=start, end=end, interval="1d", auto_adjust=False)

    if df.empty:
        raise RuntimeError(
            f"yfinance no devolvió datos para {TICKER} en el rango {start} - {end}."
        )

    df.to_csv(OUTPUT_PATH)
    print(f"Guardado: {OUTPUT_PATH} ({len(df)} filas, {df.index.min().date()} a {df.index.max().date()})")


if __name__ == "__main__":
    main()
