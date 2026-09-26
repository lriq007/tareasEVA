"""
Paso 6: Estimación de modelos (confirmado por el usuario tras ver ACF/PACF).

Serie usada: retornos (log-diferencias), la serie estacionaria según ADF.

Modelos estimados:
1. ARIMA(0,0,0) con constante -- modelo base de "ruido blanco con media",
   sirve como punto de comparación mínimo dado que ACF/PACF no mostraron
   estructura clara.
2. AR(1) con constante -- el leve repunte visible en lag 1 aparece tanto
   en ACF como en PACF con una magnitud casi idéntica (~0.073), lo que en
   términos teóricos es ambiguo entre AR(1) y MA(1) cuando el coeficiente
   es tan chico (para |phi|~0.07 un AR(1) y un MA(1) son casi
   indistinguibles en la práctica). Se elige AR(1) porque el criterio
   clásico de identificación -- "PACF con un solo spike en lag 1 y luego
   silencio" -- es exactamente la firma de un AR(1); un MA(1) habría sido
   una alternativa igualmente defendible con esta evidencia.

Este script NO decide cuál modelo es mejor: solo deja ambos resultados
(coeficientes, error estándar, p-valor, AIC, BIC) listos para que el
usuario interprete el trade-off. La validación de residuos (paso 7) se
corre en un script aparte, después de que el usuario confirme el modelo
ganador.
"""

import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*no associated frequency.*")

CLEAN_PATH = "data/clean/cotton_ct_f_clean.csv"
COEF_PATH = "results/model_coefficients.csv"
COMPARISON_PATH = "results/model_comparison.csv"
ALPHA = 0.05

MODEL_SPECS = [
    {"label": "ARIMA(0,0,0) con constante", "order": (0, 0, 0)},
    {"label": "AR(1) con constante", "order": (1, 0, 0)},
]


def fit_and_summarize(returns, label, order):
    model = ARIMA(returns, order=order, trend="c")
    res = model.fit()

    # sigma2 es la varianza del error, no un coeficiente del modelo (AR/MA/const):
    # su test contra 0 no es económicamente interpretable como "significancia",
    # así que se muestra pero se excluye del conteo de coeficientes significativos.
    coef_rows = []
    for name in res.params.index:
        is_variance_param = name == "sigma2"
        coef_rows.append(
            {
                "modelo": label,
                "coeficiente": name,
                "estimacion": res.params[name],
                "error_estandar": res.bse[name],
                "p_value": res.pvalues[name],
                "significativo_5%": (bool(res.pvalues[name] < ALPHA) if not is_variance_param else None),
            }
        )

    n_sig = sum(1 for row in coef_rows if row["significativo_5%"])
    n_structural = sum(1 for row in coef_rows if row["coeficiente"] != "sigma2")
    comparison_row = {
        "modelo": label,
        "orden_(p,d,q)": str(order),
        "aic": res.aic,
        "bic": res.bic,
        "n_coeficientes_(sin_sigma2)": n_structural,
        "n_coeficientes_significativos_5%": n_sig,
    }
    return res, coef_rows, comparison_row


def main():
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)
    returns = df["log_return"].dropna()

    all_coef_rows = []
    all_comparison_rows = []

    for spec in MODEL_SPECS:
        res, coef_rows, comparison_row = fit_and_summarize(
            returns, spec["label"], spec["order"]
        )
        all_coef_rows.extend(coef_rows)
        all_comparison_rows.append(comparison_row)

        print(f"\n=== {spec['label']} ===")
        print(res.summary())

    coef_df = pd.DataFrame(all_coef_rows)
    comparison_df = pd.DataFrame(all_comparison_rows)

    coef_df.to_csv(COEF_PATH, index=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    print("\n\n=== Tabla comparativa (sin conclusión -- a interpretar por el usuario) ===")
    print(comparison_df.to_string(index=False))
    print(f"\nGuardado: {COEF_PATH}, {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
