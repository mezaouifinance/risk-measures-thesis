# -*- coding: utf-8 -*-
"""
Analyse — Copules : gaussienne contre t-Student.
=====================================================================
La copule gaussienne impose une indépendance asymptotique de queue,
tandis que la copule t-Student admet une dépendance de queue strictement
positive : d'où une ES du portefeuille plus élevée dans les niveaux
extrêmes (mécanisme des co-krachs, leçon des CDO de 2008).

    python src/copules.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_returns, TICKERS, WEIGHTS
from risk_toolkit import (fit_marginals_t, to_uniform, fit_gaussian_copula,
                          fit_t_copula, simulate_copula)


def main(n_sim=100_000):
    returns = load_returns()

    params_t = fit_marginals_t(returns)
    U = to_uniform(returns, params_t)

    R_gauss = fit_gaussian_copula(U)
    best_df, R_t = fit_t_copula(U)
    print(f"Copule t-Student : df optimal = {best_df}")

    sims_g = simulate_copula(R_gauss, n_sim, params_t, TICKERS, df_cop=None, seed=1)
    sims_t = simulate_copula(R_t, n_sim, params_t, TICKERS, df_cop=best_df, seed=2)
    losses_g = -(sims_g @ WEIGHTS)
    losses_t = -(sims_t @ WEIGHTS)

    print(f"\n{'alpha':>7s} {'ES gaussienne':>14s} {'ES t-Student':>14s} {'ratio':>7s}")
    for a in (0.95, 0.975, 0.99, 0.995):
        eg = losses_g[losses_g > np.quantile(losses_g, a)].mean()
        et = losses_t[losses_t > np.quantile(losses_t, a)].mean()
        print(f"{a:>7.3f} {eg*100:>13.2f}% {et*100:>13.2f}% {et/eg:>7.2f}")


if __name__ == "__main__":
    main()
