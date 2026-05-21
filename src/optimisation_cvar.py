# -*- coding: utf-8 -*-
"""
Analyse — Optimisation de portefeuille par minimisation de la CVaR.
=====================================================================
Le théorème de Rockafellar-Uryasev (2000) transforme la minimisation de
la CVaR en programme linéaire. On compare équipondéré / minimum-variance /
minimum-CVaR, calibrés in-sample (2008-2018) et évalués hors échantillon
(2019-2025).

    python src/optimisation_cvar.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_returns, TICKERS
from risk_toolkit import cvar_optimal_weights, markowitz_min_variance


def main(alpha=0.95, split='2019-01-01'):
    returns = load_returns()
    train_mask = returns.index < split
    R_train, R_test = returns[train_mask].values, returns[~train_mask].values

    w_eq = np.full(len(TICKERS), 1 / len(TICKERS))
    w_mv = markowitz_min_variance(R_train)
    w_cvar, _, _ = cvar_optimal_weights(R_train, alpha=alpha)

    print("Compositions optimales (in-sample 2008-2018) :")
    for nom, w in [('Équipondéré', w_eq), ('Min-variance', w_mv), ('Min-CVaR', w_cvar)]:
        print(f"  {nom:13s} : {dict(zip(TICKERS, np.round(w, 3)))}")

    print(f"\nRisque out-of-sample (depuis {split}) :")
    for nom, w in [('Équipondéré', w_eq), ('Min-variance', w_mv), ('Min-CVaR', w_cvar)]:
        loss = -(R_test @ w)
        v95 = np.quantile(loss, 0.95)
        e95 = loss[loss > v95].mean()
        print(f"  {nom:13s} : vol={loss.std()*100:.2f}%  "
              f"VaR95={v95*100:.2f}%  ES95={e95*100:.2f}%  "
              f"pire perte={loss.max()*100:.2f}%")


if __name__ == "__main__":
    main()
