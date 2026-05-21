# -*- coding: utf-8 -*-
"""
Analyse — Allocation d'Euler des contributions au risque.
=====================================================================
Pour l'ES (positivement homogène), le théorème d'Euler fournit une
décomposition additive exacte des contributions par actif :

    rho_i = w_i * E[X_i | X_Pi > VaR_alpha(X_Pi)] ,  somme = ES_alpha(X_Pi).

On calcule ces contributions sur plusieurs sous-périodes contrastées pour
observer le déplacement de la concentration du risque en crise.

    python src/allocation_euler.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_returns, TICKERS, WEIGHTS
from risk_toolkit import euler_es_allocation


PERIODES = [
    ("Calme 2017",     '2017-01-01', '2017-12-31'),
    ("Crise 2008-09",  '2008-09-01', '2009-04-30'),
    ("COVID 2020",     '2020-02-15', '2020-05-31'),
    ("Récent 2023-25", '2023-01-01', '2025-12-31'),
]


def main(alpha=0.95):
    returns = load_returns()
    losses_per_asset = -returns          # perte par actif (convention pertes positives)

    print(f"=== Allocation d'Euler (ES {alpha*100:.0f}%) ===")
    for label, d1, d2 in PERIODES:
        sous = losses_per_asset.loc[d1:d2]
        contribs, total = euler_es_allocation(sous, WEIGHTS, alpha=alpha)
        if contribs is None:
            print(f"{label:16s} : (aucune observation de queue)")
            continue
        parts = "  ".join(f"{tk}={contribs[tk]/total*100:4.1f}%" for tk in TICKERS)
        print(f"{label:16s} ES={total*100:5.2f}%   {parts}")


if __name__ == "__main__":
    main()
