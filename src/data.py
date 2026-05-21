# -*- coding: utf-8 -*-
"""
Couche de données unifiée du dépôt.
=====================================================================
Théorie des Mesures de Risque — Mehdi Zaoui & Zakarya Sellami

Tous les scripts du dépôt passent par cette couche pour obtenir les
données, ce qui garantit qu'ils travaillent tous sur le MÊME jeu.

Stratégie « CSV figé + yfinance » :
    1. si data/prices.csv existe        -> on le charge (reproductible) ;
    2. sinon, si yfinance est installé  -> on télécharge et on met en
       cache dans data/prices.csv (le run suivant sera figé) ;
    3. sinon                            -> message explicite renvoyant
       vers la génération synthétique hors-ligne.

Usage en ligne de commande
--------------------------
    python src/data.py              # télécharge via yfinance et fige le CSV
    python src/data.py --synthetic  # fige un CSV synthétique (sans réseau)

Usage en import
---------------
    from data import load_prices, load_returns, TICKERS, WEIGHTS
    returns = load_returns()        # log-rendements alignés
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# --- Paramètres communs à tout le dépôt -----------------------------
TICKERS = ['^GSPC', '^FCHI', 'AAPL', 'TTE.PA']   # S&P 500, CAC 40, Apple, TotalEnergies
LABELS  = ['S&P 500', 'CAC 40', 'Apple', 'TotalEnergies']
WEIGHTS = np.full(len(TICKERS), 1.0 / len(TICKERS))   # portefeuille équipondéré
START   = '2008-01-01'
END     = '2025-12-31'
WINDOW  = 250                                          # fenêtre glissante (jours ouvrés)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, os.pardir, 'data')
CSV_PATH  = os.path.normpath(os.path.join(_DATA_DIR, 'prices.csv'))


# =====================================================================
#  Chargement des prix (cache CSV, sinon yfinance)
# =====================================================================
def load_prices(csv_path: str = CSV_PATH, force_download: bool = False) -> pd.DataFrame:
    """Renvoie un DataFrame de cours de clôture ajustés (colonnes = TICKERS).

    Charge le CSV figé s'il existe ; sinon télécharge via yfinance et le
    met en cache. Lève une erreur explicite si aucune des deux voies n'est
    disponible.
    """
    if os.path.exists(csv_path) and not force_download:
        prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        return prices[TICKERS].dropna()

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            f"Aucun cache trouvé ({csv_path}) et yfinance n'est pas installé.\n"
            "  -> 'pip install yfinance' puis 'python src/data.py'\n"
            "  -> ou, hors-ligne : 'python src/data.py --synthetic'"
        ) from exc

    print(f"Téléchargement via yfinance : {TICKERS} ({START} -> {END})...")
    prices = yf.download(TICKERS, start=START, end=END,
                         auto_adjust=True, progress=False)['Close']
    prices = prices[TICKERS].dropna()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    prices.to_csv(csv_path)
    print(f"Cache écrit : {csv_path}  ({len(prices)} lignes)")
    return prices


def load_returns(csv_path: str = CSV_PATH, force_download: bool = False) -> pd.DataFrame:
    """Log-rendements quotidiens (DataFrame T x d) à partir des prix."""
    prices = load_prices(csv_path, force_download=force_download)
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_losses(returns: pd.DataFrame | None = None,
                     weights: np.ndarray = WEIGHTS) -> pd.Series:
    """Pertes du portefeuille (X_t = -r_t) sous convention « pertes positives »."""
    if returns is None:
        returns = load_returns()
    return -(returns @ weights)


# =====================================================================
#  Génération synthétique hors-ligne (repli sans réseau)
# =====================================================================
def make_synthetic_prices(csv_path: str = CSV_PATH, seed: int = 2026) -> pd.DataFrame:
    """Écrit un CSV de cours SYNTHÉTIQUES calibrés sur le tableau 1 du mémoire
    (queues lourdes, asymétrie négative, clustering de volatilité, crises 2008
    et 2020 injectées). Permet de rejouer tout le pipeline SANS réseau.

    Avertissement : ce sont des données simulées, PAS les vraies séries de
    marché. À n'utiliser que pour tester l'exécution du code hors-ligne.
    """
    from scipy import stats
    rng = np.random.default_rng(seed)

    spec = {
        '^GSPC':  dict(mu=0.0003, sigma=0.0124, df=4.5, skew=-0.38, load=0.85),
        '^FCHI':  dict(mu=0.0001, sigma=0.0130, df=5.5, skew=-0.21, load=0.78),
        'AAPL':   dict(mu=0.0010, sigma=0.0188, df=6.5, skew=-0.15, load=0.65),
        'TTE.PA': dict(mu=0.0002, sigma=0.0165, df=4.8, skew=-0.28, load=0.70),
    }
    dates = pd.bdate_range('2008-01-03', '2025-12-30')
    n = len(dates)

    # Régimes de volatilité (calme / stress / hyper-stress) persistants
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        u = rng.random()
        if regime[i-1] == 0:
            regime[i] = 0 if u < 0.985 else 1
        elif regime[i-1] == 1:
            regime[i] = 0 if u < 0.04 else (1 if u < 0.97 else 2)
        else:
            regime[i] = 1 if u < 0.20 else 2

    def loc(d):
        return dates.get_loc(pd.Timestamp(d))
    i1, i2 = loc('2008-09-15'), loc('2009-04-30')
    regime[i1:i1+30] = 2; regime[i1+30:i2] = 1
    j1, j2 = loc('2020-02-20'), loc('2020-05-15')
    regime[j1:j1+25] = 2; regime[j1+25:j2] = 1
    vol_mult = np.where(regime == 0, 1.0, np.where(regime == 1, 2.2, 4.5))

    df_mkt = 5.5
    market = stats.t.rvs(df_mkt, size=n, random_state=rng) / np.sqrt(df_mkt / (df_mkt - 2))

    returns = pd.DataFrame(index=dates, columns=TICKERS, dtype=float)
    for tk, s in spec.items():
        idio = stats.t.rvs(s['df'], size=n, random_state=rng) / np.sqrt(s['df'] / (s['df'] - 2))
        raw = s['load'] * market + np.sqrt(1 - s['load']**2) * idio
        skewed = np.where(raw < 0, raw * (1 + abs(s['skew'])*0.5),
                          raw * (1 - abs(s['skew'])*0.3))
        returns[tk] = s['mu'] + s['sigma'] * vol_mult * skewed

    # Injection des séquences réelles des semaines de crise (tableaux 7 et 8)
    crises = {
        '2008-10-06': {'^GSPC':-0.0393,'^FCHI':-0.0947,'AAPL':+0.0110,'TTE.PA':-0.0933},
        '2008-10-07': {'^GSPC':-0.0591,'^FCHI':+0.0054,'AAPL':-0.0960,'TTE.PA':+0.0295},
        '2008-10-08': {'^GSPC':-0.0114,'^FCHI':-0.0651,'AAPL':+0.0070,'TTE.PA':-0.0743},
        '2008-10-09': {'^GSPC':-0.0792,'^FCHI':-0.0156,'AAPL':-0.0118,'TTE.PA':-0.0330},
        '2008-10-10': {'^GSPC':-0.0118,'^FCHI':-0.0805,'AAPL':+0.0869,'TTE.PA':-0.0800},
        '2020-03-09': {'^GSPC':-0.0790,'^FCHI':-0.0876,'AAPL':-0.0824,'TTE.PA':-0.1816},
        '2020-03-11': {'^GSPC':-0.0501,'^FCHI':-0.0057,'AAPL':-0.0353,'TTE.PA':-0.0176},
        '2020-03-12': {'^GSPC':-0.0999,'^FCHI':-0.1310,'AAPL':-0.1040,'TTE.PA':-0.1603},
        '2020-03-13': {'^GSPC':+0.0888,'^FCHI':+0.0182,'AAPL':+0.1132,'TTE.PA':-0.0135},
    }
    for d, vals in crises.items():
        if pd.Timestamp(d) in returns.index:
            for tk, v in vals.items():
                returns.loc[d, tk] = v

    # Log-rendements -> cours (base 100), pour rester homogène avec load_prices
    prices = 100.0 * np.exp(returns.cumsum())
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    prices.to_csv(csv_path)
    print(f"Cache SYNTHÉTIQUE écrit : {csv_path}  ({len(prices)} lignes)")
    return prices


# =====================================================================
#  CLI
# =====================================================================
if __name__ == '__main__':
    if '--synthetic' in sys.argv:
        make_synthetic_prices()
    else:
        load_prices(force_download=True)
    # contrôle rapide
    r = load_returns()
    print(f"Rendements chargés : {r.shape[0]} jours x {r.shape[1]} actifs "
          f"({r.index[0].date()} -> {r.index[-1].date()})")
