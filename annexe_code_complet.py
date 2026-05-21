# -*- coding: utf-8 -*-
"""
=====================================================================
ANNEXE A — CODE PYTHON COMMENTÉ
Théorie des Mesures de Risque — Mehdi Zaoui & Zakarya Sellami
=====================================================================

Ce fichier reproduit l'intégralité des résultats empiriques du mémoire.
Il est conçu pour être lu et exécuté séquentiellement : chaque bloc
s'appuie sur les variables définies dans les blocs précédents.

Sections :
    A.1  Téléchargement et préparation des données
    A.2  Estimateurs de la VaR et de l'ES (3 méthodes)
    A.3  VaR/ES glissantes et figure de l'évolution temporelle
    A.4  Backtesting de la VaR : Kupiec et Christoffersen
    A.5  Backtesting de l'ES : statistique d'Acerbi-Szekely
    A.6  Stress test : crises de 2008 et 2020
    A.7  Sensibilité du backtesting à la taille de fenêtre
    A.8  Copules : gaussienne contre t-Student
    A.9  Optimisation de portefeuille par minimisation de la CVaR
    A.10 Allocation Euler des contributions au risque

Prérequis :
    pip install numpy pandas scipy matplotlib yfinance

Temps d'exécution complet : quelques minutes sur une machine standard
(la section A.7, qui rejoue le backtesting pour 4 tailles de fenêtre,
est la plus longue).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.linalg import cholesky
import math

np.random.seed(42)        # reproductibilité globale


# =====================================================================
# A.1  TÉLÉCHARGEMENT ET PRÉPARATION DES DONNÉES
# =====================================================================
import os

TICKERS = ['^GSPC', '^FCHI', 'AAPL', 'TTE.PA']   # S&P 500, CAC 40, Apple, TotalEnergies
START   = '2008-01-01'
END     = '2025-12-31'
WINDOW  = 250                                     # fenêtre glissante (jours ouvrés)

# Stratégie « CSV figé + yfinance » : on charge le cache data/prices.csv
# s'il existe (run reproductible au chiffre près), sinon on télécharge via
# yfinance et on met en cache. Run hors-ligne : python src/data.py --synthetic
CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prices.csv')
if os.path.exists(CSV):
    prices = pd.read_csv(CSV, index_col=0, parse_dates=True)
else:
    import yfinance as yf
    prices = yf.download(TICKERS, start=START, end=END,
                         auto_adjust=True, progress=False)['Close']
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    prices.to_csv(CSV)
prices = prices[TICKERS].dropna()                 # alignement des calendriers

# Log-rendements puis pertes (convention « pertes positives »)
returns      = np.log(prices / prices.shift(1)).dropna()
weights      = np.full(len(TICKERS), 1.0 / len(TICKERS))   # équipondéré
port_returns = returns @ weights                  # rendement du portefeuille
port_losses  = -port_returns                      # X_t = -r_t

# Pertes par actif (utiles pour l'allocation Euler en A.10)
losses_per_asset = -returns

print(f"Période : {port_losses.index[0].date()} -> {port_losses.index[-1].date()}")
print(f"Nombre de jours : {len(port_losses)}")


# =====================================================================
# A.2  ESTIMATEURS DE LA VaR ET DE L'ES (3 MÉTHODES)
# =====================================================================
# Les trois méthodes du corps du mémoire.

def var_es_historique(losses, alpha):
    """VaR et ES par quantile empirique sur la fenêtre."""
    var = np.quantile(losses, alpha)
    es  = losses[losses > var].mean()
    return var, es


def var_es_parametrique(losses, alpha):
    """VaR et ES sous hypothèse gaussienne."""
    mu, sigma = losses.mean(), losses.std(ddof=1)
    z = stats.norm.ppf(alpha)
    var = mu + sigma * z
    es  = mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    return var, es


def var_es_montecarlo(losses, alpha, n_sim=100_000, seed=42):
    """VaR et ES par Monte Carlo sur une loi de Student ajustée
    par maximum de vraisemblance (capture la kurtosis élevée)."""
    df, loc, scale = stats.t.fit(losses)
    rng  = np.random.default_rng(seed)
    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sim, random_state=rng)
    var  = np.quantile(sims, alpha)
    es   = sims[sims > var].mean()
    return var, es


# --- Estimation à la dernière date sur la fenêtre de 250 jours -------
last = port_losses.iloc[-WINDOW:]
print("\nEstimations au", port_losses.index[-1].date())
for a in (0.95, 0.975):
    vh, eh = var_es_historique(last, a)
    vp, ep = var_es_parametrique(last, a)
    vm, em = var_es_montecarlo(last, a)
    print(f"\nalpha = {a}")
    print(f"  Historique    : VaR = {vh:.4f}  ES = {eh:.4f}")
    print(f"  Paramétrique  : VaR = {vp:.4f}  ES = {ep:.4f}")
    print(f"  Monte Carlo   : VaR = {vm:.4f}  ES = {em:.4f}")


# =====================================================================
# A.3  VaR/ES GLISSANTES ET FIGURE DE L'ÉVOLUTION TEMPORELLE
# =====================================================================

def backtest_glissant(losses, alpha, window, methode, mc_sims=10_000):
    """Calcule, pour chaque jour t > window, la VaR et l'ES prédites
    sur la fenêtre [t-window, t-1], ainsi que les exceptions.

    methode in {'historique', 'parametrique', 'montecarlo'}.
    """
    arr    = np.asarray(losses)
    n_test = len(arr) - window
    vars_  = np.empty(n_test)
    ess    = np.empty(n_test)
    for i in range(n_test):
        win = arr[i:i + window]
        if methode == 'historique':
            v, e = var_es_historique(win, alpha)
        elif methode == 'parametrique':
            v, e = var_es_parametrique(win, alpha)
        elif methode == 'montecarlo':
            v, e = var_es_montecarlo(win, alpha, n_sim=mc_sims, seed=i)
        else:
            raise ValueError(f"méthode inconnue : {methode}")
        vars_[i], ess[i] = v, e
    actuals    = arr[window:]
    exceptions = (actuals > vars_).astype(int)
    return vars_, ess, exceptions


# Convention de niveaux suivie dans tout le mémoire :
#   - la VaR est backtestée à 95 % (sections A.3 et A.4), niveau du corps
#     du mémoire ;
#   - l'ES est backtestée à 97,5 % (section A.5), conformément à la FRTB
#     (Bâle III/IV) : l'ES y est relevée à 97,5 % pour offrir une sévérité
#     comparable à une VaR à 99 %. Les deux niveaux coexistent donc à
#     dessein. On ne conserve ici que la VaR glissante (pour la figure) ;
#     les ES sont (re)calculées au bon niveau là où elles sont utilisées.
ALPHA      = 0.95
dates_test = port_losses.index[WINDOW:]

# VaR glissante à 95 % pour les trois méthodes du corps du mémoire
var_h, _, _ = backtest_glissant(port_losses, ALPHA, WINDOW, 'historique')
var_p, _, _ = backtest_glissant(port_losses, ALPHA, WINDOW, 'parametrique')
var_m, _, _ = backtest_glissant(port_losses, ALPHA, WINDOW, 'montecarlo')

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(dates_test, var_h * 100, label='Historique',   lw=0.9)
ax.plot(dates_test, var_p * 100, label='Paramétrique', lw=0.9)
ax.plot(dates_test, var_m * 100, label='MC Student',   lw=0.9)
ax.set_ylabel(r'VaR$_{95\%}$ (%)')
ax.set_title('VaR glissante sur fenêtre de 250 jours')
ax.legend(loc='upper right'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('var_glissante.pdf'); plt.close()


# =====================================================================
# A.4  BACKTESTING DE LA VaR : KUPIEC ET CHRISTOFFERSEN
# =====================================================================

def kupiec(N, T, alpha):
    """Test de couverture inconditionnelle (1 ddl). Loi chi2(1) sous H0."""
    if N == 0 or N == T:
        return np.nan, np.nan
    p_hat, p0 = N / T, 1 - alpha
    LR = -2 * (N * np.log(p0 / p_hat)
               + (T - N) * np.log((1 - p0) / (1 - p_hat)))
    return LR, 1 - stats.chi2.cdf(LR, df=1)


def christoffersen(exc, alpha):
    """Test conjoint couverture + indépendance (2 ddl). Loi chi2(2)."""
    exc = np.asarray(exc).astype(int)
    T, N = len(exc), exc.sum()
    n00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())
    n01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())
    n10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())
    n11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)
    eps = 1e-12
    log_indep = ((n00 + n10) * np.log(max(1 - pi, eps))
                 + (n01 + n11) * np.log(max(pi, eps)))
    log_full  = (n00 * np.log(max(1 - pi01, eps))
                 + n01 * np.log(max(pi01, eps))
                 + n10 * np.log(max(1 - pi11, eps))
                 + n11 * np.log(max(pi11, eps)))
    LR_ind = -2 * (log_indep - log_full)
    LR_uc, _ = kupiec(N, T, alpha)
    LR_cc = LR_uc + LR_ind
    return LR_cc, 1 - stats.chi2.cdf(LR_cc, df=2)


# Backtesting des trois méthodes
print(f"\n=== Backtesting VaR à {ALPHA*100:.0f}% ===")
methodes = {'Historique': 'historique', 'Paramétrique': 'parametrique',
            'MC Student': 'montecarlo'}
for nom, code in methodes.items():
    _, _, exc = backtest_glissant(port_losses, ALPHA, WINDOW, code)
    N, T_test = exc.sum(), len(exc)
    LR_uc, p_uc = kupiec(N, T_test, ALPHA)
    LR_cc, p_cc = christoffersen(exc, ALPHA)
    print(f"\n{nom} : N = {N} ({N/T_test*100:.2f}%)")
    print(f"  Kupiec        : LR = {LR_uc:5.2f}  p = {p_uc:.4f}")
    print(f"  Christoffersen: LR = {LR_cc:5.2f}  p = {p_cc:.4f}")


# =====================================================================
# A.5  BACKTESTING DE L'ES : STATISTIQUE D'ACERBI-SZEKELY
# =====================================================================

def acerbi_szekely_z2(losses, vars_, ess, alpha):
    """Statistique Z2 d'Acerbi-Szekely (2014). E[Z2] = 0 sous le
    modèle correct ; Z2 > 0 signale une sous-estimation des pertes
    en queue par l'ES."""
    mask = losses > vars_
    if mask.sum() == 0:
        return np.nan
    ratio = (losses[mask] / ess[mask]).sum()
    return ratio / (len(losses) * (1 - alpha)) - 1


def z2_pvalue_bootstrap(losses, vars_, ess, alpha, n_boot=1000, seed=42):
    """p-valeur bilatérale par bootstrap paramétrique sous une loi
    de Student ajustée sur l'ensemble des pertes."""
    z_obs = acerbi_szekely_z2(losses, vars_, ess, alpha)
    rng = np.random.default_rng(seed)
    df, loc, scale = stats.t.fit(losses)
    z_sims = np.empty(n_boot)
    for b in range(n_boot):
        sim = stats.t.rvs(df, loc=loc, scale=scale, size=len(losses),
                          random_state=rng)
        z_sims[b] = acerbi_szekely_z2(sim, vars_, ess, alpha)
    p = np.mean(np.abs(z_sims) >= np.abs(z_obs))
    return z_obs, p


# Niveau ES = 97,5 % (et non 95 %) : c'est la convention FRTB rappelée
# plus haut. On RECALCULE donc la VaR/ES glissantes à ce niveau ci-dessous,
# au lieu de réutiliser les sorties à 95 % de la section A.3.
ALPHA_ES = 0.975
actuals  = np.asarray(port_losses)[WINDOW:]   # pertes réalisées alignées sur la période de test
print(f"\n=== Backtesting ES à {ALPHA_ES*100:.1f}% (Acerbi-Szekely) ===")
for nom, code in methodes.items():
    v, e, _ = backtest_glissant(port_losses, ALPHA_ES, WINDOW, code)
    z, p = z2_pvalue_bootstrap(actuals, v, e, ALPHA_ES, n_boot=1000)
    print(f"{nom:14s} Z2 = {z:+.3f}  p = {p:.3f}")


# =====================================================================
# A.6  STRESS TEST : CRISES DE 2008 ET 2020
# =====================================================================

def stress_window(returns, weights, start, end):
    """Rendements et pertes du portefeuille sur une fenêtre donnée."""
    period = returns.loc[start:end]
    losses = -(period @ weights)
    return period, losses


def measure_at(date, alpha=0.975, window=WINDOW):
    """VaR/ES estimées la veille de `date` par les trois méthodes
    (fenêtre tronquée si l'historique disponible est plus court)."""
    pos = port_losses.index.get_indexer([date], method='ffill')[0]
    win = port_losses.iloc[max(0, pos - window):pos]
    return {
        'hist':    var_es_historique(win, alpha),
        'param':   var_es_parametrique(win, alpha),
        'student': var_es_montecarlo(win, alpha),
    }


for label, (d1, d2), pire in [
    ("Crise 2008", ('2008-10-06', '2008-10-10'), '2008-10-06'),
    ("Crise COVID 2020", ('2020-03-09', '2020-03-13'), '2020-03-12'),
]:
    period, losses = stress_window(returns, weights, d1, d2)
    cumul = (np.exp(period.sum() @ weights) - 1) * 100
    print(f"\n{label} : perte cumulée semaine = {cumul:.2f}%")
    print(f"  Pire journée : {losses.idxmax().date()}, "
          f"perte = {losses.max()*100:.2f}%")
    m = measure_at(pire)
    for k, (v, e) in m.items():
        print(f"  {k:8s} veille : VaR = {v*100:.2f}%  ES = {e*100:.2f}%")


# =====================================================================
# A.7  SENSIBILITÉ DU BACKTESTING À LA TAILLE DE FENÊTRE
# =====================================================================
# Rejoue le backtesting pour plusieurs tailles de fenêtre afin
# d'évaluer l'arbitrage réactivité / stabilité (cf. section 5).

print("\n=== A.7  Sensibilité à la fenêtre (alpha = 95%) ===")
fenetres = [60, 125, 250, 500]
# On omet Monte Carlo ici (coûteux car re-fit Student à chaque fenêtre).
methodes_sens = {'Historique': 'historique',
                 'Paramétrique': 'parametrique'}
print(f"{'Fenêtre':>8s}", *[f"{m:>14s}" for m in methodes_sens])
for w in fenetres:
    taux = []
    for code in methodes_sens.values():
        _, _, exc = backtest_glissant(port_losses, 0.95, w, code)
        taux.append(exc.sum() / len(exc) * 100)
    print(f"{w:>8d}", *[f"{t:>13.2f}%" for t in taux])


# =====================================================================
# A.8  COPULES : GAUSSIENNE CONTRE t-STUDENT
# =====================================================================
# Modélise la dépendance jointe des quatre actifs. La copule
# gaussienne sous-estime les co-extrêmes ; la copule t-Student
# possède une dépendance de queue strictement positive.

print("\n=== A.8  Copules gaussienne vs t-Student ===")

# 1) Marginales : loi de Student par actif
params_t = {c: stats.t.fit(returns[c]) for c in TICKERS}

# 2) Transformation en pseudo-observations uniformes (intégrale de proba)
U = np.column_stack([
    stats.t.cdf(returns[c], *params_t[c]) for c in TICKERS
])
U = np.clip(U, 1e-6, 1 - 1e-6)

# 3) Copule gaussienne : corrélation des scores normaux
Z_g     = stats.norm.ppf(U)
R_gauss = np.corrcoef(Z_g, rowvar=False)

# 4) Copule t-Student : (R, df) par profilage de vraisemblance sur df
def loglik_copule_t(df_, U):
    """Log-vraisemblance d'une copule t-Student de paramètre df_."""
    Z = stats.t.ppf(U, df_)
    R = np.corrcoef(Z, rowvar=False)
    Rinv = np.linalg.inv(R)
    _, logdetR = np.linalg.slogdet(R)
    d = R.shape[0]
    quad = np.einsum('ni,ij,nj->n', Z, Rinv, Z)
    cst = (math.lgamma((df_ + d) / 2) - math.lgamma(df_ / 2)
           - d * (math.lgamma((df_ + 1) / 2) - math.lgamma(df_ / 2)))
    ll = (-0.5 * logdetR + cst
          - 0.5 * (df_ + d) * np.log(1 + quad / df_)
          + 0.5 * (df_ + 1) * np.log(1 + Z**2 / df_).sum(axis=1))
    return ll.sum(), R

best_df, R_t, best_ll = None, None, -np.inf
for df_test in (3, 4, 5, 6, 8, 10, 15, 20, 30):
    ll, R = loglik_copule_t(df_test, U)
    if ll > best_ll:
        best_ll, best_df, R_t = ll, df_test, R
print(f"Copule t-Student : df optimal = {best_df}")

# 5) Simulation jointe sous chaque copule + marginales Student
def simule_copule(R, n_sim, params, tickers, df_cop=None, seed=1):
    """Simule des rendements joints. Si df_cop est None : copule
    gaussienne ; sinon : copule t-Student de paramètre df_cop."""
    rng = np.random.default_rng(seed)
    d = R.shape[0]
    L = cholesky(R, lower=True)
    Z = rng.standard_normal((n_sim, d)) @ L.T
    if df_cop is None:                      # copule gaussienne
        Uc = stats.norm.cdf(Z)
    else:                                   # copule t-Student
        W  = rng.chisquare(df_cop, n_sim) / df_cop
        Uc = stats.t.cdf(Z / np.sqrt(W[:, None]), df_cop)
    Uc = np.clip(Uc, 1e-9, 1 - 1e-9)
    out = np.empty_like(Uc)
    for i, tk in enumerate(tickers):
        out[:, i] = stats.t.ppf(Uc[:, i], *params[tk])
    return np.clip(out, -0.5, 1.0)          # borne les rendements aberrants

N_SIM    = 100_000
sims_g   = simule_copule(R_gauss, N_SIM, params_t, TICKERS, df_cop=None,    seed=1)
sims_t   = simule_copule(R_t,     N_SIM, params_t, TICKERS, df_cop=best_df, seed=2)
losses_g = -(sims_g @ weights)
losses_t = -(sims_t @ weights)

# 6) VaR/ES du portefeuille sous chaque copule
print(f"{'alpha':>7s} {'ES gaussienne':>14s} {'ES t-Student':>14s} {'ratio':>7s}")
for a in (0.95, 0.975, 0.99, 0.995):
    eg = losses_g[losses_g > np.quantile(losses_g, a)].mean()
    et = losses_t[losses_t > np.quantile(losses_t, a)].mean()
    print(f"{a:>7.3f} {eg*100:>13.2f}% {et*100:>13.2f}% {et/eg:>7.2f}")


# =====================================================================
# A.9  OPTIMISATION DE PORTEFEUILLE PAR MINIMISATION DE LA CVaR
# =====================================================================
# Programme linéaire de Rockafellar-Uryasev (2000). On compare
# équipondéré / minimum-variance / minimum-CVaR, calibrés in-sample
# (2008-2018) et évalués out-of-sample (2019-2025).

print("\n=== A.9  Optimisation CVaR (Rockafellar-Uryasev) ===")

def poids_min_cvar(R_train, alpha=0.95, w_min=0.0, w_max=1.0):
    """Minimise la CVaR par programme linéaire :

        min   v + 1/((1-alpha) T) * sum_t u_t
        s.c.  u_t >= -r_t . w - v ,  u_t >= 0
              sum_i w_i = 1 ,  w_min <= w_i <= w_max

    Variables : [w (d), v (1), u (T)]."""
    R = np.asarray(R_train)
    T, d = R.shape
    nvars = d + 1 + T
    c = np.zeros(nvars)
    c[d] = 1.0
    c[d+1:] = 1.0 / ((1 - alpha) * T)
    A_ub = np.zeros((T, nvars))
    A_ub[:, :d] = -R
    A_ub[:, d]  = -1
    A_ub[np.arange(T), d + 1 + np.arange(T)] = -1
    b_ub = np.zeros(T)
    A_eq = np.zeros((1, nvars)); A_eq[0, :d] = 1.0
    b_eq = np.array([1.0])
    bounds = [(w_min, w_max)] * d + [(None, None)] + [(0, None)] * T
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
    return res.x[:d]


def poids_min_variance(R_train, w_min=0.0, w_max=1.0):
    """Portefeuille Markowitz minimum-variance.

    On met la matrice de covariance à l'échelle (x 1e4) pour le
    conditionnement numérique : les variances de rendements quotidiens
    sont de l'ordre de 1e-4, ce qui fait que l'optimiseur SLSQP croit
    avoir convergé dès le point de départ. La mise à l'échelle ne
    change pas l'argmin."""
    Sigma = np.cov(R_train, rowvar=False) * 1e4
    d = R_train.shape[1]
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = optimize.minimize(lambda w: w @ Sigma @ w, x0=np.full(d, 1/d),
                            method='SLSQP', bounds=[(w_min, w_max)] * d,
                            constraints=cons, options={'ftol': 1e-12, 'maxiter': 500})
    return res.x


masque_train = returns.index < '2019-01-01'
R_train, R_test = returns[masque_train].values, returns[~masque_train].values

w_eq   = np.full(len(TICKERS), 1/len(TICKERS))
w_mv   = poids_min_variance(R_train)
w_cvar = poids_min_cvar(R_train, alpha=0.95)

print("Compositions optimales :")
for nom, w in [('Équipondéré', w_eq), ('Min-variance', w_mv), ('Min-CVaR', w_cvar)]:
    print(f"  {nom:13s} : {dict(zip(TICKERS, np.round(w, 3)))}")

print("\nRisque out-of-sample (2019-2025) :")
for nom, w in [('Équipondéré', w_eq), ('Min-variance', w_mv), ('Min-CVaR', w_cvar)]:
    losses_oos = -(R_test @ w)
    v95 = np.quantile(losses_oos, 0.95)
    e95 = losses_oos[losses_oos > v95].mean()
    print(f"  {nom:13s} : vol = {losses_oos.std()*100:.2f}%  "
          f"VaR95 = {v95*100:.2f}%  ES95 = {e95*100:.2f}%  "
          f"pire perte = {losses_oos.max()*100:.2f}%")


# =====================================================================
# A.10  ALLOCATION EULER DES CONTRIBUTIONS AU RISQUE
# =====================================================================
# Décompose l'ES du portefeuille en contributions par actif :
#     rho_i = w_i * E[X_i | X_Pi > VaR_alpha(X_Pi)] ,  somme = ES.

print("\n=== A.10  Allocation Euler (ES 95%) ===")

def allocation_euler_es(losses_df, weights, alpha=0.95):
    """Contributions Euler à l'ES. Renvoie un dict {actif: contribution}
    et le total (= ES du portefeuille)."""
    L = np.asarray(losses_df)
    port = L @ weights
    var  = np.quantile(port, alpha)
    mask = port > var
    contribs = {tk: weights[i] * L[mask, i].mean()
                for i, tk in enumerate(losses_df.columns)}
    return contribs, sum(contribs.values())


periodes = [
    ("Calme 2017",     '2017-01-01', '2017-12-31'),
    ("Crise 2008-09",  '2008-09-01', '2009-04-30'),
    ("COVID 2020",     '2020-02-15', '2020-05-31'),
    ("Récent 2023-25", '2023-01-01', '2025-12-31'),
]
for label, d1, d2 in periodes:
    sous = losses_per_asset.loc[d1:d2]
    contribs, total = allocation_euler_es(sous, weights, alpha=0.95)
    parts = "  ".join(f"{tk}={contribs[tk]/total*100:4.1f}%" for tk in TICKERS)
    print(f"{label:16s} ES={total*100:5.2f}%   {parts}")

print("\nTous les résultats du mémoire ont été reproduits.")
