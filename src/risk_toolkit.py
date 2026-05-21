# -*- coding: utf-8 -*-
"""
Bibliothèque de mesures de risque — Théorie des Mesures de Risque
=====================================================================
Mehdi Zaoui & Zakarya Sellami

Fonctions réutilisables (importables) couvrant :
1. Estimateurs VaR/ES : historique, paramétrique gaussien, Monte Carlo Student
2. Tests de backtesting : Kupiec, Christoffersen, Acerbi-Szekely Z2
3. Modélisation par copule (gaussienne vs t-Student)
4. Optimisation CVaR (programme linéaire de Rockafellar-Uryasev)
5. Allocation d'Euler des contributions au risque

Conventions identiques au mémoire :
- pertes = -log-rendements (perte > 0) ;
- niveaux alpha dans (0,1) (typiquement 0.95, 0.975, 0.99) ;
- fenêtre par défaut de 250 jours (un an ouvré).
"""
import math
import numpy as np
from scipy import stats, optimize
from scipy.linalg import cholesky


# ============================================================
# 1. ESTIMATEURS DE BASE (trois méthodes du corps du mémoire)
# ============================================================
def var_es_historique(losses, alpha):
    """VaR et ES historiques (quantile empirique)."""
    losses = np.asarray(losses)
    var = np.quantile(losses, alpha)
    tail = losses[losses > var]
    es = tail.mean() if len(tail) > 0 else var
    return var, es


def var_es_parametrique(losses, alpha):
    """VaR et ES sous hypothèse gaussienne."""
    mu, sigma = np.mean(losses), np.std(losses, ddof=1)
    z = stats.norm.ppf(alpha)
    var = mu + sigma * z
    es = mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    return var, es


def var_es_montecarlo(losses, alpha, n_sim=100_000, seed=42):
    """VaR et ES par Monte Carlo sur une loi de Student ajustée par
    maximum de vraisemblance (capture la kurtosis élevée)."""
    df, loc, scale = stats.t.fit(losses)
    rng = np.random.default_rng(seed)
    sims = stats.t.rvs(df, loc=loc, scale=scale, size=n_sim, random_state=rng)
    var = np.quantile(sims, alpha)
    es = sims[sims > var].mean()
    return var, es


# ============================================================
# 2. TESTS DE BACKTESTING (Kupiec, Christoffersen, Acerbi-Szekely)
# ============================================================
def kupiec_test(N, T, alpha):
    """Test de Kupiec (couverture inconditionnelle). Loi chi2(1) sous H0."""
    if N == 0 or N == T:
        return np.nan, np.nan
    p_hat, p0 = N / T, 1 - alpha
    LR = -2 * (N * np.log(p0 / p_hat) +
               (T - N) * np.log((1 - p0) / (1 - p_hat)))
    return LR, 1 - stats.chi2.cdf(LR, df=1)


def christoffersen_test(exceptions, alpha):
    """Test de Christoffersen (couverture + indépendance). Loi chi2(2)."""
    exc = np.asarray(exceptions).astype(int)
    T, N = len(exc), exc.sum()
    n00 = int(((exc[:-1] == 0) & (exc[1:] == 0)).sum())
    n01 = int(((exc[:-1] == 0) & (exc[1:] == 1)).sum())
    n10 = int(((exc[:-1] == 1) & (exc[1:] == 0)).sum())
    n11 = int(((exc[:-1] == 1) & (exc[1:] == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    eps = 1e-12
    log_indep = ((n00 + n10) * np.log(max(1 - pi, eps)) +
                 (n01 + n11) * np.log(max(pi, eps)))
    log_full = (n00 * np.log(max(1 - pi01, eps)) +
                n01 * np.log(max(pi01, eps)) +
                n10 * np.log(max(1 - pi11, eps)) +
                n11 * np.log(max(pi11, eps)))
    LR_ind = -2 * (log_indep - log_full)
    LR_uc, _ = kupiec_test(N, T, alpha)
    LR_cc = LR_uc + LR_ind
    return LR_cc, 1 - stats.chi2.cdf(LR_cc, df=2)


def acerbi_szekely_z2(losses, vars_, ess, alpha):
    """Statistique Z2 d'Acerbi-Szekely (2014). E[Z2] = 0 sous le modèle
    correct ; Z2 > 0 signale une sous-estimation des pertes de queue."""
    losses = np.asarray(losses); vars_ = np.asarray(vars_); ess = np.asarray(ess)
    mask = losses > vars_
    if mask.sum() == 0:
        return np.nan
    ratio = (losses[mask] / ess[mask]).sum()
    return ratio / (len(losses) * (1 - alpha)) - 1


# ============================================================
# 3. BACKTEST SUR FENÊTRE GLISSANTE
# ============================================================
def backtest_sliding(losses, alpha, window, method='historique',
                     mc_sims=10_000, mc_seed=42):
    """Calcule, pour chaque t > window, la VaR/ES prédites sur la fenêtre
    [t-window, t-1], plus les exceptions, alignées sur losses[window:].

    method in {'historique', 'parametrique', 'montecarlo'}.
    """
    arr = np.asarray(losses)
    n_test = len(arr) - window
    vars_ = np.empty(n_test); ess = np.empty(n_test)
    for i in range(n_test):
        win = arr[i:i + window]
        if method == 'historique':
            v, e = var_es_historique(win, alpha)
        elif method == 'parametrique':
            v, e = var_es_parametrique(win, alpha)
        elif method == 'montecarlo':
            v, e = var_es_montecarlo(win, alpha, n_sim=mc_sims, seed=mc_seed + i)
        else:
            raise ValueError(f"méthode inconnue : {method}")
        vars_[i] = v; ess[i] = e
    actuals = arr[window:]
    exceptions = (actuals > vars_).astype(int)
    return vars_, ess, exceptions


# ============================================================
# 4. COPULES (gaussienne vs t-Student)
# ============================================================
def fit_marginals_t(returns_df):
    """Ajuste une loi de Student par actif : {ticker: (df, loc, scale)}."""
    return {c: stats.t.fit(returns_df[c].dropna()) for c in returns_df.columns}


def to_uniform(returns_df, params_t):
    """Pseudo-observations uniformes via la CDF de Student (intégrale de proba)."""
    cols = list(returns_df.columns)
    U = np.column_stack([
        stats.t.cdf(returns_df[c].values, *params_t[c]) for c in cols
    ])
    return np.clip(U, 1e-6, 1 - 1e-6)


def fit_gaussian_copula(U):
    """Matrice de corrélation d'une copule gaussienne (scores normaux)."""
    Z = stats.norm.ppf(U)
    return np.corrcoef(Z, rowvar=False)


def _loglik_t_copula(df_, U):
    """Log-vraisemblance d'une copule t-Student de paramètre df_ (forme exacte).
    Renvoie (log-vraisemblance, matrice de corrélation)."""
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


def fit_t_copula(U, df_grid=(3, 4, 5, 6, 8, 10, 15, 20, 30)):
    """Estime (df, R) d'une copule t-Student par profilage de la
    log-vraisemblance sur une grille de degrés de liberté."""
    best_df, best_R, best_ll = None, None, -np.inf
    for df_ in df_grid:
        ll, R = _loglik_t_copula(df_, U)
        if ll > best_ll:
            best_ll, best_df, best_R = ll, df_, R
    return best_df, best_R


def simulate_copula(R, n_sim, params_t, tickers, df_cop=None, seed=1):
    """Simule des rendements joints. df_cop=None -> copule gaussienne ;
    sinon copule t-Student de paramètre df_cop. Marginales : Student ajustées."""
    rng = np.random.default_rng(seed)
    d = R.shape[0]
    L = cholesky(R, lower=True)
    Z = rng.standard_normal((n_sim, d)) @ L.T
    if df_cop is None:
        Uc = stats.norm.cdf(Z)
    else:
        W = rng.chisquare(df_cop, n_sim) / df_cop
        Uc = stats.t.cdf(Z / np.sqrt(W[:, None]), df_cop)
    Uc = np.clip(Uc, 1e-9, 1 - 1e-9)
    out = np.empty_like(Uc)
    for i, tk in enumerate(tickers):
        out[:, i] = stats.t.ppf(Uc[:, i], *params_t[tk])
    return np.clip(out, -0.5, 1.0)


# ============================================================
# 5. OPTIMISATION CVaR (Rockafellar-Uryasev) + min-variance
# ============================================================
def cvar_optimal_weights(returns_array, alpha=0.95, w_min=0.0, w_max=1.0):
    """Minimise la CVaR par programme linéaire :

        min   v + 1/((1-alpha) T) * somme_t u_t
        s.c.  u_t >= -r_t . w - v ,  u_t >= 0
              somme_i w_i = 1 ,  w_min <= w_i <= w_max

    Variables : [w (d), v (1), u (T)]. À l'optimum v = VaR et la valeur
    optimale vaut l'ES. Renvoie (poids, v=VaR, valeur=ES)."""
    R = np.asarray(returns_array)
    T, d = R.shape
    nvars = d + 1 + T
    c = np.zeros(nvars)
    c[d] = 1.0
    c[d + 1:] = 1.0 / ((1 - alpha) * T)
    A_ub = np.zeros((T, nvars))
    A_ub[:, :d] = -R
    A_ub[:, d] = -1
    A_ub[np.arange(T), d + 1 + np.arange(T)] = -1
    b_ub = np.zeros(T)
    A_eq = np.zeros((1, nvars)); A_eq[0, :d] = 1.0
    b_eq = np.array([1.0])
    bounds = [(w_min, w_max)] * d + [(None, None)] + [(0, None)] * T
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"linprog a échoué : {res.message}")
    return res.x[:d], res.x[d], res.fun


def markowitz_min_variance(returns_array, w_min=0.0, w_max=1.0):
    """Portefeuille Markowitz minimum-variance : min w' Sigma w, somme w = 1.

    La covariance est mise à l'échelle (x 1e4) pour le conditionnement
    numérique : les variances de rendements quotidiens sont de l'ordre de
    1e-4, ce qui ferait croire à l'optimiseur SLSQP qu'il a convergé dès le
    point de départ. La mise à l'échelle ne change pas l'argmin."""
    R = np.asarray(returns_array)
    Sigma = np.cov(R, rowvar=False) * 1e4
    d = R.shape[1]
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    res = optimize.minimize(lambda w: w @ Sigma @ w, x0=np.full(d, 1 / d),
                            method='SLSQP', bounds=[(w_min, w_max)] * d,
                            constraints=cons,
                            options={'ftol': 1e-12, 'maxiter': 500})
    return res.x


# ============================================================
# 6. ALLOCATION D'EULER DES CONTRIBUTIONS AU RISQUE (ES)
# ============================================================
def euler_es_allocation(losses_per_asset, weights, alpha=0.95):
    """Contributions d'Euler à l'ES (positivement homogène) :

        rho_i = w_i * E[X_i | X_Pi > VaR_alpha(X_Pi)] ,  somme = ES_alpha(X_Pi).

    Renvoie (dict {ticker: contribution}, total = ES du portefeuille)."""
    L = np.asarray(losses_per_asset)
    w = np.asarray(weights)
    port = L @ w
    var = np.quantile(port, alpha)
    mask = port > var
    if mask.sum() == 0:
        return None, np.nan
    cols = list(losses_per_asset.columns)
    contribs = {tk: w[i] * L[mask, i].mean() for i, tk in enumerate(cols)}
    return contribs, sum(contribs.values())


# ============================================================
# 7. AUTO-TEST RAPIDE
# ============================================================
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_returns, portfolio_losses, WINDOW

    losses = portfolio_losses(load_returns())
    win = losses.iloc[-WINDOW:]
    print("Auto-test du toolkit (dernière fenêtre de 250 jours) :")
    for alpha in (0.95, 0.975):
        vh, eh = var_es_historique(win, alpha)
        vp, ep = var_es_parametrique(win, alpha)
        vm, em = var_es_montecarlo(win, alpha)
        print(f"\nalpha = {alpha}")
        print(f"  Historique   : VaR={vh:.4f}  ES={eh:.4f}")
        print(f"  Paramétrique : VaR={vp:.4f}  ES={ep:.4f}")
        print(f"  Monte Carlo  : VaR={vm:.4f}  ES={em:.4f}")
