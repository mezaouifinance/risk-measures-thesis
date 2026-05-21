# -*- coding: utf-8 -*-
"""
=====================================================================
PACK DE FIGURES CONCEPTUELLES — Théorie des Mesures de Risque
=====================================================================
Génère les figures théoriques indispensables du mémoire (ch. 1 à 4 + EVT).
Sortie : dossier figures_concept/ (PNG haute résolution).

Charte : navy #1E2761, gold #C9A227, accent #B22234, iceblue #5B7AA9.

Dépendances : numpy, scipy, matplotlib (+ returns.csv pour la fig. 1).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy import stats

NAVY, ICEBLUE, GOLD, ACCENT, GRAY, CREAM, GREEN = \
    "#1E2761", "#5B7AA9", "#C9A227", "#B22234", "#5A6378", "#FAF7F2", "#2C7A2C"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRAY, "axes.labelcolor": NAVY, "axes.titlecolor": NAVY,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures"))
os.makedirs(OUT, exist_ok=True)
np.random.seed(7)


# =====================================================================
# FIG 1 — Queues lourdes : QQ-plot + densité log (ch. 1)
# =====================================================================
def fig_queues_lourdes():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_returns, portfolio_losses
    losses = portfolio_losses(load_returns()).values
    losses = losses[np.isfinite(losses)]
    z = (losses - losses.mean()) / losses.std()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # QQ-plot
    ax = axes[0]
    osm, osr = stats.probplot(z, dist="norm", fit=False)
    ax.scatter(osm, osr, s=6, color=NAVY, alpha=0.5)
    lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lim, lim, color=ACCENT, lw=2, label="Référence gaussienne")
    ax.set_xlabel("Quantiles théoriques $\\mathcal{N}(0,1)$")
    ax.set_ylabel("Quantiles empiriques (pertes)")
    ax.set_title("QQ-plot : les points décrochent dans les queues", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    ax.annotate("queue droite\nplus lourde", xy=(osm.max()*0.85, osr.max()*0.7),
                xytext=(1.0, osr.max()*0.55), fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT))

    # Densité en échelle log
    ax = axes[1]
    xs = np.linspace(-6, 6, 400)
    ax.hist(z, bins=200, density=True, color=ICEBLUE, alpha=0.55, label="Pertes (centrées-réduites)")
    ax.plot(xs, stats.norm.pdf(xs), color=ACCENT, lw=2, label="$\\mathcal{N}(0,1)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(-6, 6)
    ax.set_xlabel("Perte normalisée")
    ax.set_ylabel("Densité (échelle log)")
    ax.set_title("En échelle log, les queues empiriques dominent la gaussienne", fontsize=11)
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(alpha=0.25)

    plt.suptitle("Figure 1.1 — Évidence empirique des queues lourdes (kurtosis ≫ 3)",
                 fontsize=12, color=NAVY, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_01_queues_lourdes.png", dpi=170)
    plt.close()
    print("✓ fig_01_queues_lourdes")


# =====================================================================
# FIG 2 — Les 4 axiomes ADEH (ch. 2) — diagramme conceptuel
# =====================================================================
def fig_axiomes():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(5, 5.6, "Mesure cohérente : 4 axiomes (Artzner-Delbaen-Eber-Heath, 1999)",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)

    axiomes = [
        ("Monotonie", "X ≤ Y  ⇒  ρ(X) ≤ ρ(Y)", "Plus de perte, plus de capital", NAVY),
        ("Invariance par translation", "ρ(X + c) = ρ(X) + c", "Le cash couvre parfaitement", ICEBLUE),
        ("Homogénéité positive", "ρ(λX) = λ ρ(X),  λ > 0", "Invariance d'échelle", GOLD),
        ("Sous-additivité", "ρ(X + Y) ≤ ρ(X) + ρ(Y)", "Diversification favorable", ACCENT),
    ]
    pos = [(0.4, 2.9), (5.1, 2.9), (0.4, 0.5), (5.1, 0.5)]
    for (name, formula, gloss, col), (x, y) in zip(axiomes, pos):
        ax.add_patch(FancyBboxPatch((x, y), 4.5, 2.1, boxstyle="round,pad=0.08",
                     facecolor=CREAM, edgecolor=col, linewidth=2))
        ax.add_patch(FancyBboxPatch((x, y + 1.7), 4.5, 0.4, boxstyle="round,pad=0.08",
                     facecolor=col, edgecolor=col, linewidth=2))
        ax.text(x + 2.25, y + 1.88, name, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x + 2.25, y + 1.05, formula, ha="center", va="center",
                fontsize=13, color=NAVY, style="italic")
        ax.text(x + 2.25, y + 0.4, gloss, ha="center", va="center",
                fontsize=10, color=GRAY, style="italic")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_02_axiomes_adeh.png", dpi=170)
    plt.close()
    print("✓ fig_02_axiomes_adeh")


# =====================================================================
# FIG 3 — Ensembles d'acceptabilité (ch. 2) — géométrie
# =====================================================================
def fig_acceptabilite():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Gauche : ensemble convexe (cohérent)
    ax = axes[0]
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
    # demi-plan convexe A = {X : ρ(X) ≤ 0}
    poly = Polygon([(-3, 3), (3, -3), (3, 3), (-3, 3)], closed=True,
                   facecolor=ICEBLUE, alpha=0.25, edgecolor=NAVY, lw=2)
    ax.add_patch(poly)
    ax.plot([-3, 3], [3, -3], color=NAVY, lw=2)
    ax.text(-1.8, -1.8, r"$\mathcal{A}_\rho = \{X : \rho(X) \leq 0\}$",
            fontsize=13, color=NAVY, fontweight="bold")
    ax.text(1.4, 1.6, "convexe\n(cône)", fontsize=11, color=NAVY, ha="center")
    # deux positions acceptables → leur mélange aussi
    p1, p2 = np.array([-2, 2.5]), np.array([2.5, -2])
    for p in (p1, p2):
        ax.scatter(*p, s=60, color=ACCENT, zorder=5)
    mid = (p1 + p2) / 2
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=ACCENT, lw=1.5)
    ax.scatter(*mid, s=60, color=GREEN, zorder=5)
    ax.text(mid[0]+0.1, mid[1]+0.3, "mélange\nacceptable ✓", fontsize=9, color=GREEN)
    ax.set_title("Mesure cohérente : ensemble d'acceptabilité convexe", fontsize=11)
    ax.set_xlabel("scénario $\\omega_1$"); ax.set_ylabel("scénario $\\omega_2$")

    # Droite : ensemble non convexe (VaR)
    ax = axes[1]
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.axhline(0, color=GRAY, lw=0.6); ax.axvline(0, color=GRAY, lw=0.6)
    # union de deux demi-plans (non convexe) — illustration VaR
    poly1 = Polygon([(-3, 1), (3, 1), (3, 3), (-3, 3)], closed=True,
                    facecolor=GOLD, alpha=0.22, edgecolor=GOLD, lw=1.5)
    poly2 = Polygon([(1, -3), (3, -3), (3, 3), (1, 3)], closed=True,
                    facecolor=GOLD, alpha=0.22, edgecolor=GOLD, lw=1.5)
    ax.add_patch(poly1); ax.add_patch(poly2)
    ax.text(-2.6, -2.5, r"$\mathcal{A}_{VaR}$ : union, non convexe",
            fontsize=12, color="#8B6914", fontweight="bold")
    # deux points acceptables, mélange hors de l'ensemble
    p1, p2 = np.array([-2, 2.2]), np.array([2.2, -2])
    for p in (p1, p2):
        ax.scatter(*p, s=60, color=NAVY, zorder=5)
    mid = (p1 + p2) / 2
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=ACCENT, lw=1.5)
    ax.scatter(*mid, s=70, color=ACCENT, zorder=5, marker="X")
    ax.text(mid[0]+0.15, mid[1]+0.25, "mélange\nREFUSÉ ✗", fontsize=9, color=ACCENT, fontweight="bold")
    ax.set_title("VaR : ensemble non convexe → diversification pénalisée", fontsize=11)
    ax.set_xlabel("scénario $\\omega_1$"); ax.set_ylabel("scénario $\\omega_2$")

    plt.suptitle("Figure 2.1 — Lecture géométrique : la cohérence ⇔ convexité de l'ensemble d'acceptabilité",
                 fontsize=12, color=NAVY, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_03_ensembles_acceptabilite.png", dpi=170)
    plt.close()
    print("✓ fig_03_ensembles_acceptabilite")


# =====================================================================
# FIG 4 — Poids spectraux φ(u) (ch. 2)
# =====================================================================
def fig_spectraux():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    u = np.linspace(0, 1, 500)
    alpha = 0.95

    # ES : poids uniforme 1/(1-α) sur [α,1]
    es_w = np.where(u >= alpha, 1/(1-alpha), 0)
    ax.plot(u, es_w, color=NAVY, lw=2.4, label="ES$_{95\\%}$ : poids uniforme sur la queue")
    ax.fill_between(u, es_w, color=NAVY, alpha=0.12)

    # Mesure spectrale générale : poids croissant (ex. exponentiel normalisé)
    _trapz = getattr(np, "trapezoid", np.trapz)   # numpy >= 2.0 : trapezoid
    gen = np.exp(4*u); gen = gen / _trapz(gen, u)
    ax.plot(u, gen, color=GOLD, lw=2.2, label="Mesure spectrale générale : poids croissant")

    # VaR : Dirac en α (flèche)
    ax.annotate("", xy=(alpha, 16), xytext=(alpha, 0),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2.5))
    ax.text(alpha-0.01, 16.5, "VaR$_{95\\%}$ : masse de Dirac en $\\alpha$",
            ha="right", fontsize=10, color=ACCENT, fontweight="bold")

    ax.set_xlabel("Niveau de quantile $u$")
    ax.set_ylabel("Poids spectral $\\varphi(u)$")
    ax.set_title("Figure 2.2 — Représentation spectrale : VaR, ES et mesures spectrales générales",
                 fontsize=11, pad=10)
    ax.set_ylim(0, 22); ax.set_xlim(0, 1)
    ax.axvline(alpha, color=GRAY, ls=":", lw=1)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.text(0.5, 19, "La VaR ne pèse qu'un point ; l'ES moyenne toute la queue → cohérence",
            ha="center", fontsize=9.5, color=GRAY, style="italic")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_04_poids_spectraux.png", dpi=170)
    plt.close()
    print("✓ fig_04_poids_spectraux")


# =====================================================================
# FIG 5 — VaR comme quantile (ch. 3)
# =====================================================================
def fig_var_quantile():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.linspace(-3.5, 4.5, 1000)
    y = stats.norm.pdf(x)
    var = stats.norm.ppf(0.95)

    ax1.plot(x, y, color=NAVY, lw=2)
    ax1.fill_between(x[x <= var], y[x <= var], color=ICEBLUE, alpha=0.35, label="95 % de la masse")
    ax1.fill_between(x[x >= var], y[x >= var], color=ACCENT, alpha=0.5, label="5 % de queue")
    ax1.axvline(var, color=ACCENT, lw=2, ls="--")
    ax1.text(var+0.1, 0.3, f"VaR$_{{95\\%}}$ = {var:.2f}", color=ACCENT, fontweight="bold", fontsize=11)
    ax1.set_title("Densité : la VaR sépare 95 % / 5 %", fontsize=11)
    ax1.set_xlabel("Perte X"); ax1.set_ylabel("Densité")
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.2)

    # CDF
    cdf = stats.norm.cdf(x)
    ax2.plot(x, cdf, color=NAVY, lw=2)
    ax2.axhline(0.95, color=GRAY, ls=":", lw=1.2)
    ax2.axvline(var, color=ACCENT, lw=2, ls="--")
    ax2.scatter([var], [0.95], s=70, color=ACCENT, zorder=5)
    ax2.text(var+0.15, 0.80, "VaR = $F^{-1}(0{,}95)$", color=ACCENT, fontweight="bold", fontsize=11)
    ax2.set_title("Fonction de répartition : VaR = quantile d'ordre $\\alpha$", fontsize=11)
    ax2.set_xlabel("Perte X"); ax2.set_ylabel("$F_X(x)$")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.2)

    plt.suptitle("Figure 3.1 — La Value-at-Risk est un quantile",
                 fontsize=12, color=NAVY, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_05_var_quantile.png", dpi=170)
    plt.close()
    print("✓ fig_05_var_quantile")


# =====================================================================
# FIG 6 — Même VaR, ES différente (ch. 3) — figure-clé
# =====================================================================
def fig_meme_var_es_differente():
    var_cible = stats.norm.ppf(0.95)  # 1.645

    # Distribution 1 : normale standard
    es1 = stats.norm.pdf(var_cible) / 0.05

    # Distribution 2 : Student t(3) rééchelonnée pour MÊME VaR_0.95
    nu = 3
    q_t = stats.t.ppf(0.95, nu)
    s = var_cible / q_t                       # facteur d'échelle
    # ES de s*t_nu à 0.95, par intégration numérique
    grid = np.linspace(0.95, 0.999999, 200000)
    es2 = s * np.mean(stats.t.ppf(grid, nu))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.linspace(-3.5, 7, 1500)
    f1 = stats.norm.pdf(x)
    f2 = stats.t.pdf(x / s, nu) / s
    ax.plot(x, f1, color=ICEBLUE, lw=2.2, label="Distribution mince (gaussienne)")
    ax.plot(x, f2, color=ACCENT, lw=2.2, label="Distribution à queue lourde (Student rééch.)")

    ax.axvline(var_cible, color=NAVY, lw=2, ls="--")
    ax.text(var_cible-0.1, 0.40, "VaR commune\n= 1,64", ha="right", color=NAVY, fontweight="bold", fontsize=10)

    ax.scatter([es1], [0.005], s=80, color=ICEBLUE, marker="v", zorder=5)
    ax.scatter([es2], [0.005], s=80, color=ACCENT, marker="v", zorder=5)
    ax.annotate(f"ES mince = {es1:.2f}", xy=(es1, 0.01), xytext=(es1-0.3, 0.13),
                color=ICEBLUE, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ICEBLUE))
    ax.annotate(f"ES lourde = {es2:.2f}", xy=(es2, 0.01), xytext=(es2-0.2, 0.20),
                color=ACCENT, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ACCENT))

    ax.set_xlim(-3.5, 7); ax.set_ylim(0, 0.45)
    ax.set_xlabel("Perte X"); ax.set_ylabel("Densité")
    ax.set_title("Figure 3.2 — Même VaR, ES très différentes :\nla VaR est aveugle à la forme de la queue",
                 fontsize=11.5, pad=10)
    ax.legend(loc="upper right", fontsize=9.5); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_06_meme_var_es_differente.png", dpi=170)
    plt.close()
    print(f"✓ fig_06_meme_var_es_differente  (ES mince={es1:.3f}, ES lourde={es2:.3f})")


# =====================================================================
# FIG 7 — VaR non sous-additive (ch. 3) : courbe diagnostique + exemple discret
# =====================================================================
def fig_var_non_sous_additive():
    rng = np.random.default_rng(0)
    N = 2_000_000
    xm, xi = 1.0, 1.5
    A = xm * rng.random(N)**(-1/xi)
    B = xm * rng.random(N)**(-1/xi)
    S = A + B

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panneau gauche : diagnostic D(α) = VaR(A+B) - [VaR(A)+VaR(B)]
    alphas = np.linspace(0.80, 0.995, 40)
    D = []
    for a in alphas:
        vA = np.quantile(A, a); vB = np.quantile(B, a); vS = np.quantile(S, a)
        D.append(vS - (vA + vB))
    D = np.array(D)
    ax1.plot(alphas, D, color=NAVY, lw=2.2)
    ax1.axhline(0, color=GRAY, lw=1)
    ax1.fill_between(alphas, D, 0, where=(D > 0), color=ACCENT, alpha=0.3,
                     label="Violation (super-additif)")
    ax1.fill_between(alphas, D, 0, where=(D <= 0), color=ICEBLUE, alpha=0.3,
                     label="Sous-additif")
    # zone de violation
    idx = np.where(D > 0)[0]
    if len(idx):
        ax1.axvline(alphas[idx[-1]], color=ACCENT, ls=":", lw=1.3)
        ax1.text(alphas[idx[-1]], D.min()*0.5,
                 f"violation pour\nα ≲ {alphas[idx[-1]]:.2f}",
                 color=ACCENT, fontsize=9, fontweight="bold", ha="center")
    ax1.set_xlabel("Niveau α")
    ax1.set_ylabel("VaR(A+B) − [VaR(A)+VaR(B)]")
    ax1.set_title("Pareto (ξ=1,5) : la sous-additivité dépend de α", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9); ax1.grid(alpha=0.25)

    # Panneau droit : exemple discret des défauts (imparable à 95%)
    # Deux obligations indépendantes : perte 100 avec prob 4%, sinon 0
    var_single = 0.0           # P(perte=0)=0.96 > 0.95 → VaR_0.95 = 0
    # portefeuille : P(au moins un défaut) = 1-0.96^2 = 0.0784 > 0.05 → VaR_0.95 = 100
    var_port = 100.0
    bars = ax2.bar([0, 1], [var_single + var_single, var_port],
                   color=[ICEBLUE, ACCENT], alpha=0.85, width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["VaR(A) + VaR(B)\n= 0 + 0", "VaR(A + B)\n= 100"], fontsize=10)
    ax2.set_ylabel("VaR$_{95\\%}$ (perte)")
    ax2.set_title("Exemple des défauts : violation nette à 95 %", fontsize=11)
    for b, v in zip(bars, [0, 100]):
        ax2.text(b.get_x()+b.get_width()/2, v+3, f"{v:.0f}",
                 ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax2.set_ylim(0, 115)
    ax2.text(0.5, 60, "Deux obligations : défaut\n(perte 100) avec p = 4 % < 5 %.\n"
                      "Chacune : VaR = 0.\nPortefeuille : P(≥1 défaut) = 7,8 % > 5 %\n→ VaR = 100",
             ha="center", fontsize=8.5, color=GRAY,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CREAM, edgecolor=GRAY))
    ax2.grid(axis="y", alpha=0.25)

    plt.suptitle("Figure 3.3 — La VaR n'est pas sous-additive : diagnostic et contre-exemple",
                 fontsize=12, color=NAVY, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_07_var_non_sous_additive.png", dpi=170)
    plt.close()
    print("✓ fig_07_var_non_sous_additive")


# =====================================================================
# FIG 8 — VaR vs ES (ch. 4)
# =====================================================================
def fig_var_vs_es():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.linspace(-3.5, 5, 1000)
    y = stats.norm.pdf(x)
    var = stats.norm.ppf(0.95)
    es = stats.norm.pdf(var) / 0.05

    ax.plot(x, y, color=NAVY, lw=2)
    ax.fill_between(x[x >= var], y[x >= var], color=ACCENT, alpha=0.45)
    ax.axvline(var, color=ACCENT, lw=2, ls="--")
    ax.axvline(es, color=GOLD, lw=2, ls="--")
    ax.annotate(f"VaR$_{{95\\%}}$ = {var:.2f}\n« jusqu'où ? »",
                xy=(var, 0.06), xytext=(var-1.8, 0.18), fontsize=10, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.annotate(f"ES$_{{95\\%}}$ = {es:.2f}\n« combien en moyenne ? »",
                xy=(es, 0.025), xytext=(es+0.2, 0.14), fontsize=10, color="#8B6914", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOLD))
    ax.set_xlabel("Perte X"); ax.set_ylabel("Densité")
    ax.set_title("Figure 4.1 — VaR (seuil) contre ES (moyenne de la queue)",
                 fontsize=11.5, pad=10)
    ax.set_ylim(0, 0.45); ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_08_var_vs_es.png", dpi=170)
    plt.close()
    print("✓ fig_08_var_vs_es")


# =====================================================================
# FIG 9 — Représentation variationnelle de Rockafellar-Uryasev (ch. 4)
# =====================================================================
def fig_rockafellar_uryasev():
    alpha = 0.95
    # X ~ N(0,1). g(v) = v + E[(X-v)+]/(1-α)
    rng = np.random.default_rng(1)
    sample = rng.standard_normal(2_000_000)
    vs = np.linspace(-1, 4, 300)
    g = np.array([v + np.mean(np.maximum(sample - v, 0)) / (1 - alpha) for v in vs])
    v_star = stats.norm.ppf(alpha)
    es = stats.norm.pdf(v_star) / (1 - alpha)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(vs, g, color=NAVY, lw=2.4, label="$g(v) = v + \\frac{1}{1-\\alpha}\\,E[(X-v)^+]$")
    ax.scatter([v_star], [es], s=90, color=ACCENT, zorder=5)
    ax.axvline(v_star, color=ACCENT, ls="--", lw=1.5)
    ax.axhline(es, color=GOLD, ls="--", lw=1.5)
    ax.annotate(f"minimum en $v^* = $ VaR = {v_star:.2f}",
                xy=(v_star, es), xytext=(v_star+0.3, es+0.7), fontsize=10, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.text(0.05, es+0.05, f"valeur minimale = ES = {es:.2f}", color="#8B6914",
            fontsize=10, fontweight="bold")
    ax.set_xlabel("$v$"); ax.set_ylabel("$g(v)$")
    ax.set_title("Figure 4.2 — Représentation variationnelle de l'ES (Rockafellar-Uryasev) :\n"
                 "fonction convexe, minimisée en $v^* = $ VaR, de valeur l'ES",
                 fontsize=11, pad=10)
    ax.legend(loc="upper center", fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_09_rockafellar_uryasev.png", dpi=170)
    plt.close()
    print(f"✓ fig_09_rockafellar_uryasev  (v*={v_star:.3f}, ES={es:.3f})")


# =====================================================================
# FIG 10 — ES sous-additive (ch. 4) : Pareto, valeurs simulées
# =====================================================================
def fig_es_sous_additive():
    rng = np.random.default_rng(0)
    N = 2_000_000
    xm, xi = 1.0, 1.5
    A = xm * rng.random(N)**(-1/xi)
    B = xm * rng.random(N)**(-1/xi)
    S = A + B
    a = 0.95
    def es(arr):
        v = np.quantile(arr, a); return arr[arr > v].mean()
    esA, esB, esS = es(A), es(B), es(S)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(["ES(A)", "ES(B)", "ES(A)+ES(B)", "ES(A+B)"],
                  [esA, esB, esA+esB, esS],
                  color=[ICEBLUE, ICEBLUE, GOLD, GREEN], alpha=0.85)
    for b, v in zip(bars, [esA, esB, esA+esB, esS]):
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}",
                ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.set_ylabel("ES$_{95\\%}$")
    ax.set_title("Figure 4.3 — L'ES est sous-additive : ES(A+B) ≤ ES(A)+ES(B)\n"
                 f"(deux Pareto ξ=1,5 ; {esS:.1f} ≤ {esA+esB:.1f} ✓)",
                 fontsize=11, pad=10)
    ax.set_ylim(0, (esA+esB)*1.18)
    # flèche de comparaison
    ax.annotate("", xy=(3, esS), xytext=(2, esA+esB),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.text(2.5, (esS+esA+esB)/2, "diversification\nbénéfique", ha="center",
            fontsize=9, color=ACCENT, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_10_es_sous_additive.png", dpi=170)
    plt.close()
    print(f"✓ fig_10_es_sous_additive  (ES(A)={esA:.2f}, somme={esA+esB:.2f}, ES(A+B)={esS:.2f})")


# =====================================================================
# FIG 11 — EVT : mean excess plot + effet de l'indice de queue (ch. 6)
# =====================================================================
def fig_evt():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panneau gauche : mean excess function e(u) = E[X-u | X>u]
    rng = np.random.default_rng(3)
    # données à queue lourde (Pareto ξ=2 → e(u) croît linéairement)
    xi = 2.0
    data = rng.random(200000)**(-1/xi)
    us = np.linspace(np.quantile(data, 0.5), np.quantile(data, 0.995), 60)
    e_u = [data[data > u].mean() - u if (data > u).sum() > 50 else np.nan for u in us]
    ax1.plot(us, e_u, color=NAVY, lw=2, marker="o", markersize=3)
    # ligne de tendance (queue lourde → pente positive)
    valid = ~np.isnan(e_u)
    coef = np.polyfit(us[valid], np.array(e_u)[valid], 1)
    ax1.plot(us, coef[0]*us + coef[1], color=ACCENT, lw=1.8, ls="--",
             label="pente > 0 → queue lourde")
    ax1.set_xlabel("Seuil $u$")
    ax1.set_ylabel("Excès moyen $e(u) = E[X-u\\,|\\,X>u]$")
    ax1.set_title("Mean excess plot : pente positive = queue de type Pareto", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9); ax1.grid(alpha=0.25)

    # Panneau droit : survie GPD pour ξ < 0, = 0, > 0
    x = np.linspace(0, 6, 400)
    sigma = 1.0
    for xi_val, col, lab in [(-0.3, ICEBLUE, "ξ < 0 : queue bornée"),
                             (0.0, NAVY, "ξ = 0 : exponentielle"),
                             (0.5, GOLD, "ξ = 0,5 : lourde"),
                             (1.0, ACCENT, "ξ = 1 : ES diverge")]:
        if xi_val == 0:
            surv = np.exp(-x / sigma)
        else:
            base = 1 + xi_val * x / sigma
            with np.errstate(invalid="ignore"):
                surv = np.where(base > 0, base**(-1/xi_val), 0)
        ax2.plot(x, surv, color=col, lw=2, label=lab)
    ax2.set_xlabel("Excès $x$ au-delà du seuil")
    ax2.set_ylabel("Fonction de survie $\\bar G_{\\xi,\\sigma}(x)$")
    ax2.set_title("Loi de Pareto généralisée : l'indice ξ gouverne la queue", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8.5); ax2.grid(alpha=0.25)
    ax2.set_ylim(0, 1.02)

    plt.suptitle("Figure 6.1 — Théorie des valeurs extrêmes : diagnostic et rôle de l'indice de queue ξ",
                 fontsize=12, color=NAVY, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_11_evt.png", dpi=170)
    plt.close()
    print("✓ fig_11_evt")


# =====================================================================
if __name__ == "__main__":
    fig_queues_lourdes()
    fig_axiomes()
    fig_acceptabilite()
    fig_spectraux()
    fig_var_quantile()
    fig_meme_var_es_differente()
    fig_var_non_sous_additive()
    fig_var_vs_es()
    fig_rockafellar_uryasev()
    fig_es_sous_additive()
    fig_evt()
    print(f"\n11 figures générées dans {OUT}/")
