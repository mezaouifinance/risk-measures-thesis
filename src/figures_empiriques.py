# -*- coding: utf-8 -*-
"""Figures des extensions empiriques — à partir des VRAIS résultats
(données Yahoo Finance de l'utilisateur). Sans EWMA."""
import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

NAVY, ICEBLUE, GOLD, ACCENT, GRAY, GREEN = \
    "#1E2761", "#5B7AA9", "#C9A227", "#B22234", "#5A6378", "#2C7A2C"
mpl.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,
    "axes.spines.right":False,"axes.titleweight":"bold","axes.titlecolor":NAVY,
    "axes.labelcolor":NAVY,"xtick.color":GRAY,"ytick.color":GRAY,
    "savefig.dpi":160,"savefig.bbox":"tight","savefig.facecolor":"white"})
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures")); os.makedirs(OUT, exist_ok=True)
TICKERS = ['S&P 500', 'CAC 40', 'Apple', 'TotalEnergies']

# =========================================================
# A.7 — Sensibilité à la fenêtre (vrais chiffres, Hist + Param)
# =========================================================
fen = [60, 125, 250, 500]
hist = [6.81, 5.97, 5.53, 4.90]
para = [6.17, 5.81, 5.55, 4.87]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(fen, hist, marker='o', ms=8, lw=1.9, color=NAVY, label='Historique')
ax.plot(fen, para, marker='s', ms=8, lw=1.9, color=ICEBLUE, label='Paramétrique')
ax.axhline(5, color=ACCENT, ls='--', lw=1.5, label='Attendu sous H₀ : 5 %')
ax.set_xscale('log'); ax.set_xticks(fen); ax.set_xticklabels(fen)
ax.set_xlabel("Taille de fenêtre (jours)")
ax.set_ylabel("Taux d'exception observé (%)")
ax.set_title("Sensibilité du backtesting à la taille de fenêtre — VaR 95 %\n"
             "(portefeuille réel 2008-2025)", fontsize=11, pad=10)
ax.annotate("fenêtre courte\n→ sur-réaction", xy=(60, 6.8), xytext=(72, 6.6),
            fontsize=9, color=GRAY, style='italic')
ax.annotate("fenêtre longue\n→ sous-couverture", xy=(500, 4.9), xytext=(230, 4.5),
            fontsize=9, color=GRAY, style='italic')
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_sensibilite.png", dpi=160); plt.close()
print("✓ fig_sensibilite (réel)")

# =========================================================
# A.8 — Copules (vrais chiffres)
# =========================================================
al = [0.95, 0.975, 0.99, 0.995]
es_g = [2.79, 3.54, 4.72, 5.79]
es_t = [2.85, 3.71, 5.11, 6.46]
ratio = [1.02, 1.05, 1.08, 1.12]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(al)); w = 0.38
ax1.bar(x - w/2, es_g, w, color=ICEBLUE, alpha=0.88, label='Copule gaussienne')
ax1.bar(x + w/2, es_t, w, color=ACCENT, alpha=0.88, label='Copule t-Student (df=4)')
for i, (g, t) in enumerate(zip(es_g, es_t)):
    ax1.text(i - w/2, g+0.06, f"{g:.2f}", ha='center', fontsize=8, color=NAVY)
    ax1.text(i + w/2, t+0.06, f"{t:.2f}", ha='center', fontsize=8, color=NAVY, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels([f"{a:.3f}" for a in al])
ax1.set_xlabel("Niveau α"); ax1.set_ylabel("ES du portefeuille (%)")
ax1.set_title("ES selon la copule", fontsize=11)
ax1.legend(loc='upper left', fontsize=9); ax1.grid(axis='y', alpha=0.25)
ax1.set_ylim(0, 7.4)

ax2.plot(al, ratio, marker='o', ms=9, lw=2.2, color=ACCENT)
for a, r in zip(al, ratio):
    ax2.text(a, r+0.004, f"+{(r-1)*100:.0f}%", ha='center', fontsize=10, color=ACCENT, fontweight='bold')
ax2.axhline(1.0, color=GRAY, ls='--', lw=1)
ax2.set_xlabel("Niveau α"); ax2.set_ylabel("ES t-Student / ES gaussienne")
ax2.set_title("Sous-estimation des co-extrêmes\npar la copule gaussienne", fontsize=11)
ax2.set_ylim(0.99, 1.16); ax2.grid(alpha=0.25)
plt.suptitle("Copules : la dépendance de queue creuse l'écart dans les niveaux extrêmes",
             fontsize=12, color=NAVY, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_copules.png", dpi=160); plt.close()
print("✓ fig_copules (réel)")

# =========================================================
# A.9 — Optimisation CVaR (vrais chiffres : équipondéré vs min-CVaR)
# =========================================================
w_eq   = [0.25, 0.25, 0.25, 0.25]
w_cvar = [0.527, 0.101, 0.085, 0.287]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
xx = np.arange(4); ww = 0.38
ax1.bar(xx - ww/2, w_eq, ww, color=NAVY, alpha=0.85, label='Équipondéré')
ax1.bar(xx + ww/2, w_cvar, ww, color=GOLD, alpha=0.9, label='Min-CVaR (α=95 %)')
for i, v in enumerate(w_cvar):
    ax1.text(i + ww/2, v+0.01, f"{v:.2f}", ha='center', fontsize=9, color=NAVY, fontweight='bold')
ax1.set_xticks(xx); ax1.set_xticklabels(TICKERS, fontsize=9, rotation=12, ha='right')
ax1.set_ylabel("Poids"); ax1.set_title("Compositions (in-sample 2008-2018)", fontsize=11)
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.25)

# Risque out-of-sample
metrics = ['Vol', 'VaR 95 %', 'ES 95 %', 'Pire perte']
eq_vals = [1.21, 1.62, 2.96, 12.38]
cv_vals = [1.19, 1.62, 2.90, 12.07]
xm = np.arange(len(metrics)); wm = 0.38
b1 = ax2.bar(xm - wm/2, eq_vals, wm, color=NAVY, alpha=0.85, label='Équipondéré')
b2 = ax2.bar(xm + wm/2, cv_vals, wm, color=GOLD, alpha=0.9, label='Min-CVaR')
for b, v in zip(b1, eq_vals): ax2.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.2f}", ha='center', fontsize=8, color=NAVY)
for b, v in zip(b2, cv_vals): ax2.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.2f}", ha='center', fontsize=8, color=NAVY, fontweight='bold')
ax2.set_xticks(xm); ax2.set_xticklabels(metrics, fontsize=9)
ax2.set_ylabel("% (out-of-sample 2019-2025)")
ax2.set_title("Risque out-of-sample : min-CVaR réduit la queue", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.25); ax2.set_ylim(0, 14)
plt.suptitle("Optimisation CVaR (Rockafellar-Uryasev) : surpondère les indices, allège AAPL/TTE",
             fontsize=12, color=NAVY, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_cvar.png", dpi=160); plt.close()
print("✓ fig_cvar (réel)")

# =========================================================
# A.10 — Allocation Euler (vrais chiffres)
# =========================================================
periodes = ['Calme 2017', 'Crise 2008-09', 'COVID 2020', 'Récent 2023-25']
es_tot   = [1.02, 6.06, 10.00, 1.93]
contrib = {  # % par actif
    'S&P 500':       [17.1, 27.8, 22.5, 19.2],
    'CAC 40':        [21.4, 21.8, 21.2, 21.8],
    'Apple':         [36.3, 26.2, 21.8, 30.1],
    'TotalEnergies': [25.3, 24.3, 34.5, 28.9],
}
colors_a = {'S&P 500':NAVY, 'CAC 40':ICEBLUE, 'Apple':GOLD, 'TotalEnergies':ACCENT}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios':[2.1,1]})
x = np.arange(len(periodes)); bottoms = np.zeros(len(periodes))
for tk in TICKERS:
    vals = np.array(contrib[tk])
    ax1.bar(x, vals, bottom=bottoms, color=colors_a[tk], alpha=0.88, label=tk)
    for j, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 6:
            ax1.text(x[j], b+v/2, f"{v:.0f}%", ha='center', va='center',
                     color='white', fontsize=9, fontweight='bold')
    bottoms += vals
ax1.set_xticks(x); ax1.set_xticklabels(periodes, fontsize=9)
ax1.set_ylabel("Contribution à l'ES 95 % (%)")
ax1.set_title("Contributions Euler — la concentration se déplace en crise", fontsize=11)
ax1.legend(loc='upper center', fontsize=8, ncol=4, bbox_to_anchor=(0.5, -0.08))
ax1.set_ylim(0, 100)

# ES totale par période
b = ax2.bar(x, es_tot, color=[ICEBLUE, GOLD, ACCENT, ICEBLUE], alpha=0.88)
for bb, v in zip(b, es_tot):
    ax2.text(bb.get_x()+bb.get_width()/2, v+0.2, f"{v:.1f}%", ha='center', fontsize=10, color=NAVY, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(periodes, fontsize=8, rotation=15, ha='right')
ax2.set_ylabel("ES 95 % totale (%)")
ax2.set_title("Niveau de risque par période", fontsize=11)
ax2.set_ylim(0, 11.5); ax2.grid(axis='y', alpha=0.25)
plt.suptitle("Allocation Euler : TotalEnergies domine en 2020 (krach pétrolier), l'ES totale x10",
             fontsize=12, color=NAVY, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_euler.png", dpi=160); plt.close()
print("✓ fig_euler (réel)")

print(f"\n4 figures (vrais chiffres) générées dans {OUT}/")
