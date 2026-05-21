# Théorie des mesures de risque — code de reproduction

Code Python accompagnant le mémoire **« Théorie des mesures de risque »**
(Mehdi Zaoui & Zakarya Sellami, Université Paris 1 Panthéon-Sorbonne, 2026).

Il reproduit l'intégralité des résultats empiriques du mémoire — estimation
et backtesting de la VaR et de l'Expected Shortfall, stress tests, copules,
optimisation de portefeuille par minimisation de la CVaR et allocation
d'Euler — sur un portefeuille équipondéré de quatre actifs
(S&P 500, CAC 40, Apple, TotalEnergies) sur la période 2008–2025.

## Installation

```bash
git clone https://github.com/<votre-compte>/risk-measures-thesis.git
cd risk-measures-thesis
pip install -r requirements.txt
```

Python ≥ 3.10 recommandé.

## Données et reproductibilité

Tous les scripts passent par une couche de données unique
(`src/data.py`) qui suit la stratégie **« CSV figé + yfinance »** :

1. si `data/prices.csv` existe, il est chargé tel quel — le run est alors
   **reproductible au chiffre près** ;
2. sinon, les cours sont téléchargés via `yfinance` puis **mis en cache**
   dans `data/prices.csv` (le run suivant sera donc figé).

```bash
# Figer une fois pour toutes le jeu de données réel (puis le committer)
python src/data.py

# Variante 100 % hors-ligne : jeu SYNTHÉTIQUE calibré sur le mémoire
python src/data.py --synthetic
```

> Pour une reproductibilité parfaite entre machines, lancez `python src/data.py`
> une fois, puis committez le fichier `data/prices.csv` ainsi produit
> (décommentez la ligne correspondante dans `.gitignore`).

## Utilisation

```bash
# Annexe A du mémoire : pipeline complet, autonome, sections A.1 à A.10
python annexe_code_complet.py

# Analyses modulaires (importent la bibliothèque src/risk_toolkit.py)
python src/copules.py            # copule gaussienne vs t-Student
python src/optimisation_cvar.py  # Rockafellar-Uryasev (programme linéaire)
python src/allocation_euler.py   # contributions d'Euler à l'ES

# Bibliothèque réutilisable (auto-test)
python src/risk_toolkit.py

# Figures
python src/figures_conceptuelles.py   # figures pédagogiques -> figures/
python src/figures_empiriques.py      # figures des résultats -> figures/
```

## Structure du dépôt

```
risk-measures-thesis/
├── annexe_code_complet.py     # Annexe A du mémoire (autonome, A.1–A.10)
├── src/
│   ├── data.py                # couche de données (yfinance + cache CSV + synthétique)
│   ├── risk_toolkit.py        # bibliothèque : estimateurs, tests, copules, CVaR, Euler
│   ├── copules.py             # analyse : copules gaussienne vs t-Student
│   ├── optimisation_cvar.py   # analyse : minimisation de la CVaR
│   ├── allocation_euler.py    # analyse : allocation d'Euler
│   ├── figures_conceptuelles.py
│   └── figures_empiriques.py
├── data/                      # data/prices.csv (généré, à committer si souhaité)
├── requirements.txt
├── LICENSE                    # MIT
└── README.md
```

## Méthodes implémentées

| Thème | Méthodes |
|-------|----------|
| Estimateurs VaR / ES | historique (quantile empirique), paramétrique gaussien, Monte Carlo Student (max. de vraisemblance) |
| Backtesting VaR | Kupiec (couverture inconditionnelle), Christoffersen (couverture + indépendance) |
| Backtesting ES | statistique Z₂ d'Acerbi–Szekely, p-valeur par bootstrap paramétrique |
| Dépendance | copules gaussienne et t-Student, marginales de Student |
| Optimisation | minimisation de la CVaR par programme linéaire (Rockafellar–Uryasev), Markowitz min-variance |
| Décomposition du risque | contributions d'Euler à l'ES |

## Conventions

- Pertes définies en convention « pertes positives » : `X = -log-rendement`.
- VaR backtestée à 95 % (corps du mémoire) ; ES backtestée à 97,5 %
  (convention FRTB/Bâle, où l'ES est relevée pour égaler la sévérité d'une
  VaR à 99 %).
- Fenêtre glissante de 250 jours ouvrés (un an), conforme à Bâle.

## Citation

```
Zaoui, M., & Sellami, Z. (2026). Théorie des mesures de risque.
Mémoire, Université Paris 1 Panthéon-Sorbonne.
```

## Licence

Distribué sous licence MIT — voir [`LICENSE`](LICENSE).
