# Mémoire — Mesures de risque

Ce dépôt contient le code Python utilisé pour reproduire les résultats empiriques du mémoire :

**Théorie des mesures de risque**  
Master 1 Mathématiques Appliquées à l’Économie et à la Finance  
Université Paris 1 Panthéon-Sorbonne

Le script principal permet de reproduire les estimations de Value-at-Risk, d’Expected Shortfall, les tests de backtesting, les stress tests historiques, ainsi que les extensions numériques présentées dans le mémoire.

## Contenu du dépôt

```text
memoire-mesures-de-risque/
├── README.md
├── requirements.txt
├── annexe_code_complet.py
└── data/
    └── prices.csv
```

Le fichier principal est :

```text
annexe_code_complet.py
```

Il contient l’ensemble du code utilisé pour produire les résultats numériques du mémoire.

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/mezaouifinance/memoire-mesures-de-risque.git
cd memoire-mesures-de-risque
```

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

## Exécution

Pour lancer le script complet :

```bash
python annexe_code_complet.py
```

Le script affiche directement les résultats dans le terminal.

## Données utilisées

Les actifs étudiés sont :

```text
^GSPC   : indice S&P 500
^FCHI   : indice CAC 40
AAPL    : Apple Inc.
TTE.PA  : TotalEnergies
```

La période d’étude va de 2008 à 2025.

Les données sont soit chargées depuis le fichier :

```text
data/prices.csv
```

soit téléchargées automatiquement via `yfinance` si le fichier n’est pas présent.

## Résultats reproduits

Le script permet de reproduire :

- les estimations de VaR et d’Expected Shortfall ;
- le backtesting de la VaR à 95 % ;
- le backtesting de l’Expected Shortfall à 97,5 % ;
- les stress tests historiques sur 2008 et 2020 ;
- l’analyse de sensibilité à la taille de fenêtre ;
- la comparaison entre copule gaussienne et copule t-Student ;
- l’optimisation de portefeuille par minimisation de la CVaR ;
- l’allocation d’Euler des contributions au risque.

## Bibliothèques utilisées

Les principales bibliothèques Python utilisées sont :

```text
numpy
pandas
scipy
matplotlib
yfinance
```

Elles sont listées dans le fichier `requirements.txt`.

## Remarque

Le code est fourni afin de rendre reproductibles les résultats empiriques du mémoire.

Les valeurs numériques peuvent légèrement varier si les données sont retéléchargées depuis `yfinance`, en raison d’éventuels ajustements de prix.# m-moire-mesures-de-risque
