# 🚀 Hyperliquid Analytics Agent

Agent d'analyse technique en construction autour des données Hyperliquid, avec une trajectoire orientée vers l’agrégation multichaîne on-chain et des calculs off-chain avancés.

## État actuel

- **Client Hyperliquid (async)** : récupération des chandeliers `candleSnapshot`, des fills utilisateurs et du snapshot `metaAndAssetCtxs` (funding, open interest, mark price, etc.) via des modèles Pydantic stricts.
- **Modélisation** : `PerpMeta`, `PerpAssetContext`, et `MetaAndAssetCtxsResponse` garantissent la validation des payloads Hyperliquid.
- **Tests unitaires** : couvrent le client (conversion OHLCV, gestion d’erreurs, mappage `metaAndAssetCtxs`, cas avec champs `null`) pour sécuriser les futures évolutions.
- **Repository DuckDB (en cours)** : base embarquée destinée à persister l’univers perpétuel et les contextes de marché (schema défini, implémentation en cours d’intégration).
- **Structure naissante** : séparation claire Client / Repository, couche Service et CLI orchestratrice à formaliser dans les prochaines étapes.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
make install-dev
```

## Configuration

```bash
cp .env.example .env
# Éditer .env avec vos clés API Hyperliquid
```

## Lancer les tests

```bash
./venv/bin/python -m pytest
```

## Roadmap vers un système d’analytics complet

- **🐣 Phase 1 — Hyperliquid seulement (en cours)**
  - [x] Client async & modèles Pydantic
  - [x] Tests unitaires sur les endpoints principaux
  - [ ] Repository DuckDB fonctionnel (`save_snapshot`, `fetch_latest`, vues analytiques)
  - [ ] Service d’ingestion périodique + CLI (`collect`, `show-latest`, etc.)
  - [ ] Indicateurs techniques de base (SMA/EMA, RSI, MACD, Bollinger, VWAP) calculés via DuckDB

- **🌐 Phase 2 — Analytics temps réel & API interne**
  - [ ] Rafraîchissement programmatique (scheduler, WebSocket trades/l2 book)
  - [ ] API FastAPI exposant les indicateurs et snapshots
  - [ ] Tableau de bord exploratoire (Streamlit ou frontend maison)
  - [ ] Gestion des alertes (funding extrême, variations OI, divergence volume/prix)

- **🔗 Phase 3 — Extension multichaîne & on-chain**
  - [ ] Ingestion de données on-chain (déx, bridges, métriques DeFi) via indexeurs publics
  - [ ] Calculs off-chain corrélant données Hyperliquid & on-chain (flux entrants, activity whales, etc.)
  - [ ] Normalisation multi-sources et enrichissement du repository (tables additionnelles, heuristiques)
  - [ ] Optimisation de la persistence (archivage Parquet, compression, rétention intelligente)

- **🚀 Phase 4 — Industrialisation**
  - [ ] Migration potentielle vers TimescaleDB / ClickHouse selon volume
  - [ ] Pipelines d’ingestion distribués, monitoring et observabilité
  - [ ] Publication de modules analytiques avancés (backtesting, signaux ML)

## Usage (temporaire)

La CLI orchestratrice est en préparation. En attendant, pour tester les appels API :

```bash
python -m src.hyperliquid_analytics.api.test_api
```

Une fois la CLI et les services stabilisés, un script dédié (ex. `hyperliquid-analytics collect`) sera exposé via `pyproject.toml`.

---

👉 Contributions bienvenues : tests supplémentaires, nouveaux services, intégrations on-chain ! Ouvre une PR ou discute d’un plan via issues. 💬
