# 🚀 Hyperliquid Analytics Agent

Agent d’analyse technique en cours de construction autour des données Hyperliquid, avec une trajectoire orientée vers l’agrégation multichaîne on-chain et des calculs off-chain avancés.

## État actuel

- **Client Hyperliquid (async)** : appels `/info` (`candleSnapshot`, `metaAndAssetCtxs`, `userFills`) gérés, avec journalisation et validations Pydantic (`PerpMeta`, `PerpAssetContext`, `MetaAndAssetCtxsResponse`, `MarketData`…).
- **Repository DuckDB** : schéma persistant pour `perp_universe`, `margin_tables`, `perp_asset_ctxs`, transactions explicites, accès `fetch_latest` & `fetch_history`, timestamp UTC automatique.
- **Services** :
  - `AnalyticsService` orchestre le client Hyperliquid et le repository (ingestion async via `to_thread`, lectures latest/history).
  - `IndicatorService` calcule en 100 % SQL (DuckDB) les indicateurs SMA, EMA, RSI, MACD, Bollinger, ATR, Stochastic et VWAP.
  - `AnalysisPipeline` (Nouveau) : Structure pour exécuter des stratégies (`BaseStrategy`) et générer des `Signal`.
- **CLI Click** : commandes `collect snapshot`, `collect candles`, `show latest`, `show history`, `show indicator` avec option globale `--db-path`, sorties JSON prêtes pour piping.
- **Tests unitaires** : couverture des modèles, client, services (ingestion & indicateurs), repository, CLI ; suite Pytest paramétrée.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

```bash
cp .env.example .env
# Éditer .env avec vos clés/URL Hyperliquid
```

Variables principales :

- `HYPERLIQUID_ANALYTICS_BASE_URL=https://api.hyperliquid.xyz`
- `HYPERLIQUID_ANALYTICS_SYMBOLS_RAW=BTC,ETH` (etc.)
- `HYPERLIQUID_ANALYTICS_API_KEY` si nécessaire pour des endpoints privés.

## Commandes utiles

```bash
# Collecter un snapshot complet et l’enregistrer en DuckDB
python -m hyperliquid_analytics.cli collect snapshot

# Collecter des bougies OHLCV (ex : 200x 1h sur BTC)
python -m hyperliquid_analytics.cli collect candles -s BTC -t 1h -l 200
# La commande vérifie d'abord la dernière bougie stockée et ne rapatrie
# qu'en cas de gap > 1 intervalle (sauf si --limit force un backfill).

# Dernier snapshot pour un symbole
python -m hyperliquid_analytics.cli show latest -s BTC

# Historique récent (20 entrées par défaut)
python -m hyperliquid_analytics.cli show history -s BTC --limit 5

# Calculer un indicateur (ex : SMA 20 périodes en 1h)
python -m hyperliquid_analytics.cli show indicator sma -s BTC -t 1h --window 20

# Indicateurs disponibles (nov. 2025) : sma, ema, rsi, macd, bollinger, atr, stochastic, vwap

# Ingestion temps réel (WebSocket bougies + backfill auto)
python -m hyperliquid_analytics.cli scheduler ws -t 5m
# Maintient DuckDB à jour à partir du flux Hyperliquid et rattrape les trous via REST

# Scheduler basique (collecte périodique)
python -m hyperliquid_analytics.cli scheduler run -t 1h -t 4h --interval 300 --iterations 0 --snapshot
# `--interval` relance la collecte toutes les 5 minutes, `--iterations 0` = boucle infinie (Ctrl+C pour arrêter)
# utilise les symboles définis dans .env et journalise chaque collecte

# Spécifier un autre fichier DuckDB
python -m hyperliquid_analytics.cli --db-path data/dev.duckdb collect snapshot
```

## Tests

```bash
./venv/bin/python -m pytest
```

Astuce : exécuter `pip install -e .[dev]` avant les tests pour s’assurer que le package est importable avec le layout `src/`.

## Roadmap vers un système d’analytics complet

- **🐣 Phase 1 — Hyperliquid seulement (en cours)**
  - [x] Client async & modèles Pydantic
  - [x] Service + CLI de collecte/lecture
  - [x] Tests unitaires Repository / CLI / Scheduler
  - [x] Indicateurs de base (SMA/EMA, RSI, MACD, Bollinger) via DuckDB
  - [x] Extensions indicateurs : ATR, Stochastic, VWAP (calculs 100 % SQL sur `candles`)
  - [x] Scheduler d’ingestion périodique (CLI `scheduler run`)
  - [x] Architecture Pipeline (`AnalysisPipeline`, `BaseStrategy`, `Signal`)
  - [ ] Alertes locales + jobs dédiés (analysis pipeline)

- **🌐 Phase 2 — Analytics temps réel & API interne**
  - [ ] **Architecture découplée** : `scheduler ws` (collecte seule) vs `strategy run` (analyse parallèle).
  - [ ] WebSocket trades / L2 book + stockage incrémental
  - [ ] API FastAPI exposant snapshots & indicateurs
  - [ ] Tableau de bord (Streamlit / front custom)
  - [ ] Alerting (funding extrême, variations OI, divergence volume/prix)

- **🔗 Phase 3 — Extension multichaîne & on-chain**
  - [ ] Ingestion données on-chain (DEX, bridges, métriques DeFi)
  - [ ] Corrélations funding / flux on-chain
  - [ ] Normalisation multi-sources, enrichissement du repository
  - [ ] Archivage Parquet + politiques de rétention

- **🚀 Phase 4 — Industrialisation**
  - [ ] Migration possible vers TimescaleDB / ClickHouse
  - [ ] Pipelines distribués, observabilité & monitoring
  - [ ] Modules analytiques avancés (backtesting, signaux ML)

## Prochaines étapes

1.  **`scheduler ws`** : Optimiser pour l'ingestion pure (sans calculs bloquants).
2.  **`strategy run`** : Nouvelle commande dédiée qui surveille la DB et exécute les stratégies en asynchrone.
3.  **Tests** : Ajouter des mocks pour valider le flux complet WS -> DB -> Strategy.

---

👉 Contributions / feedback bienvenus : tests, intégrations de nouvelles sources, idées d’indicateurs. Ouvre une issue ou une PR pour en discuter ! 💬
