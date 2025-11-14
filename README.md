# 🚀 Hyperliquid Analytics Agent

Agent d’analyse technique en cours de construction autour des données Hyperliquid, avec une trajectoire orientée vers l’agrégation multichaîne on-chain et des calculs off-chain avancés.

## État actuel

- **Client Hyperliquid (async)** : appels `/info` (`candleSnapshot`, `metaAndAssetCtxs`, `userFills`) gérés, avec journalisation et validations Pydantic (`PerpMeta`, `PerpAssetContext`, `MetaAndAssetCtxsResponse`, `MarketData`…).
- **Repository DuckDB** : schéma persistant pour `perp_universe`, `margin_tables`, `perp_asset_ctxs`, transactions explicites, accès `fetch_latest` & `fetch_history`, timestamp UTC automatique.
- **Service d’ingestion** : `AnalyticsService` orchestre le client et le repository (insertion `asyncio.to_thread`), renvoie un snapshot + horodatage, expose les lectures `get_market_data/history`.
- **CLI Click** : commandes `collect snapshot`, `show latest`, `show history` avec option globale `--db-path`, sortie JSON.
- **Tests unitaires** : couverture des modèles, client, service (asynchrone), repository (à compléter) ; suite Pytest paramétrée.

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

# Dernier snapshot pour un symbole
python -m hyperliquid_analytics.cli show latest -s BTC

# Historique récent (20 entrées par défaut)
python -m hyperliquid_analytics.cli show history -s BTC --limit 5

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
  - [ ] Tests unitaires Repository / CLI / Scheduler
  - [ ] Calculs d’indicateurs de base (SMA/EMA, RSI, MACD, Bollinger, VWAP) via DuckDB
  - [ ] Scheduler d’ingestion périodique

- **🌐 Phase 2 — Analytics temps réel & API interne**
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

---

👉 Contributions / feedback bienvenus : tests, intégrations de nouvelles sources, idées d’indicateurs. Ouvre une issue ou une PR pour en discuter ! 💬
