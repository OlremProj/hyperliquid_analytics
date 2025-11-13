import duckdb
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from hyperliquid_analytics.models.perp_models import (
    MetaAndAssetCtxsResponse,
    PerpAssetContext,
    PerpMeta,
)

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "hyperliquid.duckdb"

    
class PerpRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        db_file = db_path or DB_PATH
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(database=str(db_file), read_only=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS perp_universe (
                symbol TEXT PRIMARY KEY,
                sz_decimals INTEGER,
                max_leverage DOUBLE,
                margin_mode TEXT,
                only_isolated BOOLEAN,
                is_delisted BOOLEAN,
                ingested_at TIMESTAMPTZ DEFAULT current_timestamp
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS margin_tables (
                margin_id INTEGER,
                tier_index INTEGER,
                lower_bound DOUBLE,
                max_leverage DOUBLE,
                description TEXT,
                PRIMARY KEY (margin_id, tier_index)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS perp_asset_ctxs (
                symbol TEXT,
                fetched_at TIMESTAMPTZ,
                day_ntl_vlm DOUBLE,
                funding DOUBLE,
                mark_px DOUBLE,
                mid_px DOUBLE,
                open_interest DOUBLE,
                oracle_px DOUBLE,
                premium DOUBLE,
                prev_day_px DOUBLE,
                impact_bid DOUBLE,
                impact_ask DOUBLE,
                PRIMARY KEY (symbol, fetched_at)
            );
            """
        )

    def save_meta(self, meta: PerpMeta) -> None:
        try:
            self._conn.execute("DELETE FROM margin_tables;")
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO perp_universe
                    (symbol, sz_decimals, max_leverage, margin_mode, only_isolated, is_delisted)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        asset.name,
                        asset.sz_decimals,
                        float(asset.max_leverage),
                        asset.margin_mode,
                        asset.only_isolated,
                        asset.is_delisted,
                    )
                    for asset in meta.universe
                ],
            )
            self._conn.executemany(
                """
                INSERT INTO margin_tables (margin_id, tier_index, lower_bound, max_leverage, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        entry.identifier,
                        idx,
                        float(tier.lower_bound),
                        float(tier.max_leverage),
                        entry.table.description,
                    )
                    for entry in meta.margin_tables
                    for idx, tier in enumerate(entry.table.margin_tiers)
                ],
            )
        except Exception:
            raise

    def save_asset_contexts(
        self, contexts: Iterable[tuple[str, PerpAssetContext]], fetched_at: datetime
    ) -> None:
        rows = []
        for symbol, ctx in contexts:
            impact_bid, impact_ask = (ctx.impact_prices or (None, None))
            rows.append(
                (
                    symbol,
                    fetched_at,
                    float(ctx.day_notional_volume),
                    float(ctx.funding),
                    float(ctx.mark_price),
                    float(ctx.mid_price) if ctx.mid_price is not None else None,
                    float(ctx.open_interest),
                    float(ctx.oracle_price),
                    float(ctx.premium) if ctx.premium is not None else None,
                    float(ctx.previous_day_price) if ctx.previous_day_price is not None else None,
                    float(impact_bid) if impact_bid is not None else None,
                    float(impact_ask) if impact_ask is not None else None,
                )
            )

        if not rows:
            return

        with self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO perp_asset_ctxs
                    (symbol, fetched_at, day_ntl_vlm, funding, mark_px, mid_px, open_interest,
                     oracle_px, premium, prev_day_px, impact_bid, impact_ask)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def save_snapshot(self, snapshot: MetaAndAssetCtxsResponse, fetched_at: datetime) -> None:
        self.save_meta(snapshot.meta)
        self.save_asset_contexts(
            ((asset.name, ctx) for asset, ctx in zip(snapshot.meta.universe, snapshot.asset_contexts)),
            fetched_at=fetched_at,
        )

    def fetch_latest(self, symbol: str) -> PerpAssetContext | None:
        row = self._conn.execute(
            """
            SELECT day_ntl_vlm, funding, mark_px, mid_px, open_interest, oracle_px,
                   premium, prev_day_px, impact_bid, impact_ask
            FROM perp_asset_ctxs
            WHERE symbol = ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            [symbol.upper()],
        ).fetchone()

        if row is None:
            return None

        day_ntl_vlm, funding, mark_px, mid_px, open_interest, oracle_px, premium, prev_day_px, impact_bid, impact_ask = row
        return PerpAssetContext(
            day_notional_volume=day_ntl_vlm,
            funding=funding,
            mark_price=mark_px,
            mid_price=mid_px,
            open_interest=open_interest,
            oracle_price=oracle_px,
            premium=premium,
            previous_day_price=prev_day_px,
            impact_prices=(impact_bid, impact_ask) if impact_bid is not None and impact_ask is not None else None,
        )

    def close(self) -> None:
        self._conn.close()
