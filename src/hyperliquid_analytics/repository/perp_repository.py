import duckdb
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame
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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                timestamp TIMESTAMPTZ,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,       
                volume DOUBLE,
                timeframe text,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );
            """
        )

    def save_candles(self, symbol: str, timeframe: str, candles: Iterable[OHLCVData]) -> None:
        rows = [
            (
                symbol.upper(),
                candle.timestamp,
                float(candle.open),
                float(candle.high),
                float(candle.low),
                float(candle.close),
                float(candle.volume),
                timeframe,
            )
            for candle in candles
        ]
        if not rows:
            return

        self._conn.execute("BEGIN")
        try:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO candles
                    (symbol, timestamp, open, high, low, close, volume, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def fetch_latest_candle_timestamp(self, symbol: str, timeframe: str) -> datetime | None:
        row = self._conn.execute(
            """
            SELECT timestamp
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (symbol.upper(), timeframe),
        ).fetchone()
        if row:
            return row[0]
        return None

    def save_meta(self, meta: PerpMeta) -> None:
        self._conn.execute("BEGIN")
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
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
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

        self._conn.execute("BEGIN")
        try:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO perp_asset_ctxs
                    (symbol, fetched_at, day_ntl_vlm, funding, mark_px, mid_px, open_interest,
                     oracle_px, premium, prev_day_px, impact_bid, impact_ask)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def save_snapshot(self, snapshot: MetaAndAssetCtxsResponse, fetched_at: datetime) -> None:
        self.save_meta(snapshot.meta)
        self.save_asset_contexts(
            ((asset.name, ctx) for asset, ctx in zip(snapshot.meta.universe, snapshot.asset_contexts)),
            fetched_at=fetched_at or datetime.now(timezone.utc),
        )
    
    def fetch_candles(
            self,
            symbol: str,
            timeframe: TimeFrame | str,
            *,
            limit: int | None = None,
            since: datetime | None = None,
            ascending: bool = True,
        ) -> list[OHLCVData]:
        tf = timeframe.value if isinstance(timeframe, TimeFrame) else str(timeframe)
        params = [symbol.upper(), tf]
        filters = ["symbol = ?", "timeframe = ?"]

        if since is not None:
            filters.append("timestamp >= ?")
            params.append(since)

        order = "ASC" if ascending else "DESC"
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE {' AND '.join(filters)}
            ORDER BY timestamp {order}
            {limit_clause}
        """
        rows = self._conn.execute(query, params).fetchall()
        return [
            OHLCVData(
                symbol=symbol.upper(),
                timestamp=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5],
            )
            for row in rows
        ]

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

        return self.mapping_prep_asset_context(row)


    def fetch_history(
        self,
        symbol: str,
        *,
        since: datetime | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> list[tuple[datetime, PerpAssetContext]]:
        clauses = ["symbol = ?"]
        params: list[Any] = [symbol.upper()]
        if since is not None:
            clauses.append("fetched_at >= ?")
            params.append(since)
        order = "ASC" if ascending else "DESC"
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)
        query = f"""
            SELECT fetched_at, day_ntl_vlm, funding, mark_px, mid_px, open_interest, oracle_px,
                   premium, prev_day_px, impact_bid, impact_ask
            FROM perp_asset_ctxs
            WHERE {' AND '.join(clauses)}
            ORDER BY fetched_at {order}
            {limit_clause}
        """
        rows = self._conn.execute(query, params).fetchall()
        history: list[tuple[datetime, PerpAssetContext]] = []
        for row in rows:
            fetched_at_dt = row[0]
            context = self.mapping_prep_asset_context(row[1:])
            history.append((fetched_at_dt, context))
        return history

    def mapping_prep_asset_context(self, row):
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
