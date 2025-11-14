"""Tests pour PerpRepository."""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import duckdb
import pytest
from tempfile import TemporaryDirectory

from hyperliquid_analytics.models.perp_models import (
    MetaAndAssetCtxsResponse,
    PerpAssetContext,
    PerpMeta,
    PerpUniverseAsset,
    MarginTableEntry,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def build_meta() -> PerpMeta:
    universe = [
        PerpUniverseAsset.model_validate(
            {
                "name": "BTC",
                "szDecimals": 4,
                "maxLeverage": "50",
                "onlyIsolated": False,
                "marginMode": "cross",
                "isDelisted": False,
            }
        )
    ]
    margin_tables = [
        MarginTableEntry.model_validate(
            (
                1,
                {
                    "description": "standard",
                    "marginTiers": [
                        {"lowerBound": "0", "maxLeverage": "50"},
                        {"lowerBound": "10", "maxLeverage": "25"},
                    ],
                },
            )
        )
    ]
    return PerpMeta(universe=universe, margin_tables=margin_tables)


def build_context(day: Decimal = Decimal("1000")) -> PerpAssetContext:
    return PerpAssetContext.model_validate(
        {
            "dayNtlVlm": str(day),
            "funding": "0.0001",
            "impactPxs": ["200.5", "201.5"],
            "markPx": "200.0",
            "midPx": "200.5",
            "openInterest": "150",
            "oraclePx": "199.9",
            "premium": "0.01",
            "prevDayPx": "198.0",
        }
    )


def test_init_schema_creates_tables():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        with duckdb.connect(str(db_path)) as conn:
            tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}

        assert {"perp_universe", "margin_tables", "perp_asset_ctxs"} <= tables

        repo.close()


def test_save_meta_inserts_universe_and_margin_tables():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)
        meta = build_meta()

        repo.save_meta(meta)

        with duckdb.connect(str(db_path)) as conn:
            universe_rows = conn.execute(
                "SELECT symbol, sz_decimals, max_leverage, margin_mode, only_isolated, is_delisted FROM perp_universe"
            ).fetchall()
            margin_rows = conn.execute(
                "SELECT margin_id, tier_index, lower_bound, max_leverage, description FROM margin_tables ORDER BY tier_index"
            ).fetchall()

        assert universe_rows == [
            ("BTC", 4, 50.0, "cross", False, False),
        ]
        assert margin_rows == [
            (1, 0, 0.0, 50.0, "standard"),
            (1, 1, 10.0, 25.0, "standard"),
        ]

        repo.close()


def test_save_asset_contexts_persists_rows():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        fetched_at = datetime.now(timezone.utc)
        context = build_context()
        repo.save_asset_contexts([("BTC", context)], fetched_at)

        with duckdb.connect(str(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT symbol, fetched_at, day_ntl_vlm, funding, mark_px, mid_px,
                       open_interest, oracle_px, premium, prev_day_px, impact_bid, impact_ask
                FROM perp_asset_ctxs
                """
            ).fetchall()

        assert len(rows) == 1
        (symbol, stored_ts, day_ntl_vlm, _, _, _, _, _, _, _, impact_bid, impact_ask) = rows[0]
        assert symbol == "BTC"
        assert isinstance(stored_ts, datetime)
        assert day_ntl_vlm == pytest.approx(1000.0)
        assert impact_bid == pytest.approx(200.5)
        assert impact_ask == pytest.approx(201.5)

        repo.close()


def test_save_snapshot_combines_meta_and_contexts():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        meta = build_meta()
        contexts = [build_context(Decimal("1234.56"))]
        response = MetaAndAssetCtxsResponse(meta=meta, asset_contexts=contexts)

        fetched_at = datetime.now(timezone.utc)
        repo.save_snapshot(response, fetched_at)

        with duckdb.connect(str(db_path)) as conn:
            universe_count = conn.execute("SELECT COUNT(*) FROM perp_universe").fetchone()[0]
            ctx_rows = conn.execute(
                "SELECT symbol, day_ntl_vlm FROM perp_asset_ctxs"
            ).fetchall()

        assert universe_count == 1
        assert len(ctx_rows) == 1
        assert ctx_rows[0][0] == "BTC"
        assert ctx_rows[0][1] == pytest.approx(1234.56)

        repo.close()


def test_fetch_latest_returns_latest_context():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        first_repo = PerpRepository(db_path=db_path)

        older = datetime.now(timezone.utc) - timedelta(minutes=5)
        newer = datetime.now(timezone.utc)
        first_repo.save_asset_contexts([("BTC", build_context(Decimal("10")))], older)
        first_repo.close()

        repo = PerpRepository(db_path=db_path)
        repo.save_asset_contexts([("BTC", build_context(Decimal("20")))], newer)
        repo.close()

        fetch_repo = PerpRepository(db_path=db_path)
        latest = fetch_repo.fetch_latest("btc")
        assert latest is not None
        assert latest.day_notional_volume == Decimal("20")
        assert latest.impact_prices == (Decimal("200.5"), Decimal("201.5"))

        fetch_repo.close()


def test_close_closes_connection():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        repo.close()

        with pytest.raises(duckdb.Error):
            repo._conn.execute("SELECT 1")

