"""Tests du repository DuckDB."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame
from hyperliquid_analytics.models.perp_models import (
    MetaAndAssetCtxsResponse,
    PerpAssetContext,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def build_meta_response() -> MetaAndAssetCtxsResponse:
    return MetaAndAssetCtxsResponse.model_validate(
        [
            {
                "universe": [
                    {
                        "name": "BTC",
                        "szDecimals": 5,
                        "maxLeverage": "50",
                        "onlyIsolated": False,
                        "marginMode": None,
                        "isDelisted": False,
                    },
                    {
                        "name": "ETH",
                        "szDecimals": 4,
                        "maxLeverage": "30",
                        "onlyIsolated": False,
                        "marginMode": None,
                        "isDelisted": False,
                    },
                ],
                "marginTables": [
                    [
                        1,
                        {
                            "description": "default",
                            "marginTiers": [
                                {"lowerBound": "0", "maxLeverage": "50"},
                            ],
                        },
                    ]
                ],
            },
            [
                {
                    "dayNtlVlm": "1000",
                    "funding": "0.0001",
                    "impactPxs": ["43000", "43010"],
                    "markPx": "43005",
                    "midPx": "43007",
                    "openInterest": "120.5",
                    "oraclePx": "43002",
                    "premium": "0.0003",
                    "prevDayPx": "42000",
                },
                {
                    "dayNtlVlm": "500",
                    "funding": "0.00005",
                    "impactPxs": None,
                    "markPx": "2300",
                    "midPx": None,
                    "openInterest": "80.0",
                    "oraclePx": "2295",
                    "premium": None,
                    "prevDayPx": None,
                },
            ],
        ]
    )


def build_context() -> PerpAssetContext:
    return PerpAssetContext.model_validate(
        {
            "dayNtlVlm": "1500",
            "funding": "0.0002",
            "impactPxs": ["43001", "43002"],
            "markPx": "43001.5",
            "midPx": "43001.4",
            "openInterest": "110.2",
            "oraclePx": "43000",
            "premium": "0.00015",
            "prevDayPx": "42050",
        }
    )


def repo(tmp_path: Path) -> PerpRepository:
    db_path = tmp_path / "perp.duckdb"
    return PerpRepository(db_path=db_path)


@pytest.fixture
def temporary_repo():
    with TemporaryDirectory() as tmpdir:
        repo_instance = repo(Path(tmpdir))
        try:
            yield repo_instance
        finally:
            repo_instance.close()


def make_candle(ts: datetime, price: float) -> OHLCVData:
    return OHLCVData(
        symbol="BTC",
        timestamp=ts,
        open=price,
        high=price + 5,
        low=price - 5,
        close=price + 1,
        volume=10.0,
    )


def test_save_meta_persists_universe_and_tables(temporary_repo: PerpRepository):
    meta_response = build_meta_response()

    temporary_repo.save_meta(meta_response.meta)

    universe_rows = temporary_repo._conn.execute(
        "SELECT symbol, sz_decimals, max_leverage FROM perp_universe ORDER BY symbol"
    ).fetchall()
    assert universe_rows == [
        ("BTC", 5, 50.0),
        ("ETH", 4, 30.0),
    ]

    margin_rows = temporary_repo._conn.execute(
        "SELECT margin_id, tier_index, lower_bound, max_leverage FROM margin_tables"
    ).fetchall()
    assert margin_rows == [(1, 0, 0.0, 50.0)]


def test_save_asset_contexts_handles_optional_fields(temporary_repo: PerpRepository):
    fetched_at = datetime.now(timezone.utc)
    ctx = build_context()

    temporary_repo.save_asset_contexts([("BTC", ctx)], fetched_at=fetched_at)

    stored = temporary_repo._conn.execute(
        """
        SELECT symbol, fetched_at, mark_px, mid_px, premium, impact_bid, impact_ask
        FROM perp_asset_ctxs
        """
    ).fetchone()

    assert stored[0] == "BTC"
    assert stored[1] == fetched_at
    assert stored[2] == float(ctx.mark_price)
    assert stored[3] == float(ctx.mid_price)
    assert stored[4] == float(ctx.premium)
    assert stored[5:] == (float(ctx.impact_prices[0]), float(ctx.impact_prices[1]))


def test_fetch_latest_returns_context(temporary_repo: PerpRepository):
    fetched_at = datetime.now(timezone.utc)
    ctx = build_context()
    temporary_repo.save_asset_contexts([("BTC", ctx)], fetched_at=fetched_at)

    latest = temporary_repo.fetch_latest("btc")

    assert latest is not None
    assert latest.mark_price == ctx.mark_price
    assert latest.impact_prices == ctx.impact_prices


def test_fetch_history_supports_limit_and_ordering(temporary_repo: PerpRepository):
    base_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    ctx = build_context()
    for i in range(3):
        temporary_repo.save_asset_contexts(
            [("BTC", ctx)],
            fetched_at=base_time + timedelta(minutes=i),
        )

    history = temporary_repo.fetch_history("btc", limit=2, ascending=False)

    assert len(history) == 2
    first_ts, _ = history[0]
    second_ts, _ = history[1]
    assert first_ts > second_ts


def test_fetch_latest_candle_timestamp_returns_latest_entry(temporary_repo: PerpRepository):
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    candles = [
        make_candle(base_time, 100.0),
        make_candle(base_time + timedelta(hours=1), 110.0),
    ]
    temporary_repo.save_candles("BTC", "1h", candles)

    latest = temporary_repo.fetch_latest_candle_timestamp("btc", "1h")

    assert latest == candles[-1].timestamp


def test_fetch_candles_respects_limit_and_sorting(temporary_repo: PerpRepository):
    base_time = datetime.now(timezone.utc) - timedelta(hours=4)
    candles = [
        make_candle(base_time + timedelta(hours=idx), 100.0 + idx) for idx in range(4)
    ]
    temporary_repo.save_candles("BTC", "1h", candles)

    recent = temporary_repo.fetch_candles(
        "btc",
        TimeFrame.ONE_HOUR,
        limit=2,
        ascending=False,
    )
    assert len(recent) == 2
    assert recent[0].timestamp > recent[1].timestamp

    since = base_time + timedelta(hours=1, minutes=30)
    asc = temporary_repo.fetch_candles(
        "btc",
        TimeFrame.ONE_HOUR,
        since=since,
        ascending=True,
    )
    assert all(c.timestamp >= since for c in asc)
    assert asc[0].timestamp <= asc[-1].timestamp


def test_save_snapshot_combines_meta_and_contexts(temporary_repo: PerpRepository):
    snapshot = build_meta_response()
    fetched_at = datetime.now(timezone.utc)

    temporary_repo.save_snapshot(snapshot, fetched_at=fetched_at)

    universe_count = temporary_repo._conn.execute(
        "SELECT COUNT(*) FROM perp_universe"
    ).fetchone()[0]
    ctx_count = temporary_repo._conn.execute(
        "SELECT COUNT(*) FROM perp_asset_ctxs"
    ).fetchone()[0]

    assert universe_count == len(snapshot.meta.universe)
    assert ctx_count == len(snapshot.asset_contexts)
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


def test_fetch_history_supports_filters_and_ordering():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        base_time = datetime.now(timezone.utc)
        repo.save_asset_contexts(
            [
                ("BTC", build_context(Decimal("10"))),
            ],
            base_time - timedelta(minutes=3),
        )
        repo.save_asset_contexts(
            [
                ("BTC", build_context(Decimal("20"))),
            ],
            base_time - timedelta(minutes=2),
        )
        repo.save_asset_contexts(
            [
                ("BTC", build_context(Decimal("30"))),
            ],
            base_time - timedelta(minutes=1),
        )

        history = repo.fetch_history("btc", limit=2)
        assert [ctx.day_notional_volume for _, ctx in history] == [
            Decimal("30"),
            Decimal("20"),
        ]

        since_time = base_time - timedelta(minutes=2, seconds=30)
        asc_history = repo.fetch_history("btc", since=since_time, ascending=True)
        assert [ctx.day_notional_volume for _, ctx in asc_history] == [
            Decimal("20"),
            Decimal("30"),
        ]

        repo.close()


def test_close_closes_connection():
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perp.duckdb"
        repo = PerpRepository(db_path=db_path)

        repo.close()

        with pytest.raises(duckdb.Error):
            repo._conn.execute("SELECT 1")

