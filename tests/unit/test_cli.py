"""Tests de la CLI Hyperliquid."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from hyperliquid_analytics.cli import app
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorPoint,
    IndicatorResult,
    IndicatorType,
)


class SnapshotStub(SimpleNamespace):
    asset_contexts: list


class ContextStub:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, *, by_alias: bool = False) -> dict:
        return self._payload


@pytest.fixture
def runner():
    return CliRunner()


def make_stub_service():
    class StubService:
        def __init__(self):
            self.saved = False
            self.perp_repository = SimpleNamespace()

        async def save_market_data(self):
            self.saved = True
            snapshot = SnapshotStub(asset_contexts=[object(), object()])
            fetched_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
            return snapshot, fetched_at

        async def get_market_data(self, symbol: str):
            if symbol == "missing":
                return None
            return ContextStub({"symbol": symbol.upper(), "mark_price": 123.4})

        async def get_market_history(self, symbol: str, *, limit: int | None = None, since=None, ascending=False):
            return [
                (
                    datetime(2025, 1, 1, tzinfo=timezone.utc),
                    ContextStub({"symbol": symbol.upper(), "mark_price": 100.0}),
                ),
                (
                    datetime(2025, 1, 2, tzinfo=timezone.utc),
                    ContextStub({"symbol": symbol.upper(), "mark_price": 105.0}),
                ),
            ][: limit or 2]

    return StubService()


@pytest.fixture(autouse=True)
def stub_service(monkeypatch):
    service = make_stub_service()
    monkeypatch.setattr("hyperliquid_analytics.cli.make_service", lambda db_path: service)

    class StubIndicatorService:
        def __init__(self):
            self.calls: list[dict] = []

        async def compute_indicator(
            self,
            symbol: str,
            indicator: IndicatorType,
            *,
            window: int | None = None,
            limit: int | None = None,
        ):
            self.calls.append(
                {
                    "symbol": symbol,
                    "indicator": indicator,
                    "window": window,
                    "limit": limit,
                }
            )
            return IndicatorResult(
                symbol=symbol.upper(),
                indicator=indicator,
                metadata={"window": window, "limit": limit},
                points=[
                    IndicatorPoint(
                        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                        values={"value": 100.0},
                    ),
                    IndicatorPoint(
                        timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
                        values={"macd": 1.0, "signal": 0.5, "hist": 0.5},
                    ),
                ],
            )

    indicator_stub = StubIndicatorService()
    service._indicator_stub = indicator_stub
    monkeypatch.setattr(
        "hyperliquid_analytics.cli.IndicatorService",
        lambda analytics_service: indicator_stub,
    )

    yield service


@pytest.fixture
def indicator_stub(stub_service):
    return stub_service._indicator_stub


def test_collect_snapshot_outputs_json(runner, stub_service):
    result = runner.invoke(app, ["collect", "snapshot"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["symbols"] == 2
    assert payload["timestamp"] == "2025-01-01T00:00:00+00:00"


def test_show_latest_requires_symbol(runner):
    result = runner.invoke(app, ["show", "latest"])

    assert result.exit_code != 0
    assert "Missing option '--symbol'" in result.output


def test_show_latest_returns_context_payload(runner):
    result = runner.invoke(app, ["show", "latest", "-s", "btc"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["symbol"] == "BTC"
    assert payload["mark_price"] == 123.4


def test_show_latest_with_missing_symbol_returns_error(runner):
    result = runner.invoke(app, ["show", "latest", "-s", "missing"])

    assert result.exit_code != 0
    assert "Aucun snapshot trouvé pour MISSING" in result.output


def test_show_history_outputs_series(runner):
    result = runner.invoke(app, ["show", "history", "-s", "eth", "--limit", "1"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)
    assert len(data) == 1
    entry = data[0]
    assert entry["symbol"] == "ETH"
    assert entry["mark_price"] == 100.0


def test_show_indicator_serializes_series_and_passes_args(runner, indicator_stub):
    result = runner.invoke(
        app,
        [
            "show",
            "indicator",
            "EMA",
            "-s",
            "btc",
            "--window",
            "10",
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["symbol"] == "BTC"
    assert payload["indicator"] == IndicatorType.EMA.value
    assert payload["metadata"] == {"window": 10, "limit": 2}
    assert payload["series"][0]["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert payload["series"][0]["value"] == 100.0
    second = payload["series"][1]
    assert second["timestamp"] == "2025-01-02T00:00:00+00:00"
    assert second["macd"] == 1.0
    assert "value" not in second

    call = indicator_stub.calls[-1]
    assert call["symbol"] == "btc"
    assert call["indicator"] is IndicatorType.EMA
    assert call["window"] == 10
    assert call["limit"] == 2


def test_show_indicator_unknown_indicator_returns_error(runner):
    result = runner.invoke(app, ["show", "indicator", "unknown", "-s", "btc"])

    assert result.exit_code != 0
    assert "Indicateur inconnu 'unknown'" in result.output


def test_show_indicator_requires_symbol_option(runner):
    result = runner.invoke(app, ["show", "indicator", "sma"])

    assert result.exit_code != 0
    assert "Missing option '--symbol'" in result.output
