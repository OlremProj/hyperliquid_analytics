"""Tests de la CLI Hyperliquid."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from hyperliquid_analytics.cli import app
from hyperliquid_analytics.services.indicator_service import IndicatorResult, IndicatorType


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
        async def compute_indicator(self, symbol: str, indicator: IndicatorType, *, window: int | None = None, limit: int | None = None):
            return IndicatorResult(
                symbol=symbol.upper(),
                indicator=indicator,
                params={"window": window, "limit": limit},
                series=[
                    (datetime(2025, 1, 1, tzinfo=timezone.utc), 100.0),
                    (datetime(2025, 1, 2, tzinfo=timezone.utc), 105.0),
                ],
            )

    monkeypatch.setattr(
        "hyperliquid_analytics.cli.IndicatorService",
        lambda analytics_service: StubIndicatorService(),
    )

    yield service


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
