"""Tests pour le module CLI."""

import json
from datetime import datetime, timezone

import pytest
from click.testing import CliRunner

from hyperliquid_analytics.cli import app


class DummyContext:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, *, by_alias: bool = False) -> dict:
        return self._payload

class DummySnapshot:
    def __init__(self, count: int):
        self.asset_contexts = [object() for _ in range(count)]

@pytest.fixture
def stub_service(monkeypatch):
    class StubService:
        def __init__(self) -> None:
            self.snapshot = DummySnapshot(count=3)
            self.fetched_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
            self.latest_result: DummyContext | None = DummyContext({"symbol": "BTC"})
            self.history_result = [
                (
                    self.fetched_at,
                    DummyContext({"symbol": "BTC", "mark_price": 42}),
                ),
            ]
            self.save_called = False
            self.latest_symbol = None
            self.history_args: dict | None = None

        async def save_market_data(self):
            self.save_called = True
            return self.snapshot, self.fetched_at

        async def get_market_data(self, symbol: str):
            self.latest_symbol = symbol
            return self.latest_result

        async def get_market_history(
            self,
            symbol: str,
            *,
            since: datetime | None = None,
            limit: int | None = None,
            ascending: bool = False,
        ):
            self.history_args = {
                "symbol": symbol,
                "since": since,
                "limit": limit,
                "ascending": ascending,
            }
            return self.history_result

    service = StubService()
    monkeypatch.setattr(
        "hyperliquid_analytics.cli.AnalyticsService",
        lambda db_path=None: service,
    )
    return service

def test_collect_snapshot_outputs_summary(stub_service):
    runner = CliRunner()

    result = runner.invoke(app, ["collect", "snapshot"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["symbols"] == len(stub_service.snapshot.asset_contexts)
    assert payload["timestamp"] == stub_service.fetched_at.isoformat()
    assert stub_service.save_called is True

def test_show_latest_prints_context(stub_service):
    stub_service.latest_result = DummyContext({"symbol": "ETH"})
    runner = CliRunner()

    result = runner.invoke(app, ["show", "latest", "--symbol", "eth"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["symbol"] == "ETH"
    assert stub_service.latest_symbol == "eth"

def test_show_history_prints_json_array(stub_service):
    history_time = datetime(2024, 1, 2, tzinfo=timezone.utc)
    stub_service.history_result = [
        (history_time, DummyContext({"symbol": "ARB", "mark_price": 12.5})),
    ]
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["show", "history", "--symbol", "arb", "--limit", "10", "--ascending"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert payload[0]["timestamp"] == history_time.isoformat()
    assert payload[0]["symbol"] == "ARB"
    assert stub_service.history_args == {
        "symbol": "arb",
        "since": None,
        "limit": 10,
        "ascending": True,
    }

def test_show_latest_errors_when_symbol_not_found(stub_service):
    stub_service.latest_result = None
    runner = CliRunner()

    result = runner.invoke(app, ["show", "latest", "--symbol", "eth"])

    assert result.exit_code == 1
    assert "Aucun snapshot trouvé pour ETH" in result.output

