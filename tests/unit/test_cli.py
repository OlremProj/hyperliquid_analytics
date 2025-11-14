"""Tests pour le module CLI."""

import asyncio

import pytest
from click.testing import CliRunner

from hyperliquid_analytics.cli import cli


@pytest.fixture
def dummy_service(monkeypatch):
    class DummyService:
        def __init__(self) -> None:
            self.market_data = {"status": "ok"}
            self.latest = {"symbol": "BTC"}
            self.save_called = False
            self.last_symbol = None

        async def save_market_data(self):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(self.market_data)
            self.save_called = True
            return await future

        async def get_market_data(self, symbol: str):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self.last_symbol = symbol
            future.set_result(self.latest)
            return await future

    service = DummyService()
    monkeypatch.setattr("hyperliquid_analytics.cli.AnalyticsService", lambda: service)
    return service


def test_cli_update_data_success(dummy_service):
    runner = CliRunner()

    result = runner.invoke(cli, ["update_data"])

    assert result.exit_code == 0
    assert "'status': 'ok'" in result.output
    assert dummy_service.save_called is True


def test_cli_get_data_success(dummy_service):
    dummy_service.latest = {"symbol": "ETH"}
    runner = CliRunner()

    result = runner.invoke(cli, ["get_data", "--symbol", "eth"])

    assert result.exit_code == 0
    assert "'symbol': 'ETH'" in result.output
    assert dummy_service.last_symbol == "eth"


def test_cli_get_data_missing_symbol(dummy_service):
    runner = CliRunner()

    result = runner.invoke(cli, ["get_data"])

    assert result.exit_code == 1
    assert "Symbol missing ! like BTC, ETH..." in result.output

