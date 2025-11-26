import asyncio
import json
from pathlib import Path

import click

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.services.analytics_service import AnalyticsService
from hyperliquid_analytics.services.indicator_service import IndicatorService, IndicatorType

from hyperliquid_analytics.config import Settings


def make_service(db_path: Path | None) -> AnalyticsService:
    return AnalyticsService(db_path=db_path)


@click.group()
@click.option("--db-path", type=click.Path(path_type=Path), default=None, help="DuckDB file path")
@click.pass_context
def app(ctx, db_path):
    ctx.ensure_object(dict)
    analytics_service = make_service(db_path)
    ctx.obj["analytics_service"] = analytics_service
    ctx.obj["indicator_service"] = IndicatorService(analytics_service)
    ctx.obj["settings"] = Settings()


@app.group("collect")
@click.pass_context
def collect_group(ctx):
    """Actions de collecte de données Hyperliquid."""
    pass

@collect_group.command("show_candles")
@click.option("--symbol", "-s", required=True, help="Symbol (BTC, ETH...)")
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    required=True,
    help="1m, 5m, 15m, 1h, 4h, 1d",
)
@click.option("--limit", "-l", type=int, default=None, show_default=False)
@click.pass_obj
def show_candles(obj, symbol, timeframe, limit):
    service: AnalyticsService = obj["analytics_service"]
    summary = asyncio.run(
        service.get_candles(
            symbol,
            timeframe,
            limit=limit,
            )
    )
    click.echo(json.dumps(summary, default=str))

@collect_group.command("candles")
@click.option("--symbol", "-s", required=True, help="Symbol (BTC, ETH...)")
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    required=True,
    help="1m, 5m, 15m, 1h, 4h, 1d",
)
@click.option("--limit", "-l", type=int, default=None, show_default=False)
@click.pass_obj
def collect_candles(obj, symbol, timeframe, limit):
    service: AnalyticsService = obj["analytics_service"]
    summary = asyncio.run(
        service.save_candles(
            symbol,
            TimeFrame(timeframe),
            limit=limit,
        )
    )
    click.echo(json.dumps(summary, default=str))

@collect_group.command("snapshot")
@click.pass_obj
def collect_snapshot(obj):
    service: AnalyticsService = obj["analytics_service"]
    snapshot, fetched_at = asyncio.run(service.save_market_data())
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "symbols": len(snapshot.asset_contexts),
                "timestamp": fetched_at.isoformat(),
            }
        )
    )


@app.group("show")
@click.pass_context
def show_group(ctx):
    """Afficher les données enregistrées."""
    pass


@show_group.command("latest")
@click.option("--symbol", "-s", required=True, help="Symbol (BTC, ETH...)")
@click.pass_obj
def show_latest(obj, symbol):
    service: AnalyticsService = obj["analytics_service"]
    ctx_value = asyncio.run(service.get_market_data(symbol))
    if ctx_value is None:
        raise click.ClickException(f"Aucun snapshot trouvé pour {symbol.upper()}")
    click.echo(json.dumps(ctx_value.model_dump(by_alias=False), default=str))


@show_group.command("history")
@click.option("--symbol", "-s", required=True)
@click.option("--limit", "-l", type=int, default=20)
@click.option("--since", type=click.DateTime(), default=None)
@click.option("--ascending/--descending", default=False)
@click.pass_obj
def show_history(obj, symbol, limit, since, ascending):
    service: AnalyticsService = obj["analytics_service"]
    history = asyncio.run(
        service.get_market_history(symbol, limit=limit, since=since, ascending=ascending)
    )
    payload = [
        {
            "timestamp": ts.isoformat(),
            **ctx_value.model_dump(by_alias=False),
        }
        for ts, ctx_value in history
    ]
    click.echo(json.dumps(payload, default=str))


@show_group.command("indicator")
@click.argument("indicator")
@click.option("--symbol", "-s", required=True)
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    default=TimeFrame.ONE_HOUR.value,
    show_default=True,
    help="1m, 5m, 15m, 1h, 4h, 1d",
)
@click.option("--window", "-w", type=int, default=20)
@click.option("--limit", "-l", type=int, default=None)
@click.pass_obj
def show_indicator(obj, indicator: str, symbol: str,timeframe: TimeFrame, window: int, limit: int | None):
    indicator_service: IndicatorService = obj["indicator_service"]
    try:
        indicator_type = IndicatorType(indicator.lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in IndicatorType)
        raise click.ClickException(f"Indicateur inconnu '{indicator}'. Valeurs possibles : {valid}") from exc

    indicator_result = asyncio.run(
        indicator_service.compute_indicator(
            symbol,
            indicator_type,
            timeframe=TimeFrame(timeframe),
            window=window,
            limit=limit,
        )
    )

    payload = {
        "symbol": indicator_result.symbol,
        "indicator": indicator_type.value,
        "metadata": indicator_result.metadata,
        "series": [
            {"timestamp": point.timestamp.isoformat(), **(point.values or {})}
            for point in indicator_result.points
        ],
    }
    click.echo(json.dumps(payload, default=str))

@app.group("scheduler")
@click.pass_context
def scheduler_group(ctx):
    """Pilote les fonctions de refresh data."""
    pass

@scheduler_group.command("run")
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    required=True,
    help="1m, 5m, 15m, 1h, 4h, 1d",
)
@click.pass_obj
def run(obj, timeframe):
    analytics_service: AnalyticsService = obj["analytics_service"]
    tf = TimeFrame(timeframe)
    for symbol in obj["settings"].symbols:
        run_task(
               analytics_service.save_candles,
               symbol=symbol,
               timeframe=tf,
        )

def run_task(runner, *, symbol: str, timeframe: TimeFrame, **kwargs):
    try:
        result = asyncio.run(runner(symbol, timeframe, **kwargs))
        click.echo(json.dumps(result, default=str))
    except Exception as exc:
        click.echo(f"[ERROR] : {symbol} :: {timeframe} -> {exc}", err=True)
        return None


if __name__ == "__main__":
    app()
