import asyncio
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from typing import Awaitable, Callable, Sequence
import websockets
import click
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential
from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame
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

@scheduler_group.command("ws")
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    default=TimeFrame.ONE_HOUR.value,
    show_default=True,
    help="1m, 5m, 15m, 1h, 4h, 1d",
)
@click.pass_obj 
def run_ws(obj, timeframe: TimeFrame):
    settings = obj["settings"]
    analytics_service: AnalyticsService = obj["analytics_service"]
    asyncio.run(ws_runner(analytics_service, settings, timeframe))


async def ws_runner(analytics_service: AnalyticsService, settings: Settings, timeframe: TimeFrame):
    last_by_symbol: dict[tuple[str,str], datetime] = {}

    for sym in settings.symbols:
        ts = await asyncio.to_thread(
                analytics_service.get_latest_candle_timestamp
                sym,
                timeframe,
                )
        if ts:
            last_by_symbol[(sym, timeframe)] = ts

    async for attempt in AsyncRetrying(
                stop=stop_after_attempt(5),
                wait=wait_exponential(min=1, max=30),
            ):
        
        with attempt:
        
            async with websockets.connect(settings.ws_uri) as ws:
                await subscribe(ws, settings.symbols, timeframe)
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("channel") != "candle":
                        continue
                    data = message["data"]
                    symbol = data["s"]
                    i = data["i"]
                    timeframe = TimeFrame(data["i"])
                    start_ts = datetime.fromtimestamp(data["t"] / 1000, tz=timezone.utc)
                    key = (symbol, i)
                    last = last_by_symbol.get(key)
                    if last and start_ts > last:
                        # lancer un to_thread qui rattrape les element manquant pendnat qu'on continue à trainer les element des ws '
                    candle = OHLCVData(
                            symbol=symbol,
                            timestamp=start_ts,
                            open=float(data["o"]),
                            high=float(data["h"]),
                            low=float(data["l"]),
                            close=float(data["c"]),
                            volume=float(data["v"]),
                    )
                    click.echo(f"[ws] {candle}")


  
async def subscribe(ws, symbols, timeframe):
    for sym in symbols:
        await ws.send(
                    json.dumps({
                        "method":"subscribe",
                        "subscription": {"type": "candle", "coin": sym, "interval": timeframe}
                        })
                )

@scheduler_group.command("run")
@click.option(
    "--timeframe",
    "-t",
    type=click.Choice([tf.value for tf in TimeFrame]),
    multiple=True,
    required=True,
    help="One or more timeframes (1m, 5m, 15m, 1h, 4h, 1d)",
)
@click.option(
    "--interval",
    "-i",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds to wait between iterations (0 to run once)",
)
@click.option(
    "--iterations",
    type=int,
    default=0,
    show_default=True,
    help="Max iterations when using --interval (0 = infinite until Ctrl+C)",
)
@click.option(
    "--snapshot/--no-snapshot",
    default=False,
    show_default=True,
    help="Collect market snapshot before candles",
)
@click.pass_obj
def run(obj, timeframe: Sequence[str], interval: float, iterations: int, snapshot: bool):
    analytics_service: AnalyticsService = obj["analytics_service"]
    settings: Settings = obj["settings"]
    symbols = settings.symbols
    timeframes = [TimeFrame(value) for value in timeframe]

    def run_iteration() -> None:
        if snapshot:
            result = run_async_task(
                "snapshot",
                lambda: analytics_service.save_market_data(),
            )
            if result:
                _, fetched_at = result
                click.echo(f"[snapshot] fetched_at={fetched_at.isoformat()}")

        for symbol in symbols:
            for tf in timeframes:
                result = run_async_task(
                    f"candles:{symbol}:{tf.value}",
                    lambda symbol=symbol, tf=tf: analytics_service.save_candles(symbol, tf),
                )
                if isinstance(result, dict):
                    status = result.get("status", "unknown")
                    fetched = result.get("fetched")
                    click.echo(
                        f"[candles] symbol={symbol} timeframe={tf.value} status={status} fetched={fetched}"
                    )
                elif result is not None:
                    click.echo(f"[candles] symbol={symbol} timeframe={tf.value} completed")

    iteration = 0
    try:
        while True:
            iteration += 1
            run_iteration()
            if interval <= 0:
                break
            if iterations > 0 and iteration >= iterations:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("Scheduler stopped.")


def run_async_task(tag: str, coro_factory: Callable[[], Awaitable[object]]):
    try:
        return asyncio.run(coro_factory())
    except Exception as exc:
        click.echo(f"[{tag}] ERROR: {exc}", err=True)
        return None


if __name__ == "__main__":
    app()
