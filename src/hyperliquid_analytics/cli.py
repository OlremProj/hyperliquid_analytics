import asyncio
import json
 
from pathlib import Path

import click

from hyperliquid_analytics.services.analytics_service import AnalyticsService

def make_service(db_path: Path | None) -> AnalyticsService:
    return AnalyticsService(db_path=db_path)

@click.group()
@click.option("--db-path", type=click.Path(path_type=Path), default=None, help="DuckDB file path")
@click.pass_context
def app(ctx, db_path):
    ctx.ensure_object(dict)
    ctx.obj["service"] = make_service(db_path)

@app.group("collect")
@click.pass_context
def collect_group(ctx):
    """Actions de collecte de données Hyperliquid."""
    pass

@collect_group.command("snapshot")
@click.pass_obj
def collect_snapshot(obj):
    service: AnalyticsService = obj["service"]
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
    service: AnalyticsService = obj["service"]
    ctx = asyncio.run(service.get_market_data(symbol))
    if ctx is None:
        raise click.ClickException(f"Aucun snapshot trouvé pour {symbol.upper()}")
    click.echo(json.dumps(ctx.model_dump(by_alias=False), default=str))

@show_group.command("history")
@click.option("--symbol", "-s", required=True)
@click.option("--limit", "-l", type=int, default=20)
@click.option("--since", type=click.DateTime(), default=None)
@click.option("--ascending/--descending", default=False)
@click.pass_obj
def show_history(obj, symbol, limit, since, ascending):
    service: AnalyticsService = obj["service"]
    history = asyncio.run(
        service.get_market_history(symbol, limit=limit, since=since, ascending=ascending)
    )
    payload = [
        {
            "timestamp": ts.isoformat(),
            **ctx.model_dump(by_alias=False),
        }
        for ts, ctx in history
    ]
    click.echo(json.dumps(payload, default=str))

if __name__ == '__main__':
    app()
