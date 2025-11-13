import click
import asyncio
from hyperliquid_analytics.services.analytics_service import AnalyticsService

@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    ctx.obj["service"] = AnalyticsService()
    pass

@cli.command("update_data")
@click.pass_obj
def update_data(obj):
    service: AnalyticsService = obj["service"]
    data = asyncio.run(service.save_market_data())
    click.echo(data)
    return

@cli.command("get_data")
@click.option("--symbol", "-s")
@click.pass_obj
def get_data(obj, symbol:str):
    service: AnalyticsService = obj["service"]
    if not symbol:
        raise click.ClickException("Symbol missing ! like BTC, ETH...")
    data = asyncio.run(service.get_market_data(symbol))
    click.echo(data)
    return

if __name__ == '__main__':
    cli()
