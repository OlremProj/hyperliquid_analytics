from decimal import Decimal
from typing import List, Tuple

from pydantic import BaseModel, Field, model_validator, ConfigDict


class MarginTier(BaseModel):
    lower_bound: Decimal = Field(alias="lowerBound")
    max_leverage: Decimal = Field(alias="maxLeverage")


class MarginTable(BaseModel):
    description: str
    margin_tiers: List[MarginTier] = Field(alias="marginTiers")


class MarginTableEntry(BaseModel):
    identifier: int
    table: MarginTable

    @model_validator(mode="before")
    @classmethod
    def _coerce_tuple(cls, value):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"identifier": value[0], "table": value[1]}
        return value


class PerpUniverseAsset(BaseModel):
    name: str
    sz_decimals: int = Field(alias="szDecimals")
    max_leverage: Decimal = Field(alias="maxLeverage")
    only_isolated: bool | None = Field(default=None, alias="onlyIsolated")
    margin_mode: str | None = Field(default=None, alias="marginMode")
    is_delisted: bool | None = Field(default=None, alias="isDelisted")


class PerpMeta(BaseModel):
    universe: List[PerpUniverseAsset]
    margin_tables: List[MarginTableEntry] = Field(default_factory=list, alias="marginTables")
    model_config = ConfigDict(populate_by_name=True)


class PerpAssetContext(BaseModel):
    day_notional_volume: Decimal = Field(alias="dayNtlVlm")
    funding: Decimal
    impact_prices: tuple[Decimal, Decimal] | None = Field(default=None, alias="impactPxs")
    mark_price: Decimal = Field(alias="markPx")
    mid_price: Decimal | None = Field(default=None, alias="midPx")
    open_interest: Decimal = Field(alias="openInterest")
    oracle_price: Decimal = Field(alias="oraclePx")
    premium: Decimal | None = Field(default=None)
    previous_day_price: Decimal | None = Field(default=None, alias="prevDayPx")

class MetaAndAssetCtxsResponse(BaseModel):
    meta: PerpMeta
    asset_contexts: List[PerpAssetContext]

    @model_validator(mode="before")
    @classmethod
    def _from_raw(cls, value):
        if isinstance(value, list) and len(value) == 2:
            return {"meta": value[0], "asset_contexts": value[1]}
        return value
