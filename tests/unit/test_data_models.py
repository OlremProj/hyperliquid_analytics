"""Tests pour les modèles de données."""

from datetime import datetime
import pytest
from pydantic import ValidationError
from hyperliquid_analytics.models.data_models import (
    TimeFrame,
    OHLCVData,
    MarketData,
)


def test_ohlcv_data_valid():
    """Test création d'une bougie valide."""
    candle = OHLCVData(
        timestamp=datetime(2024, 1, 1, 12, 0),
        open=100.0,
        high=110.0,
        low=95.0,
        close=105.0,
        volume=1000.0,
    )
    
    assert candle.open == 100.0
    assert candle.high == 110.0
    assert candle.low == 95.0
    assert candle.close == 105.0
    assert candle.volume == 1000.0


def test_ohlcv_data_high_lower_than_low():
    """Test qu'on ne peut pas avoir high < low."""
    with pytest.raises(ValidationError) as exc_info:
        OHLCVData(
            timestamp=datetime(2024, 1, 1, 12, 0),
            open=100.0,            
            low=95.0,
            high=90.0,  # Plus bas que low !
            close=105.0,
            volume=1000.0,
        )
    
    assert "low" in str(exc_info.value)
    assert "must be >=" in str(exc_info.value)

def test_ohlcv_data_negative_volume():
    """Test qu'on ne peut pas avoir un volume négatif."""
    with pytest.raises(ValidationError):
        OHLCVData(
            timestamp=datetime(2024, 1, 1, 12, 0),
            open=100.0,
            low=95.0,
            high=110.0,
            close=105.0,
            volume=-100.0,  # Volume négatif !
        )


def test_market_data_valid():
    """Test création de données de marché valides."""
    candles = [
        OHLCVData(
            timestamp=datetime(2024, 1, 1, 12, 0),
            open=100.0,
            low=95.0,
            high=110.0,
            close=105.0,
            volume=1000.0,
        )
    ]
    
    market_data = MarketData(
        symbol="btc",  # En minuscules pour tester la normalisation
        timeframe=TimeFrame.ONE_HOUR,
        candles=candles,
        last_updated=datetime.now(),
    )
    
    assert market_data.symbol == "BTC"  # Doit être en majuscules
    assert market_data.timeframe == TimeFrame.ONE_HOUR
    assert len(market_data.candles) == 1


def test_market_data_empty_candles():
    """Test qu'on ne peut pas avoir une liste vide de bougies."""
    with pytest.raises(ValidationError):
        MarketData(
            symbol="BTC",
            timeframe=TimeFrame.ONE_HOUR,
            candles=[],  # Liste vide !
            last_updated=datetime.now(),
        )


def test_timeframe_enum():
    """Test que les TimeFrames sont corrects."""
    assert TimeFrame.ONE_MINUTE.value == "1m"
    assert TimeFrame.ONE_HOUR.value == "1h"
    assert TimeFrame.ONE_DAY.value == "1d"

