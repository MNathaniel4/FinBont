# strategies/my_custom_strategies.py
import pandas as pd
import numpy as np
from strategies.momentum_strategies import BaseStrategy
from strategies.strategy_registry import strategy_registry

class ATRBreakoutStrategy(BaseStrategy):
    """
    Estrategia de breakout basada en ATR (Average True Range)
    Compra cuando el precio rompe el máximo de N días + factor * ATR
    Vende cuando rompe el mínimo de N días - factor * ATR
    """
    def __init__(self, period: int = 20, factor: float = 2.0):
        super().__init__(f"ATR_Breakout_{period}_{factor}")
        self.period = period
        self.factor = factor
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Obtener precios
        if isinstance(data.columns, pd.MultiIndex):
            high = data['High'].iloc[:, 0]
            low = data['Low'].iloc[:, 0]
            close = data['Close'].iloc[:, 0]
        else:
            high = data['High']
            low = data['Low']
            close = data['Close']
        
        # Calcular ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        # Calcular niveles de breakout
        highest = high.rolling(window=self.period).max()
        lowest = low.rolling(window=self.period).min()
        
        # Generar señales
        signals = pd.Series(0, index=data.index)
        
        # Compra: precio cierra por encima del máximo + factor*ATR
        buy_signal = close > (highest + self.factor * atr)
        signals[buy_signal] = 1
        
        # Venta: precio cierra por debajo del mínimo - factor*ATR
        sell_signal = close < (lowest - self.factor * atr)
        signals[sell_signal] = -1
        
        return signals


class VolumeWeightedMAStrategy(BaseStrategy):
    """
    Estrategia de Media Móvil Ponderada por Volumen
    """
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__(f"VWMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0]
            volume = data['Volume'].iloc[:, 0]
        else:
            close = data['Close']
            volume = data['Volume']
        
        # Calcular VWMA
        def vwma(price, volume, period):
            pv = price * volume
            return pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        fast_vwma = vwma(close, volume, self.fast_period)
        slow_vwma = vwma(close, volume, self.slow_period)
        
        signals = pd.Series(0, index=data.index)
        signals[fast_vwma > slow_vwma] = 1
        signals[fast_vwma < slow_vwma] = -1
        
        return signals


# Registrar las nuevas estrategias
strategy_registry.register_strategy(
    name="ATR Breakout",
    strategy_class=ATRBreakoutStrategy,
    category="Volatilidad",
    description="Estrategia de breakout usando Average True Range",
    params={
        "period": {
            "type": "int",
            "default": 20,
            "min": 10,
            "max": 50,
            "description": "Período ATR"
        },
        "factor": {
            "type": "float",
            "default": 2.0,
            "min": 1.0,
            "max": 5.0,
            "step": 0.5,
            "description": "Factor multiplicador del ATR"
        }
    }
)

strategy_registry.register_strategy(
    name="VWMA Crossover",
    strategy_class=VolumeWeightedMAStrategy,
    category="Tendencia",
    description="Cruce de Medias Móviles Ponderadas por Volumen",
    params={
        "fast_period": {
            "type": "int",
            "default": 10,
            "min": 5,
            "max": 30,
            "description": "Período VWMA Rápida"
        },
        "slow_period": {
            "type": "int",
            "default": 30,
            "min": 15,
            "max": 60,
            "description": "Período VWMA Lenta"
        }
    }
)