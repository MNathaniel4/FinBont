# strategies/strategy_registry.py
from typing import Dict, Type, Any, List
from strategies.momentum_strategies import (
    BaseStrategy,
    RSIMomentumStrategy,
    MACDMomentumStrategy,
    StochasticMomentumStrategy,
    MovingAverageCrossover,
    BollingerBandsStrategy,
    StochRSIStrategy,
    TSIStrategy,
    UltimateOscillatorStrategy,
    WilliamsRStrategy,
    AwesomeOscillatorStrategy,
    KAMAStrategy,
    ROCStrategy,
    PPOStrategy,
    PVOStrategy,
    CMFStrategy,
    ForceIndexStrategy,
    EOMStrategy,
    VPTrategy,
    ADXStrategy,
    CCIIndicatorStrategy,
    TRIXStrategy,
    VortexStrategy,
    DPOStrategy,
    KSTStrategy,
    STCStrategy,
    KeltnerChannelStrategy,
    DonchianChannelStrategy,
    UlcerIndexStrategy,
    ATRStrategy,
    MACDBBStrategy,
    RSIMACDStrategy,
    CombinedStrategy
)

class StrategyRegistry:
    """Registro central de estrategias disponibles"""
    
    def __init__(self):
        self._strategies: Dict[str, Dict[str, Any]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Registra todas las estrategias"""
        
        # ============ MOMENTUM ============
        self.register_strategy(
            name="RSI", strategy_class=RSIMomentumStrategy,
            category="Momentum",
            description="Relative Strength Index - Identifica sobrecompra/sobreventa",
            params={
                "period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Período"},
                "oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "description": "Sobreventa"},
                "overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="Estocástico", strategy_class=StochasticMomentumStrategy,
            category="Momentum",
            description="Oscilador Estocástico",
            params={
                "k": {"type": "int", "default": 14, "min": 5, "max": 30, "description": "%K"},
                "d": {"type": "int", "default": 3, "min": 2, "max": 10, "description": "%D"},
                "oversold": {"type": "int", "default": 20, "min": 10, "max": 30, "description": "Sobreventa"},
                "overbought": {"type": "int", "default": 80, "min": 70, "max": 90, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="StochRSI", strategy_class=StochRSIStrategy,
            category="Momentum",
            description="Stochastic RSI - Mayor sensibilidad",
            params={
                "period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Período"},
                "oversold": {"type": "float", "default": 0.2, "min": 0.1, "max": 0.3, "step": 0.05, "description": "Sobreventa"},
                "overbought": {"type": "float", "default": 0.8, "min": 0.7, "max": 0.9, "step": 0.05, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="TSI", strategy_class=TSIStrategy,
            category="Momentum",
            description="True Strength Index - Doble suavizado",
            params={
                "window_slow": {"type": "int", "default": 25, "min": 10, "max": 50, "description": "Ventana Lenta"},
                "window_fast": {"type": "int", "default": 13, "min": 5, "max": 30, "description": "Ventana Rápida"},
                "signal": {"type": "int", "default": 13, "min": 5, "max": 30, "description": "Señal"}
            }
        )
        
        self.register_strategy(
            name="Ultimate Oscillator", strategy_class=UltimateOscillatorStrategy,
            category="Momentum",
            description="Ultimate Oscillator - 3 períodos combinados",
            params={
                "period1": {"type": "int", "default": 7, "min": 3, "max": 15, "description": "Período Corto"},
                "period2": {"type": "int", "default": 14, "min": 10, "max": 25, "description": "Período Medio"},
                "period3": {"type": "int", "default": 28, "min": 20, "max": 50, "description": "Período Largo"},
                "oversold": {"type": "int", "default": 30, "min": 20, "max": 40, "description": "Sobreventa"},
                "overbought": {"type": "int", "default": 70, "min": 60, "max": 80, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="Williams %R", strategy_class=WilliamsRStrategy,
            category="Momentum",
            description="Williams %R - Sobrecompra/sobreventa (0 a -100)",
            params={
                "period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Período"},
                "oversold": {"type": "int", "default": -80, "min": -100, "max": -60, "description": "Sobreventa"},
                "overbought": {"type": "int", "default": -20, "min": -40, "max": 0, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="Awesome Oscillator", strategy_class=AwesomeOscillatorStrategy,
            category="Momentum",
            description="Awesome Oscillator - Momento del mercado",
            params={
                "window1": {"type": "int", "default": 5, "min": 3, "max": 10, "description": "Ventana Rápida"},
                "window2": {"type": "int", "default": 34, "min": 20, "max": 50, "description": "Ventana Lenta"}
            }
        )
        
        self.register_strategy(
            name="KAMA", strategy_class=KAMAStrategy,
            category="Momentum",
            description="Kaufman Adaptive Moving Average",
            params={
                "window": {"type": "int", "default": 10, "min": 5, "max": 30, "description": "Ventana"},
                "pow1": {"type": "int", "default": 2, "min": 1, "max": 5, "description": "Constante Rápida"},
                "pow2": {"type": "int", "default": 30, "min": 20, "max": 50, "description": "Constante Lenta"}
            }
        )
        
        self.register_strategy(
            name="ROC", strategy_class=ROCStrategy,
            category="Momentum",
            description="Rate of Change - Cambio porcentual del precio",
            params={
                "window": {"type": "int", "default": 12, "min": 5, "max": 50, "description": "Ventana"},
                "buy_threshold": {"type": "float", "default": 2.0, "min": 0.5, "max": 10.0, "step": 0.5, "description": "Umbral Compra (%)"},
                "sell_threshold": {"type": "float", "default": -2.0, "min": -10.0, "max": -0.5, "step": 0.5, "description": "Umbral Venta (%)"}
            }
        )
        
        self.register_strategy(
            name="PPO", strategy_class=PPOStrategy,
            category="Momentum",
            description="Percentage Price Oscillator - MACD en porcentaje",
            params={
                "window_slow": {"type": "int", "default": 26, "min": 15, "max": 50, "description": "Ventana Lenta"},
                "window_fast": {"type": "int", "default": 12, "min": 5, "max": 30, "description": "Ventana Rápida"},
                "window_sign": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "Señal"}
            }
        )
        
        self.register_strategy(
            name="PVO", strategy_class=PVOStrategy,
            category="Momentum",
            description="Percentage Volume Oscillator - PPO del volumen",
            params={
                "window_slow": {"type": "int", "default": 26, "min": 15, "max": 50, "description": "Ventana Lenta"},
                "window_fast": {"type": "int", "default": 12, "min": 5, "max": 30, "description": "Ventana Rápida"},
                "window_sign": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "Señal"}
            }
        )
        
        # ============ TENDENCIA ============
        self.register_strategy(
            name="MACD", strategy_class=MACDMomentumStrategy,
            category="Tendencia",
            description="Moving Average Convergence Divergence",
            params={
                "fast": {"type": "int", "default": 12, "min": 5, "max": 30, "description": "Rápida"},
                "slow": {"type": "int", "default": 26, "min": 15, "max": 50, "description": "Lenta"},
                "signal": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "Señal"}
            }
        )
        
        self.register_strategy(
            name="Medias Móviles", strategy_class=MovingAverageCrossover,
            category="Tendencia",
            description="Cruce de Medias Móviles",
            params={
                "fast_ma": {"type": "int", "default": 10, "min": 5, "max": 50, "description": "MA Rápida"},
                "slow_ma": {"type": "int", "default": 30, "min": 15, "max": 100, "description": "MA Lenta"}
            }
        )
        
        self.register_strategy(
            name="ADX", strategy_class=ADXStrategy,
            category="Tendencia",
            description="Average Directional Index - Fuerza de tendencia",
            params={
                "window": {"type": "int", "default": 14, "min": 7, "max": 30, "description": "Ventana"},
                "adx_threshold": {"type": "int", "default": 25, "min": 15, "max": 50, "description": "Umbral ADX"}
            }
        )
        
        self.register_strategy(
            name="Vortex", strategy_class=VortexStrategy,
            category="Tendencia",
            description="Vortex Indicator - Dirección de tendencias",
            params={
                "window": {"type": "int", "default": 14, "min": 7, "max": 30, "description": "Ventana"}
            }
        )
        
        self.register_strategy(
            name="TRIX", strategy_class=TRIXStrategy,
            category="Tendencia",
            description="Triple Exponential Average",
            params={
                "window": {"type": "int", "default": 15, "min": 7, "max": 30, "description": "Ventana"},
                "signal": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "Señal"}
            }
        )
        
        self.register_strategy(
            name="CCI", strategy_class=CCIIndicatorStrategy,
            category="Tendencia",
            description="Commodity Channel Index - Ciclos de precio",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Ventana"},
                "constant": {"type": "float", "default": 0.015, "min": 0.01, "max": 0.03, "step": 0.005, "description": "Constante"},
                "oversold": {"type": "float", "default": -100, "min": -200, "max": -50, "description": "Sobreventa"},
                "overbought": {"type": "float", "default": 100, "min": 50, "max": 200, "description": "Sobrecompra"}
            }
        )
        
        self.register_strategy(
            name="DPO", strategy_class=DPOStrategy,
            category="Tendencia",
            description="Detrended Price Oscillator - Ciclos sin tendencia",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Ventana"}
            }
        )
        
        self.register_strategy(
            name="KST", strategy_class=KSTStrategy,
            category="Tendencia",
            description="Know Sure Thing - 4 tasas de cambio",
            params={
                "roc1": {"type": "int", "default": 10, "min": 5, "max": 20, "description": "ROC1"},
                "roc2": {"type": "int", "default": 15, "min": 10, "max": 25, "description": "ROC2"},
                "roc3": {"type": "int", "default": 20, "min": 15, "max": 30, "description": "ROC3"},
                "roc4": {"type": "int", "default": 30, "min": 20, "max": 40, "description": "ROC4"},
                "window1": {"type": "int", "default": 10, "min": 5, "max": 20, "description": "W1"},
                "window2": {"type": "int", "default": 10, "min": 5, "max": 20, "description": "W2"},
                "window3": {"type": "int", "default": 10, "min": 5, "max": 20, "description": "W3"},
                "window4": {"type": "int", "default": 15, "min": 5, "max": 25, "description": "W4"},
                "signal": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "Señal"}
            }
        )
        
        self.register_strategy(
            name="STC", strategy_class=STCStrategy,
            category="Tendencia",
            description="Schaff Trend Cycle - MACD + Estocástico",
            params={
                "window_fast": {"type": "int", "default": 23, "min": 10, "max": 50, "description": "Ventana Rápida"},
                "window_slow": {"type": "int", "default": 50, "min": 30, "max": 100, "description": "Ventana Lenta"},
                "cycle": {"type": "int", "default": 10, "min": 5, "max": 20, "description": "Ciclo"},
                "smooth1": {"type": "int", "default": 3, "min": 2, "max": 10, "description": "Suavizado 1"},
                "smooth2": {"type": "int", "default": 3, "min": 2, "max": 10, "description": "Suavizado 2"},
                "oversold": {"type": "int", "default": 25, "min": 10, "max": 40, "description": "Sobreventa"},
                "overbought": {"type": "int", "default": 75, "min": 60, "max": 90, "description": "Sobrecompra"}
            }
        )
        
        # ============ VOLUMEN ============
        self.register_strategy(
            name="Chaikin Money Flow", strategy_class=CMFStrategy,
            category="Volumen",
            description="Chaikin Money Flow - Presión compra/venta",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Ventana"},
                "buy_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01, "description": "Umbral Compra"},
                "sell_threshold": {"type": "float", "default": -0.05, "min": -0.2, "max": -0.01, "step": 0.01, "description": "Umbral Venta"}
            }
        )
        
        self.register_strategy(
            name="Force Index", strategy_class=ForceIndexStrategy,
            category="Volumen",
            description="Force Index - Precio y volumen combinados",
            params={
                "window": {"type": "int", "default": 13, "min": 5, "max": 50, "description": "Ventana"},
                "buy_threshold": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1, "description": "Umbral"}
            }
        )
        
        self.register_strategy(
            name="Ease of Movement", strategy_class=EOMStrategy,
            category="Volumen",
            description="Ease of Movement - Cambio precio vs volumen",
            params={
                "window": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Ventana"}
            }
        )
        
        self.register_strategy(
            name="Volume Price Trend", strategy_class=VPTrategy,
            category="Volumen",
            description="Volume Price Trend - Volumen acumulado",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Ventana MA"}
            }
        )
        
        # ============ VOLATILIDAD ============
        self.register_strategy(
            name="Bandas Bollinger", strategy_class=BollingerBandsStrategy,
            category="Volatilidad",
            description="Bandas de Bollinger - Reversión a la media",
            params={
                "period": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Período"},
                "std_dev": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.1, "description": "Desviaciones"}
            }
        )
        
        self.register_strategy(
            name="Keltner Channels", strategy_class=KeltnerChannelStrategy,
            category="Volatilidad",
            description="Keltner Channels - Bandas basadas en ATR",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "Ventana EMA"},
                "window_atr": {"type": "int", "default": 10, "min": 5, "max": 30, "description": "Ventana ATR"},
                "multiplier": {"type": "float", "default": 2.0, "min": 1.0, "max": 4.0, "step": 0.5, "description": "Multiplicador"}
            }
        )
        
        self.register_strategy(
            name="Donchian Channels", strategy_class=DonchianChannelStrategy,
            category="Volatilidad",
            description="Donchian Channels - Turtle Trading",
            params={
                "window": {"type": "int", "default": 20, "min": 10, "max": 100, "description": "Ventana"}
            }
        )
        
        self.register_strategy(
            name="Ulcer Index", strategy_class=UlcerIndexStrategy,
            category="Volatilidad",
            description="Ulcer Index - Riesgo de drawdown",
            params={
                "window": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Ventana"},
                "threshold": {"type": "float", "default": 5.0, "min": 1.0, "max": 20.0, "step": 1.0, "description": "Umbral"}
            }
        )
        
        self.register_strategy(
            name="ATR Volatility", strategy_class=ATRStrategy,
            category="Volatilidad",
            description="ATR - Expansión de volatilidad",
            params={
                "window": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "Ventana ATR"},
                "multiplier": {"type": "float", "default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "description": "Multiplicador"}
            }
        )
        
        # ============ COMBINADAS ============
        self.register_strategy(
            name="MACD + Bollinger", strategy_class=MACDBBStrategy,
            category="Combinadas",
            description="MACD + Bollinger Bands - Tendencia + Volatilidad",
            params={
                "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 30, "description": "MACD Rápido"},
                "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50, "description": "MACD Lento"},
                "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "MACD Señal"},
                "bb_period": {"type": "int", "default": 20, "min": 10, "max": 50, "description": "BB Período"},
                "bb_std": {"type": "float", "default": 2.0, "min": 1.0, "max": 3.0, "step": 0.5, "description": "BB Desv."}
            }
        )
        
        self.register_strategy(
            name="RSI + MACD + Volumen", strategy_class=RSIMACDStrategy,
            category="Combinadas",
            description="RSI + MACD con filtro de volumen",
            params={
                "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 50, "description": "RSI Período"},
                "rsi_oversold": {"type": "int", "default": 30, "min": 10, "max": 40, "description": "RSI Sobreventa"},
                "rsi_overbought": {"type": "int", "default": 70, "min": 60, "max": 90, "description": "RSI Sobrecompra"},
                "macd_fast": {"type": "int", "default": 12, "min": 5, "max": 30, "description": "MACD Rápido"},
                "macd_slow": {"type": "int", "default": 26, "min": 15, "max": 50, "description": "MACD Lento"},
                "macd_signal": {"type": "int", "default": 9, "min": 5, "max": 20, "description": "MACD Señal"},
                "volume_threshold": {"type": "float", "default": 1.2, "min": 1.0, "max": 3.0, "step": 0.1, "description": "Factor Volumen"}
            }
        )
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy],
                         category: str, description: str, params: Dict[str, Dict[str, Any]]):
        """Registra una nueva estrategia"""
        self._strategies[name] = {
            'class': strategy_class,
            'category': category,
            'description': description,
            'params': params
        }
    
    def get_strategy(self, name: str, **params) -> BaseStrategy:
        """Crea una instancia de la estrategia"""
        if name not in self._strategies:
            raise ValueError(f"Estrategia '{name}' no encontrada")
        strategy_class = self._strategies[name]['class']
        return strategy_class(**params)
    
    def get_all_strategies(self) -> Dict:
        """Retorna todas las estrategias"""
        return self._strategies.copy()
    
    def get_strategies_by_category(self) -> Dict[str, List[str]]:
        """Agrupa por categoría"""
        categories = {}
        for name, info in self._strategies.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories
    
    def get_strategy_params(self, name: str) -> Dict:
        """Obtiene parámetros de una estrategia"""
        if name not in self._strategies:
            raise ValueError(f"Estrategia '{name}' no encontrada")
        return self._strategies[name]['params']


# Instancia global
strategy_registry = StrategyRegistry()