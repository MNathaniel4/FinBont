
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import ta
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    TSIIndicator,
    UltimateOscillator,
    WilliamsRIndicator,
    AwesomeOscillatorIndicator,
    KAMAIndicator,
    PercentagePriceOscillator,
    PercentageVolumeOscillator,
    ROCIndicator,
    StochRSIIndicator
)
from ta.volume import (
    ChaikinMoneyFlowIndicator,
    ForceIndexIndicator,
    EaseOfMovementIndicator,
    VolumePriceTrendIndicator,
    NegativeVolumeIndexIndicator,
    AccDistIndexIndicator
)
from ta.trend import (
    MACD,
    ADXIndicator,
    VortexIndicator,
    TRIXIndicator,
    MassIndex,
    CCIIndicator,
    DPOIndicator,
    KSTIndicator,
    IchimokuIndicator,
    PSARIndicator,
    STCIndicator
)
from ta.volatility import (
    BollingerBands,
    KeltnerChannel,
    DonchianChannel,
    UlcerIndex,
    AverageTrueRange
)
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Clase base para todas las estrategias"""
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera señales de trading
        1: Compra (Long)
        -1: Venta (Short)
        0: Neutral (Sin posición)
        """
        pass
    
    def _get_price_data(self, data: pd.DataFrame):
        """Extrae datos de precios del DataFrame"""
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0]
            high = data['High'].iloc[:, 0] if 'High' in data.columns.get_level_values(0) else close
            low = data['Low'].iloc[:, 0] if 'Low' in data.columns.get_level_values(0) else close
            volume = data['Volume'].iloc[:, 0] if 'Volume' in data.columns.get_level_values(0) else None
        else:
            close = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            high = data['High'] if 'High' in data.columns else close
            low = data['Low'] if 'Low' in data.columns else close
            volume = data['Volume'] if 'Volume' in data.columns else None
        
        return close, high, low, volume


# ============================================
# ESTRATEGIAS DE MOMENTUM (TA.MOMENTUM)
# ============================================

class MACDMomentumStrategy(BaseStrategy):
    '''Estrategía con MACD'''
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f'MACD_{fast}_{slow}_{signal}')
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)

        macd = MACD(close,
                    window_slow= self.slow,
                    window_fast= self.fast,
                    window_sign= self.signal)
    
        signals = pd.Series(0, index=data.index)
        # Señal de compra: MACD cruza por encima de la señal
        signals[macd.macd() > macd.macd_signal()] = 1
        # Señal de venta: MACD cruza por debajo de la señal
        signals[macd.macd() < macd.macd_signal()] = -1
        
        return signals

class MovingAverageCrossover(BaseStrategy):
    """Estrategia de Cruce de Medias Móviles"""
    def __init__(self, fast_ma: int = 10, slow_ma: int = 30, ma_type: str = 'SMA'):
        super().__init__(f"MA_{ma_type}_{fast_ma}_{slow_ma}")
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.ma_type = ma_type
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        
        if self.ma_type == 'EMA':
            fast_ma_line = close.ewm(span=self.fast_ma, adjust=False).mean()
            slow_ma_line = close.ewm(span=self.slow_ma, adjust=False).mean()
        else:  # SMA por defecto
            fast_ma_line = close.rolling(window=self.fast_ma).mean()
            slow_ma_line = close.rolling(window=self.slow_ma).mean()
        
        signals = pd.Series(0, index=data.index)
        # Compra: MA rápida cruza por encima de MA lenta (Golden Cross)
        signals[fast_ma_line > slow_ma_line] = 1
        # Venta: MA rápida cruza por debajo de MA lenta (Death Cross)
        signals[fast_ma_line < slow_ma_line] = -1
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """Estrategia con Bandas de Bollinger"""
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        
        bb = BollingerBands(
            close,
            window=self.period,
            window_dev=self.std_dev
        )
        
        signals = pd.Series(0, index=data.index)
        
        # Compra: precio toca/cruza banda inferior (sobreventa)
        signals[close <= bb.bollinger_lband()] = 1
        
        # Venta: precio toca/cruza banda superior (sobrecompra)
        signals[close >= bb.bollinger_hband()] = -1
        
        # Media móvil: cerrar posiciones cuando vuelve a la media
        middle_band = bb.bollinger_mavg()
        
        # Suavizar señales (mantener posición hasta cruzar la media)
        for i in range(1, len(signals)):
            if signals.iloc[i] == 0:
                if signals.iloc[i-1] == 1 and close.iloc[i] >= middle_band.iloc[i]:
                    signals.iloc[i] = 0  # Cerrar largos al volver a la media
                elif signals.iloc[i-1] == -1 and close.iloc[i] <= middle_band.iloc[i]:
                    signals.iloc[i] = 0  # Cerrar cortos al volver a la media
                else:
                    signals.iloc[i] = signals.iloc[i-1]  # Mantener posición
        
        return signals


class StochasticMomentumStrategy(BaseStrategy):
    """Estrategia con Oscilador Estocástico (versión mejorada)"""
    def __init__(self, k: int = 14, d: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__(f"Stoch_{k}_{d}_{oversold}_{overbought}")
        self.k = k
        self.d = d
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        stoch = StochasticOscillator(high, low, close, window=self.k, smooth_window=self.d)
        
        stoch_k = stoch.stoch()
        stoch_d = stoch.stoch_signal()
        
        signals = pd.Series(0, index=data.index)
        
        # Compra: %K cruza por encima de %D en zona de sobreventa
        signals[(stoch_k > stoch_d) & (stoch_k < self.oversold)] = 1
        
        # Venta: %K cruza por debajo de %D en zona de sobrecompra
        signals[(stoch_k < stoch_d) & (stoch_k > self.overbought)] = -1
        
        return signals    


class RSIMomentumStrategy(BaseStrategy):
    """Estrategia basada en RSI (Relative Strength Index)"""
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(f"RSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        rsi = RSIIndicator(close, window=self.period).rsi()
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1
        return signals


class StochasticMomentumStrategy(BaseStrategy):
    """Estrategia con Oscilador Estocástico"""
    def __init__(self, k: int = 14, d: int = 3, oversold: int = 20, overbought: int = 80):
        super().__init__(f"Stoch_{k}_{d}_{oversold}_{overbought}")
        self.k = k
        self.d = d
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        stoch = StochasticOscillator(high, low, close, window=self.k, smooth_window=self.d)
        
        signals = pd.Series(0, index=data.index)
        signals[stoch.stoch() < self.oversold] = 1
        signals[stoch.stoch() > self.overbought] = -1
        return signals


class StochRSIStrategy(BaseStrategy):
    """Estrategia con Stochastic RSI"""
    def __init__(self, period: int = 14, oversold: float = 0.2, overbought: float = 0.8):
        super().__init__(f"StochRSI_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        stoch_rsi = StochRSIIndicator(close, window=self.period).stochrsi()
        
        signals = pd.Series(0, index=data.index)
        signals[stoch_rsi < self.oversold] = 1
        signals[stoch_rsi > self.overbought] = -1
        return signals


class TSIStrategy(BaseStrategy):
    """Estrategia con True Strength Index"""
    def __init__(self, window_slow: int = 25, window_fast: int = 13, signal: int = 13):
        super().__init__(f"TSI_{window_slow}_{window_fast}_{signal}")
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        tsi = TSIIndicator(close, window_slow=self.window_slow, window_fast=self.window_fast)
        tsi_line = tsi.tsi()
        signal_line = tsi.tsi().rolling(window=self.signal).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[tsi_line > signal_line] = 1
        signals[tsi_line < signal_line] = -1
        return signals


class UltimateOscillatorStrategy(BaseStrategy):
    """Estrategia con Ultimate Oscillator"""
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28, 
                 oversold: int = 30, overbought: int = 70):
        super().__init__(f"UO_{period1}_{period2}_{period3}_{oversold}_{overbought}")
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        uo = UltimateOscillator(high, low, close, 
                                window1=self.period1, 
                                window2=self.period2, 
                                window3=self.period3)
        
        signals = pd.Series(0, index=data.index)
        signals[uo.ultimate_oscillator() < self.oversold] = 1
        signals[uo.ultimate_oscillator() > self.overbought] = -1
        return signals


class WilliamsRStrategy(BaseStrategy):
    """Estrategia con Williams %R"""
    def __init__(self, period: int = 14, oversold: int = -80, overbought: int = -20):
        super().__init__(f"WilliamsR_{period}_{oversold}_{overbought}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        wr = WilliamsRIndicator(high, low, close, lbp=self.period)
        
        signals = pd.Series(0, index=data.index)
        signals[wr.williams_r() < self.oversold] = 1
        signals[wr.williams_r() > self.overbought] = -1
        return signals


class AwesomeOscillatorStrategy(BaseStrategy):
    """Estrategia con Awesome Oscillator"""
    def __init__(self, window1: int = 5, window2: int = 34):
        super().__init__(f"AO_{window1}_{window2}")
        self.window1 = window1
        self.window2 = window2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        _, high, low, _ = self._get_price_data(data)
        ao = AwesomeOscillatorIndicator(high, low, 
                                        window1=self.window1, 
                                        window2=self.window2)
        
        signals = pd.Series(0, index=data.index)
        ao_values = ao.awesome_oscillator()
        
        # Cruce por encima de cero = compra
        signals[(ao_values > 0) & (ao_values.shift(1) <= 0)] = 1
        # Cruce por debajo de cero = venta
        signals[(ao_values < 0) & (ao_values.shift(1) >= 0)] = -1
        
        # Mantener señal hasta cambio
        for i in range(1, len(signals)):
            if signals.iloc[i] == 0:
                signals.iloc[i] = signals.iloc[i-1]
        
        return signals


class KAMAStrategy(BaseStrategy):
    """Estrategia con Kaufman's Adaptive Moving Average"""
    def __init__(self, window: int = 10, pow1: int = 2, pow2: int = 30):
        super().__init__(f"KAMA_{window}_{pow1}_{pow2}")
        self.window = window
        self.pow1 = pow1
        self.pow2 = pow2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        kama = KAMAIndicator(close, window=self.window, pow1=self.pow1, pow2=self.pow2)
        
        signals = pd.Series(0, index=data.index)
        kama_line = kama.kama()
        
        # Señal cuando el precio cruza el KAMA
        signals[close > kama_line] = 1
        signals[close < kama_line] = -1
        return signals


class ROCStrategy(BaseStrategy):
    """Estrategia con Rate of Change"""
    def __init__(self, window: int = 12, buy_threshold: float = 2.0, sell_threshold: float = -2.0):
        super().__init__(f"ROC_{window}_{buy_threshold}_{sell_threshold}")
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        roc = ROCIndicator(close, window=self.window).roc()
        
        signals = pd.Series(0, index=data.index)
        signals[roc > self.buy_threshold] = 1
        signals[roc < self.sell_threshold] = -1
        return signals


class PPOStrategy(BaseStrategy):
    """Estrategia con Percentage Price Oscillator"""
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        super().__init__(f"PPO_{window_slow}_{window_fast}_{window_sign}")
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        ppo = PercentagePriceOscillator(close, 
                                       window_slow=self.window_slow,
                                       window_fast=self.window_fast,
                                       window_sign=self.window_sign)
        
        signals = pd.Series(0, index=data.index)
        signals[ppo.ppo() > ppo.ppo_signal()] = 1
        signals[ppo.ppo() < ppo.ppo_signal()] = -1
        return signals


class PVOStrategy(BaseStrategy):
    """Estrategia con Percentage Volume Oscillator"""
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
        super().__init__(f"PVO_{window_slow}_{window_fast}_{window_sign}")
        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        _, _, _, volume = self._get_price_data(data)
        if volume is None:
            return pd.Series(0, index=data.index)
        
        pvo = PercentageVolumeOscillator(volume,
                                        window_slow=self.window_slow,
                                        window_fast=self.window_fast,
                                        window_sign=self.window_sign)
        
        signals = pd.Series(0, index=data.index)
        signals[pvo.pvo() > pvo.pvo_signal()] = 1
        signals[pvo.pvo() < pvo.pvo_signal()] = -1
        return signals


# ============================================
# ESTRATEGIAS DE VOLUMEN
# ============================================

class CMFStrategy(BaseStrategy):
    """Estrategia con Chaikin Money Flow"""
    def __init__(self, window: int = 20, buy_threshold: float = 0.05, sell_threshold: float = -0.05):
        super().__init__(f"CMF_{window}_{buy_threshold}_{sell_threshold}")
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, volume = self._get_price_data(data)
        if volume is None:
            return pd.Series(0, index=data.index)
        
        cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        signals[cmf.chaikin_money_flow() > self.buy_threshold] = 1
        signals[cmf.chaikin_money_flow() < self.sell_threshold] = -1
        return signals


class ForceIndexStrategy(BaseStrategy):
    """Estrategia con Force Index"""
    def __init__(self, window: int = 13, buy_threshold: float = 0.0):
        super().__init__(f"ForceIndex_{window}_{buy_threshold}")
        self.window = window
        self.buy_threshold = buy_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, volume = self._get_price_data(data)
        if volume is None:
            return pd.Series(0, index=data.index)
        
        fi = ForceIndexIndicator(close, volume, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        fi_values = fi.force_index()
        signals[fi_values > self.buy_threshold] = 1
        signals[fi_values < -self.buy_threshold] = -1
        return signals


class EOMStrategy(BaseStrategy):
    """Estrategia con Ease of Movement"""
    def __init__(self, window: int = 14):
        super().__init__(f"EOM_{window}")
        self.window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        _, high, low, volume = self._get_price_data(data)
        if volume is None:
            return pd.Series(0, index=data.index)
        
        eom = EaseOfMovementIndicator(high, low, volume, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        eom_values = eom.ease_of_movement()
        signals[eom_values > 0] = 1
        signals[eom_values < 0] = -1
        return signals


class VPTrategy(BaseStrategy):
    """Estrategia con Volume Price Trend"""
    def __init__(self, window: int = 20):
        super().__init__(f"VPT_{window}")
        self.window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, volume = self._get_price_data(data)
        if volume is None:
            return pd.Series(0, index=data.index)
        
        vpt = VolumePriceTrendIndicator(close, volume)
        
        signals = pd.Series(0, index=data.index)
        vpt_values = vpt.volume_price_trend()
        vpt_ma = vpt_values.rolling(window=self.window).mean()
        
        signals[vpt_values > vpt_ma] = 1
        signals[vpt_values < vpt_ma] = -1
        return signals


# ============================================
# ESTRATEGIAS DE TENDENCIA ADICIONALES
# ============================================

class ADXStrategy(BaseStrategy):
    """Estrategia con ADX (Average Directional Index)"""
    def __init__(self, window: int = 14, adx_threshold: int = 25):
        super().__init__(f"ADX_{window}_{adx_threshold}")
        self.window = window
        self.adx_threshold = adx_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        adx = ADXIndicator(high, low, close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        
        # Solo tomar señales cuando ADX > umbral (tendencia fuerte)
        strong_trend = adx.adx() > self.adx_threshold
        bullish = adx.adx_pos() > adx.adx_neg()
        
        signals[strong_trend & bullish] = 1
        signals[strong_trend & ~bullish] = -1
        return signals


class CCIIndicatorStrategy(BaseStrategy):
    """Estrategia con Commodity Channel Index"""
    def __init__(self, window: int = 20, constant: float = 0.015,
                 oversold: float = -100, overbought: float = 100):
        super().__init__(f"CCI_{window}_{constant}_{oversold}_{overbought}")
        self.window = window
        self.constant = constant
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        cci = CCIIndicator(high, low, close, window=self.window, constant=self.constant)
        
        signals = pd.Series(0, index=data.index)
        signals[cci.cci() < self.oversold] = 1
        signals[cci.cci() > self.overbought] = -1
        return signals


class TRIXStrategy(BaseStrategy):
    """Estrategia con TRIX (Triple Exponential Average)"""
    def __init__(self, window: int = 15, signal: int = 9):
        super().__init__(f"TRIX_{window}_{signal}")
        self.window = window
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        trix = TRIXIndicator(close, window=self.window)
        
        trix_line = trix.trix()
        signal_line = trix_line.rolling(window=self.signal).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[trix_line > signal_line] = 1
        signals[trix_line < signal_line] = -1
        return signals


class VortexStrategy(BaseStrategy):
    """Estrategia con Vortex Indicator"""
    def __init__(self, window: int = 14):
        super().__init__(f"Vortex_{window}")
        self.window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        vortex = VortexIndicator(high, low, close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        signals[vortex.vortex_indicator_pos() > vortex.vortex_indicator_neg()] = 1
        signals[vortex.vortex_indicator_pos() < vortex.vortex_indicator_neg()] = -1
        return signals


class DPOStrategy(BaseStrategy):
    """Estrategia con Detrended Price Oscillator"""
    def __init__(self, window: int = 20):
        super().__init__(f"DPO_{window}")
        self.window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        dpo = DPOIndicator(close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        dpo_values = dpo.dpo()
        signals[dpo_values > 0] = 1
        signals[dpo_values < 0] = -1
        return signals


class KSTStrategy(BaseStrategy):
    """Estrategia con Know Sure Thing (KST)"""
    def __init__(self, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
                 window1: int = 10, window2: int = 10, window3: int = 10, window4: int = 15,
                 signal: int = 9):
        super().__init__(f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{signal}")
        self.roc1, self.roc2, self.roc3, self.roc4 = roc1, roc2, roc3, roc4
        self.window1, self.window2, self.window3, self.window4 = window1, window2, window3, window4
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        kst = KSTIndicator(close,
                          roc1=self.roc1, roc2=self.roc2, roc3=self.roc3, roc4=self.roc4,
                          window1=self.window1, window2=self.window2, 
                          window3=self.window3, window4=self.window4)
        
        kst_line = kst.kst()
        signal_line = kst.kst_sig()
        
        signals = pd.Series(0, index=data.index)
        signals[kst_line > signal_line] = 1
        signals[kst_line < signal_line] = -1
        return signals


class STCStrategy(BaseStrategy):
    """Estrategia con Schaff Trend Cycle"""
    def __init__(self, window_fast: int = 23, window_slow: int = 50,
                 cycle: int = 10, smooth1: int = 3, smooth2: int = 3,
                 oversold: int = 25, overbought: int = 75):
        super().__init__(f"STC_{window_fast}_{window_slow}_{cycle}_{oversold}_{overbought}")
        self.window_fast = window_fast
        self.window_slow = window_slow
        self.cycle = cycle
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        stc = STCIndicator(close,
                          window_fast=self.window_fast,
                          window_slow=self.window_slow,
                          cycle=self.cycle,
                          smooth1=self.smooth1,
                          smooth2=self.smooth2)
        
        signals = pd.Series(0, index=data.index)
        stc_values = stc.stc()
        signals[stc_values < self.oversold] = 1
        signals[stc_values > self.overbought] = -1
        return signals


# ============================================
# ESTRATEGIAS DE VOLATILIDAD
# ============================================

class KeltnerChannelStrategy(BaseStrategy):
    """Estrategia con Keltner Channels"""
    def __init__(self, window: int = 20, window_atr: int = 10, multiplier: float = 2.0):
        super().__init__(f"Keltner_{window}_{window_atr}_{multiplier}")
        self.window = window
        self.window_atr = window_atr
        self.multiplier = multiplier
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        kc = KeltnerChannel(high, low, close,
                           window=self.window,
                           window_atr=self.window_atr,
                           multiplier=self.multiplier)
        
        signals = pd.Series(0, index=data.index)
        signals[close < kc.keltner_channel_lband()] = 1  # Compra en banda inferior
        signals[close > kc.keltner_channel_hband()] = -1  # Venta en banda superior
        return signals


class DonchianChannelStrategy(BaseStrategy):
    """Estrategia con Donchian Channels (Turtle Trading)"""
    def __init__(self, window: int = 20):
        super().__init__(f"Donchian_{window}")
        self.window = window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        dc = DonchianChannel(high, low, close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        # Compra en nuevo máximo del período
        signals[close >= dc.donchian_channel_hband()] = 1
        # Venta en nuevo mínimo del período
        signals[close <= dc.donchian_channel_lband()] = -1
        return signals


class UlcerIndexStrategy(BaseStrategy):
    """Estrategia con Ulcer Index (para identificar drawdowns)"""
    def __init__(self, window: int = 14, threshold: float = 5.0):
        super().__init__(f"Ulcer_{window}_{threshold}")
        self.window = window
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, _, _, _ = self._get_price_data(data)
        ui = UlcerIndex(close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        ulcer_values = ui.ulcer_index()
        
        # Alto Ulcer Index = alto drawdown = oportunidad de compra
        signals[ulcer_values > self.threshold] = 1
        signals[ulcer_values < self.threshold / 2] = -1
        return signals


class ATRStrategy(BaseStrategy):
    """Estrategia con Average True Range para volatilidad"""
    def __init__(self, window: int = 14, multiplier: float = 1.5):
        super().__init__(f"ATR_{window}_{multiplier}")
        self.window = window
        self.multiplier = multiplier
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close, high, low, _ = self._get_price_data(data)
        atr = AverageTrueRange(high, low, close, window=self.window)
        
        signals = pd.Series(0, index=data.index)
        atr_values = atr.average_true_range()
        atr_ma = atr_values.rolling(window=self.window * 2).mean()
        
        # Alta volatilidad = señal de compra (breakout)
        signals[atr_values > atr_ma * self.multiplier] = 1
        # Baja volatilidad = señal de venta (consolidación)
        signals[atr_values < atr_ma / self.multiplier] = -1
        return signals


# ============================================
# ESTRATEGIAS COMBINADAS AVANZADAS
# ============================================

class MACDBBStrategy(BaseStrategy):
    """MACD + Bollinger Bands combinados"""
    def __init__(self, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 bb_period: int = 20, bb_std: float = 2.0):
        super().__init__(f"MACDBB_{macd_fast}_{macd_slow}_{macd_signal}_{bb_period}_{bb_std}")
        self.macd = MACDMomentumStrategy(macd_fast, macd_slow, macd_signal)
        self.bb = BollingerBandsStrategy(bb_period, bb_std)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        macd_signals = self.macd.generate_signals(data)
        bb_signals = self.bb.generate_signals(data)
        
        # Combinar señales (ambas deben coincidir)
        combined = pd.Series(0, index=data.index)
        combined[(macd_signals == 1) & (bb_signals == 1)] = 1
        combined[(macd_signals == -1) & (bb_signals == -1)] = -1
        return combined


class RSIMACDStrategy(BaseStrategy):
    """RSI + MACD combinados con filtro de volumen"""
    def __init__(self, rsi_period: int = 14, rsi_oversold: int = 30, rsi_overbought: int = 70,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 volume_threshold: float = 1.2):
        super().__init__(f"RSIMACD_{rsi_period}_{macd_fast}_{macd_slow}_{volume_threshold}")
        self.rsi = RSIMomentumStrategy(rsi_period, rsi_oversold, rsi_overbought)
        self.macd = MACDMomentumStrategy(macd_fast, macd_slow, macd_signal)
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi_signals = self.rsi.generate_signals(data)
        macd_signals = self.macd.generate_signals(data)
        
        # Filtro de volumen
        _, _, _, volume = self._get_price_data(data)
        if volume is not None:
            volume_ma = volume.rolling(window=20).mean()
            high_volume = volume > volume_ma * self.volume_threshold
        else:
            high_volume = pd.Series(True, index=data.index)
        
        combined = pd.Series(0, index=data.index)
        combined[(rsi_signals == 1) & (macd_signals == 1) & high_volume] = 1
        combined[(rsi_signals == -1) & (macd_signals == -1) & high_volume] = -1
        return combined


# Clase CombinedStrategy original (mantener por compatibilidad)
class CombinedStrategy(BaseStrategy):
    """Combina múltiples estrategias con lógica booleana"""
    def __init__(self, strategies: list, combination_type: str = 'AND'):
        filtered_strategies = [s for s in strategies if not isinstance(s, CombinedStrategy)]
        
        if not filtered_strategies:
            raise ValueError("Se necesita al menos una estrategia base")
        
        name = f"Combined_{combination_type}_" + "_".join([s.name for s in filtered_strategies])
        super().__init__(name)
        self.strategies = filtered_strategies
        self.combination_type = combination_type
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.generate_signals(data)
            all_signals.append(signals)
        
        if not all_signals:
            return pd.Series(0, index=data.index)
        
        signals_df = pd.concat(all_signals, axis=1)
        
        if self.combination_type == 'AND':
            combined = pd.Series(0, index=data.index)
            combined[(signals_df == 1).all(axis=1)] = 1
            combined[(signals_df == -1).all(axis=1)] = -1
            return combined
        elif self.combination_type == 'OR':
            combined = pd.Series(0, index=data.index)
            combined[(signals_df == 1).any(axis=1)] = 1
            combined[(signals_df == -1).any(axis=1)] = -1
            return combined
        elif self.combination_type == 'MAJORITY':
            buy_votes = (signals_df == 1).sum(axis=1)
            sell_votes = (signals_df == -1).sum(axis=1)
            
            combined = pd.Series(0, index=data.index)
            combined[buy_votes > sell_votes] = 1
            combined[sell_votes > buy_votes] = -1
            return combined