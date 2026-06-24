import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000,
        commission_fixed: float = 0.0,
        commission_rate: float = 0.0
    ):
        self.initial_capital = initial_capital
        self.commission_fixed = commission_fixed
        self.commission_rate = commission_rate
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        benchmark_weights: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Ejecuta backtest de la estrategia
        
        Args:
            data: DataFrame con datos de precios
            signals: Serie con señales de trading (1, -1, 0)
            benchmark_weights: Pesos del portafolio benchmark
            
        Returns:
            Dict con resultados del backtest
        """
        try:
            # Obtener precios de cierre
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data['Close']
            else:
                close_prices = data[['Close']] if 'Close' in data.columns else data
            
            # Calcular retornos diarios
            if isinstance(close_prices, pd.DataFrame):
                returns = close_prices.pct_change().fillna(0)
            else:
                returns = close_prices.pct_change().fillna(0)
            
            # Asegurar que signals tenga el mismo índice que returns
            signals = signals.reindex(returns.index, fill_value=0)
            
            # Calcular retornos de la estrategia
            if isinstance(returns, pd.DataFrame) and returns.shape[1] > 1:
                # Múltiples activos: promedio de señales
                strategy_returns = (returns * signals.values.reshape(-1, 1)).mean(axis=1)
            else:
                # Un solo activo
                if isinstance(returns, pd.DataFrame):
                    strategy_returns = returns.iloc[:, 0] * signals
                else:
                    strategy_returns = returns * signals
            
            # Calcular costos de transacción
            signal_changes = signals.diff().abs()
            signal_changes.iloc[0] = abs(signals.iloc[0])  # Primera señal
            
            # Costos fijos
            fixed_costs = signal_changes * self.commission_fixed / self.initial_capital
            
            # Costos variables (como porcentaje del valor transado)
            if isinstance(strategy_returns, pd.Series):
                variable_costs = signal_changes * self.commission_rate * abs(strategy_returns)
            else:
                variable_costs = pd.Series(0, index=returns.index)
            
            total_costs = fixed_costs + variable_costs
            
            # Aplicar costos
            strategy_returns = strategy_returns - total_costs
            
            # Calcular métricas de la estrategia
            strategy_metrics = self.calculate_metrics(strategy_returns, "Estrategia")
            
            # Calcular benchmark (Buy & Hold)
            if benchmark_weights is not None:
                if isinstance(returns, pd.DataFrame):
                    # Asegurar que los pesos tengan la forma correcta
                    weights_array = np.array(benchmark_weights).flatten()
                    if len(weights_array) == returns.shape[1]:
                        benchmark_returns = (returns * weights_array).sum(axis=1)
                    else:
                        benchmark_returns = returns.mean(axis=1)
                else:
                    benchmark_returns = returns
            else:
                # Si no hay pesos, usar promedio igual
                if isinstance(returns, pd.DataFrame):
                    benchmark_returns = returns.mean(axis=1)
                else:
                    benchmark_returns = returns
            
            benchmark_metrics = self.calculate_metrics(benchmark_returns, "Benchmark")
            
            return {
                'strategy_metrics': strategy_metrics,
                'benchmark_metrics': benchmark_metrics,
                'strategy_returns': strategy_returns,
                'benchmark_returns': benchmark_returns
            }
            
        except Exception as e:
            logger.error(f"Error en backtest: {str(e)}")
            raise
    
    def calculate_metrics(self, returns: pd.Series, name: str = "Strategy") -> Dict:
        """
        Calcula métricas de rendimiento de forma segura
        
        Args:
            returns: Serie de retornos
            name: Nombre para las métricas
            
        Returns:
            Dict con métricas calculadas
        """
        try:
            # Asegurar que returns sea una Serie
            if not isinstance(returns, pd.Series):
                returns = pd.Series(returns)
            
            # Eliminar NaN
            returns = returns.dropna()
            
            if len(returns) == 0:
                return self._empty_metrics(name)
            
            # Retorno acumulado
            cumulative_returns = (1 + returns).cumprod()
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
            
            # Métricas anualizadas
            trading_days = 252
            years = len(returns) / trading_days
            
            if years > 0 and total_return > -1:
                annual_return = (1 + total_return) ** (1 / years) - 1
            else:
                annual_return = 0
            
            annual_volatility = returns.std() * np.sqrt(trading_days) if len(returns) > 1 else 0
            
            # Sharpe Ratio
            if annual_volatility > 0:
                sharpe_ratio = annual_return / annual_volatility
            else:
                sharpe_ratio = 0
            
            # Maximum Drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            # Win Rate
            winning_days = (returns > 0).sum()
            total_days = len(returns)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            # Profit Factor
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
            
            # Calmar Ratio
            if abs(max_drawdown) > 0:
                calmar_ratio = annual_return / abs(max_drawdown)
            else:
                calmar_ratio = 0
            
            return {
                'name': name,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'calmar_ratio': calmar_ratio,
                'cumulative_returns': cumulative_returns
            }
            
        except Exception as e:
            logger.error(f"Error calculando métricas para {name}: {str(e)}")
            return self._empty_metrics(name)
    
    def _empty_metrics(self, name: str) -> Dict:
        """
        Retorna métricas vacías en caso de error
        """
        return {
            'name': name,
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'cumulative_returns': pd.Series()
        }
    
    def compare_strategies(self, results: Dict) -> pd.DataFrame:
        """
        Compara múltiples estrategias
        
        Args:
            results: Diccionario con resultados de backtest
            
        Returns:
            DataFrame con comparativa
        """
        comparison = []
        
        for strategy_name, result in results.items():
            try:
                metrics = result['strategy_metrics']
                comparison.append({
                    'Estrategia': strategy_name,
                    'Retorno Total': metrics['total_return'],
                    'Retorno Anual': metrics['annual_return'],
                    'Volatilidad Anual': metrics['annual_volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Win Rate': metrics['win_rate'],
                    'Profit Factor': metrics['profit_factor'],
                    'Calmar Ratio': metrics['calmar_ratio']
                })
            except Exception as e:
                logger.error(f"Error procesando {strategy_name}: {str(e)}")
                continue
        
        if not comparison:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison)
        
        # Formatear columnas de porcentaje
        pct_columns = ['Retorno Total', 'Retorno Anual', 'Volatilidad Anual', 
                      'Max Drawdown', 'Win Rate']
        for col in pct_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        
        # Formatear columnas numéricas
        num_columns = ['Sharpe Ratio', 'Profit Factor', 'Calmar Ratio']
        for col in num_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and x != float('inf') else "∞")
        
        return df
    
    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """
        Calcula la serie de drawdown
        
        Args:
            returns: Serie de retornos
            
        Returns:
            Serie con drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calcula Sharpe Ratio rodante
        
        Args:
            returns: Serie de retornos
            window: Ventana para cálculo rodante
            
        Returns:
            Serie con Sharpe Ratio rodante
        """
        rolling_return = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        return rolling_sharpe
    
    # Agregar estos métodos a utils/backtest_engine.py

    def run_backtest_individual(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        asset_capital: float
    ) -> Dict:
        """
        Ejecuta backtest para un activo individual con capital específico
        """
        # Obtener precios de cierre
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].iloc[:, 0]
        elif 'Close' in data.columns:
            close_prices = data['Close']
        else:
            close_prices = data.iloc[:, 0]
        
        # Calcular retornos
        returns = close_prices.pct_change().fillna(0)
        
        # Alinear señales
        signals = signals.reindex(returns.index, fill_value=0)
        
        # Calcular retornos de estrategia
        strategy_returns = returns * signals
        
        # Aplicar costos
        signal_changes = signals.diff().abs()
        signal_changes.iloc[0] = abs(signals.iloc[0])
        
        # Costos sobre el capital del activo
        fixed_costs = signal_changes * self.commission_fixed / asset_capital
        variable_costs = signal_changes * self.commission_rate * abs(strategy_returns)
        total_costs = fixed_costs + variable_costs
        
        strategy_returns = strategy_returns - total_costs
        
        # Calcular métricas
        strategy_metrics = self.calculate_metrics(strategy_returns, "Estrategia")
        benchmark_metrics = self.calculate_metrics(returns, "Buy & Hold")
        
        # Ajustar retornos acumulados al capital
        strategy_metrics['cumulative_returns'] = (1 + strategy_returns).cumprod() * asset_capital
        benchmark_metrics['cumulative_returns'] = (1 + returns).cumprod() * asset_capital
        
        return {
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'strategy_returns': strategy_returns,
            'benchmark_returns': returns
        }

    def run_backtest_buy_hold(
        self,
        data: pd.DataFrame,
        asset_capital: float
    ) -> Dict:
        """
        Simula Buy & Hold para un activo
        """
        # Obtener precios
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close'].iloc[:, 0]
        elif 'Close' in data.columns:
            close_prices = data['Close']
        else:
            close_prices = data.iloc[:, 0]
        
        returns = close_prices.pct_change().fillna(0)
        
        # Buy & Hold: siempre invertido
        strategy_returns = returns.copy()
        
        metrics = self.calculate_metrics(returns, "Buy & Hold")
        metrics['cumulative_returns'] = (1 + returns).cumprod() * asset_capital
        
        return {
            'strategy_metrics': metrics,
            'benchmark_metrics': metrics.copy(),
            'strategy_returns': strategy_returns,
            'benchmark_returns': returns
        }