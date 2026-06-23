import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple

class RiskMetrics:
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Value at Risk (VaR) paramétrico
        """
        mu = returns.mean()
        sigma = returns.std()
        var = stats.norm.ppf(1 - confidence_level, mu, sigma)
        return abs(var)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR)
        """
        var = RiskMetrics.calculate_var(returns, confidence_level)
        cvar = returns[returns <= -var].mean()
        return abs(cvar) if not np.isnan(cvar) else abs(var)
    
    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, 
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Sharpe Ratio tradicional
        """
        excess_returns = returns - risk_free_rate/252
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, 
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Sortino Ratio (solo considera volatilidad negativa)
        """
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_var_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95
    ) -> float:
        """
        Ratio Sharpe-VaR
        """
        excess_returns = returns.mean() - risk_free_rate/252
        var = RiskMetrics.calculate_var(returns, confidence_level)
        
        if var == 0:
            return 0
        return excess_returns / var
    
    @staticmethod
    def calculate_staar_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        confidence_level: float = 0.95
    ) -> float:
        """
        STARR Ratio (Stable Tail Adjusted Return Ratio)
        """
        excess_returns = returns.mean() - risk_free_rate/252
        cvar = RiskMetrics.calculate_cvar(returns, confidence_level)
        
        if cvar == 0:
            return 0
        return excess_returns / cvar
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Máximo drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())