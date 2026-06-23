import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Callable,Tuple
import cvxpy as cp

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
    def portfolio_performance(self, weights: np.array) -> Tuple[float, float]:
        """
        Calcula retorno y riesgo del portafolio
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return portfolio_return, portfolio_std
    
    def minimize_risk(self, target_return: float = None) -> Dict:
        """
        Minimiza el riesgo para un retorno objetivo o el mínimo riesgo posible
        """
        def objective(weights):
            return np.sqrt(weights.T @ self.cov_matrix @ weights)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda w: np.sum(self.mean_returns * w) - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective, 
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': np.sum(self.mean_returns * result.x),
            'risk': objective(result.x)
        }
    
    def maximize_return(self, max_risk: float = None) -> Dict:
        """
        Maximiza el retorno para un nivel de riesgo dado
        """
        def objective(weights):
            return -np.sum(self.mean_returns * weights)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if max_risk is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: max_risk - np.sqrt(w.T @ self.cov_matrix @ w)
            })
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': -result.fun,
            'risk': np.sqrt(result.x.T @ self.cov_matrix @ result.x)
        }
    
    def maximize_sharpe_ratio(self) -> Dict:
        """
        Maximiza el Sharpe Ratio
        """
        def neg_sharpe_ratio(weights):
            p_return, p_std = self.portfolio_performance(weights)
            if p_std == 0:
                return 0
            return -(p_return - self.risk_free_rate) / p_std
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        p_return, p_std = self.portfolio_performance(result.x)
        
        return {
            'weights': result.x,
            'return': p_return,
            'risk': p_std,
            'sharpe_ratio': (p_return - self.risk_free_rate) / p_std
        }
    
    def maximize_var_sharpe_ratio(self, confidence_level: float = 0.95) -> Dict:
        """
        Maximiza el Ratio Sharpe-VaR usando optimización con CVXPY
        """
        # Esta es una versión simplificada. En producción usarías CVXPY
        # para mejor precisión con restricciones no lineales
        
        # Por ahora, usamos un grid search simple para demostración
        from itertools import product
        
        best_ratio = -np.inf
        best_weights = None
        
        # Grid search simple (en producción usar optimización más sofisticada)
        grid_points = 20
        ranges = [np.linspace(0, 1, grid_points) for _ in range(self.n_assets - 1)]
        
        for combo in product(*ranges):
            if sum(combo) > 1:
                continue
            
            weights = list(combo) + [1 - sum(combo)]
            weights = np.array(weights)
            
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            from utils.risk_metrics import RiskMetrics
            var = RiskMetrics.calculate_var(portfolio_returns, confidence_level)
            
            if var > 0:
                ratio = (portfolio_returns.mean() * 252 - self.risk_free_rate) / var
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_weights = weights
        
        if best_weights is None:
            best_weights = np.array([1/self.n_assets] * self.n_assets)
        
        p_return, p_std = self.portfolio_performance(best_weights)
        
        return {
            'weights': best_weights,
            'return': p_return,
            'risk': p_std,
            'var_sharpe_ratio': best_ratio
        }
    
    def maximize_staar_ratio(self, confidence_level: float = 0.95) -> Dict:
        """
        Maximiza el STARR Ratio
        """
        # Implementación simplificada con grid search
        from itertools import product
        
        best_ratio = -np.inf
        best_weights = None
        
        grid_points = 15
        ranges = [np.linspace(0, 1, grid_points) for _ in range(self.n_assets - 1)]
        
        for combo in product(*ranges):
            if sum(combo) > 1:
                continue
            
            weights = list(combo) + [1 - sum(combo)]
            weights = np.array(weights)
            
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            from utils.risk_metrics import RiskMetrics
            cvar = RiskMetrics.calculate_cvar(portfolio_returns, confidence_level)
            
            if cvar > 0:
                ratio = (portfolio_returns.mean() * 252 - self.risk_free_rate) / cvar
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_weights = weights
        
        if best_weights is None:
            best_weights = np.array([1/self.n_assets] * self.n_assets)
        
        p_return, p_std = self.portfolio_performance(best_weights)
        
        return {
            'weights': best_weights,
            'return': p_return,
            'risk': p_std,
            'staar_ratio': best_ratio
        }
    
# Agregar estos métodos al PortfolioOptimizer existente

    def generate_efficient_frontier(
        self, 
        n_points: int = 50, 
        risk_metric: str = 'std'
    ) -> pd.DataFrame:
        """
        Genera la frontera eficiente con diferentes métricas de riesgo
        
        Args:
            n_points: Número de puntos en la frontera
            risk_metric: 'std', 'var', 'cvar'
            
        Returns:
            DataFrame con puntos de la frontera
        """
        from utils.risk_metrics import RiskMetrics
        
        # Encontrar portafolios extremos
        min_risk_port = self.minimize_risk()
        max_return_port = self.maximize_return()
        
        # Generar puntos en la frontera
        target_returns = np.linspace(
            min_risk_port['return'], 
            max_return_port['return'], 
            n_points
        )
        
        frontier_points = []
        
        for target_return in target_returns:
            try:
                # Optimizar para el retorno objetivo
                portfolio = self.minimize_risk(target_return)
                weights = portfolio['weights']
                
                # Calcular retorno del portafolio
                portfolio_returns = (self.returns * weights).sum(axis=1)
                
                # Calcular riesgo según la métrica seleccionada
                if risk_metric == 'std':
                    risk = portfolio['risk']
                elif risk_metric == 'var':
                    risk = RiskMetrics.calculate_var(portfolio_returns)
                elif risk_metric == 'cvar':
                    risk = RiskMetrics.calculate_cvar(portfolio_returns)
                else:
                    risk = portfolio['risk']
                
                # Calcular Sharpe Ratio
                sharpe = (portfolio['return'] - self.risk_free_rate) / risk if risk > 0 else 0
                
                frontier_points.append({
                    'risk': risk,
                    'return': portfolio['return'],
                    'sharpe_ratio': sharpe
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(frontier_points)

    def minimize_var(self, confidence_level: float = 0.95) -> Dict:
        """
        Minimiza el Value at Risk del portafolio
        """
        from utils.risk_metrics import RiskMetrics
        
        def objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            return RiskMetrics.calculate_var(portfolio_returns, confidence_level)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        p_return, p_std = self.portfolio_performance(result.x)
        
        return {
            'weights': result.x,
            'return': p_return,
            'risk': objective(result.x),
            'var': objective(result.x)
        }

    def minimize_cvar(self, confidence_level: float = 0.95) -> Dict:
        """
        Minimiza el Conditional Value at Risk del portafolio
        """
        from utils.risk_metrics import RiskMetrics
        
        def objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            return RiskMetrics.calculate_cvar(portfolio_returns, confidence_level)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        p_return, p_std = self.portfolio_performance(result.x)
        
        return {
            'weights': result.x,
            'return': p_return,
            'risk': objective(result.x),
            'cvar': objective(result.x)
        }