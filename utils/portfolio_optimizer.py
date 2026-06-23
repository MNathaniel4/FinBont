import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from itertools import product

class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calcula retorno y riesgo del portafolio (anualizados)"""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(weights.T @ self.cov_matrix @ weights)
        return portfolio_return, portfolio_std
    
    def minimize_risk(self, target_return: Optional[float] = None) -> Dict:
        """Minimiza el riesgo (desviación estándar)"""
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
    
    def maximize_return(self, max_risk: Optional[float] = None) -> Dict:
        """Maximiza el retorno para un nivel de riesgo dado"""
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
        """Maximiza el Sharpe Ratio"""
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
    

    def generate_efficient_frontier(
            self, 
            n_points: int = 50, 
            risk_metric: str = 'std'
        ) -> pd.DataFrame:
            """
            Genera la frontera eficiente optimizando directamente para la métrica de riesgo seleccionada.
            risk_metric: 'std', 'var', 'cvar'
            """
            from utils.risk_metrics import RiskMetrics
            
            # 1. Encontrar el rango de retornos posibles en base a los activos individuales
            # El retorno mínimo razonable es el del portafolio de mínima varianza (o mínimo VaR/CVaR global)
            # El retorno máximo es siempre el del activo con mayor rendimiento individual
            min_risk_port = self.minimize_risk()
            max_return_port = self.maximize_return()
            
            target_returns = np.linspace(
                min_risk_port['return'], 
                max_return_port['return'], 
                n_points
            )
            
            frontier_points = []
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_weights = np.array([1/self.n_assets] * self.n_assets)
            
            for target_return in target_returns:
                try:
                    # Definir restricciones comunes: pesos suman 1 y el retorno esperado es igual al target
                    constraints = [
                        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                        {'type': 'eq', 'fun': lambda w: np.sum(self.mean_returns * w) - target_return}
                    ]
                    
                    # Definir la función objetivo dinámicamente según la métrica
                    if risk_metric == 'std':
                        def objective(w):
                            return np.sqrt(w.T @ self.cov_matrix @ w)
                            
                    elif risk_metric == 'var':
                        def objective(w):
                            portfolio_daily_returns = (self.returns * w).sum(axis=1)
                            # Minimizamos el VaR anualizado
                            return RiskMetrics.calculate_var(portfolio_daily_returns) * np.sqrt(252)
                            
                    elif risk_metric == 'cvar':
                        def objective(w):
                            portfolio_daily_returns = (self.returns * w).sum(axis=1)
                            # Minimizamos el CVaR anualizado
                            return RiskMetrics.calculate_cvar(portfolio_daily_returns) * np.sqrt(252)
                    else:
                        def objective(w):
                            return np.sqrt(w.T @ self.cov_matrix @ w)
                    
                    # Ejecutar la optimización real para la métrica seleccionada
                    result = minimize(
                        objective, 
                        initial_weights,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    if result.success:
                        weights = result.x
                        risk = result.fun
                        portfolio_return = np.sum(self.mean_returns * weights)
                        
                        if risk > 0 and not np.isnan(risk) and not np.isinf(risk) and risk < 5.0:
                            sharpe = (portfolio_return - self.risk_free_rate) / risk
                            frontier_points.append({
                                'risk': risk,
                                'return': portfolio_return,
                                'sharpe_ratio': sharpe
                            })
                except Exception as e:
                    continue
                    
            if len(frontier_points) < 3:
                # Fallback en caso de que falle SLSQP con VaR/CVaR en algún punto crítico
                return self.generate_efficient_frontier(n_points, 'std')
                
            df = pd.DataFrame(frontier_points)
            return df.sort_values('return')

    def minimize_var(self, confidence_level: float = 0.95) -> Dict:
        """Minimiza el Value at Risk del portafolio"""
        from utils.risk_metrics import RiskMetrics
        
        def objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            return RiskMetrics.calculate_var(portfolio_returns, confidence_level)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Múltiples puntos de inicio
        best_result = None
        best_obj = float('inf')
        
        np.random.seed(42)
        for _ in range(10):
            initial_weights = np.random.dirichlet(np.ones(self.n_assets))
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_obj:
                best_result = result
                best_obj = result.fun
        
        if best_result is None:
            return self.minimize_risk()
        
        weights = best_result.x
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        var_daily = RiskMetrics.calculate_var(portfolio_returns, confidence_level)
        var_annual = var_daily * np.sqrt(252)
        
        p_return, p_std = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': p_return,
            'risk': var_annual,
            'var': var_annual
        }
    
    def minimize_cvar(self, confidence_level: float = 0.95) -> Dict:
        """Minimiza el CVaR del portafolio"""
        from utils.risk_metrics import RiskMetrics
        
        def objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            return RiskMetrics.calculate_cvar(portfolio_returns, confidence_level)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        best_result = None
        best_obj = float('inf')
        
        np.random.seed(42)
        for _ in range(10):
            initial_weights = np.random.dirichlet(np.ones(self.n_assets))
            
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success and result.fun < best_obj:
                best_result = result
                best_obj = result.fun
        
        if best_result is None:
            return self.minimize_risk()
        
        weights = best_result.x
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        cvar_daily = RiskMetrics.calculate_cvar(portfolio_returns, confidence_level)
        cvar_annual = cvar_daily * np.sqrt(252)
        
        p_return, p_std = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': p_return,
            'risk': cvar_annual,
            'cvar': cvar_annual
        }
    
    def maximize_var_sharpe_ratio(self, confidence_level: float = 0.95) -> Dict:
        """
        Maximiza el Ratio Sharpe-VaR buscando sobre la frontera eficiente
        """
        from utils.risk_metrics import RiskMetrics
        
        # Generar la frontera con VaR
        frontier = self.generate_efficient_frontier(n_points=100, risk_metric='var')
        
        if frontier.empty:
            # Fallback
            return self.maximize_sharpe_ratio()
        
        # Calcular el ratio para cada punto de la frontera
        frontier['var_sharpe'] = (frontier['return'] - self.risk_free_rate) / frontier['risk']
        
        # Encontrar el punto con máximo ratio
        best_idx = frontier['var_sharpe'].idxmax()
        best_point = frontier.loc[best_idx]
        
        # Obtener los pesos para ese punto
        target_return = best_point['return']
        portfolio = self.minimize_risk(target_return)
        weights = portfolio['weights']
        
        # Calcular métricas con los pesos óptimos
        portfolio_returns = (self.returns * weights).sum(axis=1)
        var_daily = RiskMetrics.calculate_var(portfolio_returns, confidence_level)
        var_annual = var_daily * np.sqrt(252)
        
        p_return, p_std = self.portfolio_performance(weights)
        
        return{
            'weights': weights,
            'return': p_return,
            'risk': var_annual,
            'var_sharpe_ratio': best_point['var_sharpe']
        }


    def maximize_staar_ratio(self, confidence_level: float = 0.95) -> Dict:
        """
        Maximiza el STARR Ratio buscando sobre la frontera eficiente
        """
        from utils.risk_metrics import RiskMetrics
        
        # Generar la frontera con CVaR
        frontier = self.generate_efficient_frontier(n_points=100, risk_metric='cvar')
        
        if frontier.empty:
            # Fallback
            return self.maximize_sharpe_ratio()
        
        # Calcular el ratio para cada punto de la frontera
        frontier['staar'] = (frontier['return'] - self.risk_free_rate) / frontier['risk']
        
        # Encontrar el punto con máximo ratio
        best_idx = frontier['staar'].idxmax()
        best_point = frontier.loc[best_idx]
        
        # Obtener los pesos para ese punto
        target_return = best_point['return']
        portfolio = self.minimize_risk(target_return)
        weights = portfolio['weights']
        
        # Calcular métricas con los pesos óptimos
        portfolio_returns = (self.returns * weights).sum(axis=1)
        cvar_daily = RiskMetrics.calculate_cvar(portfolio_returns, confidence_level)
        cvar_annual = cvar_daily * np.sqrt(252)
        
        p_return, p_std = self.portfolio_performance(weights)
        
        return {
            'weights': weights,
            'return': p_return,
            'risk': cvar_annual,
            'staar_ratio': best_point['staar']
        }