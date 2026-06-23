import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple

class DataFetcher:
    @staticmethod
    def fetch_data(
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos para múltiples tickers
        """
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            interval=interval,
            progress=False
        )
        
        # Manejar el caso de un solo ticker
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product(
                [data.columns, tickers]
            )
        
        return data
    
    @staticmethod
    def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos diarios desde precios de cierre
        """
        if 'Close' in prices.columns:
            close_prices = prices['Close']
        else:
            close_prices = prices
            
        returns = close_prices.pct_change().dropna()
        return returns
    
    @staticmethod
    def split_periods(
        data: pd.DataFrame,
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en periodos de entrenamiento y prueba
        """
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        return train_data, test_data