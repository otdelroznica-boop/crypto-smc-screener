"""
Единый интерфейс к биржам.
Переключение: exchange = "binance" | "bybit"
"""
import pandas as pd
import data.binance as _bn
import data.bybit   as _bb


def _mod(exchange: str):
    return _bb if exchange == "bybit" else _bn


def get_futures_tickers(exchange: str = "binance", exclude_stablecoins: bool = True) -> pd.DataFrame:
    return _mod(exchange).get_futures_tickers(exclude_stablecoins=exclude_stablecoins)


def get_klines(symbol: str, interval: str = "5m", limit: int = 200,
               exchange: str = "binance") -> pd.DataFrame:
    return _mod(exchange).get_klines(symbol, interval=interval, limit=limit)


def get_oi_history(symbol: str, period: str = "5m", limit: int = 50,
                   exchange: str = "binance") -> pd.DataFrame:
    return _mod(exchange).get_oi_history(symbol, period=period, limit=limit)


def get_oi_change(symbol: str, period: str = "5m", lookback: int = 6,
                  exchange: str = "binance") -> tuple[float, float]:
    return _mod(exchange).get_oi_change(symbol, period=period, lookback=lookback)


def compute_cvd(symbol: str, interval: str = "5m", limit: int = 50,
                exchange: str = "binance") -> pd.DataFrame:
    return _mod(exchange).compute_cvd(symbol, interval=interval, limit=limit)


def get_cvd_signal(symbol: str, interval: str = "5m", limit: int = 20,
                   exchange: str = "binance") -> int:
    return _mod(exchange).get_cvd_signal(symbol, interval=interval, limit=limit)


def get_funding_rates(symbols: list[str] | None = None,
                      exchange: str = "binance") -> pd.DataFrame:
    return _mod(exchange).get_funding_rates(symbols=symbols)
