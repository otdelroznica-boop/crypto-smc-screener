"""
Binance Futures REST API — публичные эндпоинты (авторизация не нужна).
"""
import requests
import pandas as pd
import time

FAPI = "https://fapi.binance.com"

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
})

STABLECOINS = {
    "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT",
    "USDPUSDT", "DAIUSDT", "FRAXUSDT", "USTCUSDT",
}


_last_error: str = ""

def _get(endpoint: str, params: dict = None, timeout: int = 15) -> dict | list | None:
    global _last_error
    try:
        r = _session.get(f"{FAPI}{endpoint}", params=params, timeout=timeout)
        r.raise_for_status()
        _last_error = ""
        return r.json()
    except Exception as e:
        _last_error = str(e)
        return None

def get_last_error() -> str:
    return _last_error


# ── Тикеры ────────────────────────────────────────────────────────────────────

def get_futures_tickers(exclude_stablecoins: bool = True) -> pd.DataFrame:
    """Все USDT-перп тикеры с 24ч статистикой."""
    data = _get("/fapi/v1/ticker/24hr")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")]

    if exclude_stablecoins:
        df = df[~df["symbol"].isin(STABLECOINS)]

    num_cols = ["lastPrice", "priceChangePercent", "quoteVolume", "highPrice", "lowPrice"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={
        "symbol":               "Symbol",
        "lastPrice":            "Price",
        "priceChangePercent":   "Change24h",
        "quoteVolume":          "Volume24h",
        "highPrice":            "High24h",
        "lowPrice":             "Low24h",
    })

    return df[["Symbol", "Price", "Change24h", "Volume24h", "High24h", "Low24h"]].copy()


# ── Открытый интерес ──────────────────────────────────────────────────────────

def get_oi_history(symbol: str, period: str = "5m", limit: int = 10) -> pd.DataFrame:
    """История ОИ (в USD) для символа."""
    data = _get("/futures/data/openInterestHist", {
        "symbol": symbol, "period": period, "limit": limit
    })
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["sumOpenInterest"]      = pd.to_numeric(df["sumOpenInterest"])
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
    return df


def get_oi_change(symbol: str, period: str = "5m", lookback: int = 6) -> tuple[float, float]:
    """
    Возвращает (oi_change_pct, oi_usd_now).
    Сравнивает последнее значение с точкой lookback периодов назад.
    """
    df = get_oi_history(symbol, period=period, limit=lookback + 1)
    if df.empty or len(df) < 2:
        return 0.0, 0.0

    oi_now  = df["sumOpenInterestValue"].iloc[-1]
    oi_prev = df["sumOpenInterestValue"].iloc[0]

    if oi_prev == 0:
        return 0.0, oi_now

    return (oi_now - oi_prev) / oi_prev * 100, oi_now


# ── Свечи ─────────────────────────────────────────────────────────────────────

def get_klines(symbol: str, interval: str = "5m", limit: int = 100) -> pd.DataFrame:
    """
    Свечи с полями: timestamp, open, high, low, close,
                    volume (base), quote_volume, taker_buy_quote.
    """
    data = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c])

    return df[["timestamp", "open", "high", "low", "close",
               "volume", "quote_volume", "taker_buy_quote"]].copy()


# ── CVD ───────────────────────────────────────────────────────────────────────

def compute_cvd(symbol: str, interval: str = "5m", limit: int = 50) -> pd.DataFrame:
    """
    Реальный CVD из taker buy/sell volume Binance.
    delta = taker_buy_quote - taker_sell_quote
    cvd   = cumsum(delta)
    """
    df = get_klines(symbol, interval=interval, limit=limit)
    if df.empty:
        return pd.DataFrame()

    df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote"]
    df["delta"]            = df["taker_buy_quote"] - df["taker_sell_quote"]
    df["cvd"]              = df["delta"].cumsum()
    return df


def get_cvd_signal(symbol: str, interval: str = "5m", limit: int = 20) -> int:
    """
    +1 если CVD растёт (последние 3 свечи delta > 0 суммарно)
    -1 если CVD падает
     0 нейтрально
    """
    df = compute_cvd(symbol, interval=interval, limit=limit)
    if df.empty:
        return 0
    recent_delta = df["delta"].tail(3).sum()
    if recent_delta > df["quote_volume"].tail(3).sum() * 0.05:
        return 1
    if recent_delta < -df["quote_volume"].tail(3).sum() * 0.05:
        return -1
    return 0


# ── Фандинг ───────────────────────────────────────────────────────────────────

def get_funding_rates(symbols: list[str] | None = None) -> pd.DataFrame:
    """Текущие ставки фандинга."""
    data = _get("/fapi/v1/premiumIndex")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")]
    df["lastFundingRate"] = pd.to_numeric(df["lastFundingRate"]) * 100  # в %

    if symbols:
        df = df[df["symbol"].isin(symbols)]

    return df[["symbol", "lastFundingRate"]].rename(columns={
        "symbol": "Symbol", "lastFundingRate": "Funding"
    })
