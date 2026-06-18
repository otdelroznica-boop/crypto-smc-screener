"""
Bybit V5 REST API — публичные эндпоинты Linear (USDT-перп).
"""
import requests
import pandas as pd

BYBIT = "https://api.bybit.com"

_session = requests.Session()
_session.headers.update({"User-Agent": "TradingScreener/1.0"})

# Маппинг интервалов Binance → Bybit
_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "1d": "D",
}
# ОИ периоды
_OI_PERIOD_MAP = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d",
}

STABLECOINS = {"USDCUSDT","BUSDUSDT","TUSDUSDT","USDPUSDT","DAIUSDT","FRAXUSDT"}


def _get(path: str, params: dict = None) -> dict | None:
    try:
        r = _session.get(f"{BYBIT}{path}", params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") == 0:
            return j.get("result", {})
        return None
    except Exception:
        return None


# ── Тикеры ────────────────────────────────────────────────────────────────────

def get_futures_tickers(exclude_stablecoins: bool = True) -> pd.DataFrame:
    data = _get("/v5/market/tickers", {"category": "linear"})
    if not data:
        return pd.DataFrame()

    items = data.get("list", [])
    rows = []
    for t in items:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if exclude_stablecoins and sym in STABLECOINS:
            continue
        rows.append({
            "Symbol":    sym,
            "Price":     float(t.get("lastPrice", 0) or 0),
            "Change24h": float(t.get("price24hPcnt", 0) or 0) * 100,
            "Volume24h": float(t.get("turnover24h", 0) or 0),
            "High24h":   float(t.get("highPrice24h", 0) or 0),
            "Low24h":    float(t.get("lowPrice24h", 0) or 0),
        })

    return pd.DataFrame(rows)


# ── Свечи ─────────────────────────────────────────────────────────────────────

def get_klines(symbol: str, interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    iv = _INTERVAL_MAP.get(interval, "5")
    data = _get("/v5/market/kline", {
        "category": "linear", "symbol": symbol,
        "interval": iv, "limit": limit,
    })
    if not data:
        return pd.DataFrame()

    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(data.get("list", []), columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c])

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.rename(columns={"turnover": "quote_volume"})
    # Bybit не даёт taker_buy отдельно — ставим 0 для совместимости
    df["taker_buy_quote"] = df["quote_volume"] * 0.5
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]]


# ── Открытый интерес ──────────────────────────────────────────────────────────

def get_oi_history(symbol: str, period: str = "5m", limit: int = 10) -> pd.DataFrame:
    iv = _OI_PERIOD_MAP.get(period, "5min")
    data = _get("/v5/market/open-interest", {
        "category": "linear", "symbol": symbol,
        "intervalTime": iv, "limit": limit,
    })
    if not data:
        return pd.DataFrame()

    rows = data.get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"]            = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df["sumOpenInterest"]      = pd.to_numeric(df["openInterest"])
    # Получить цену для перевода в USD
    ticker_data = _get("/v5/market/tickers", {"category": "linear", "symbol": symbol})
    price = 0.0
    if ticker_data and ticker_data.get("list"):
        price = float(ticker_data["list"][0].get("lastPrice", 0) or 0)
    df["sumOpenInterestValue"] = df["sumOpenInterest"] * price
    return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]].sort_values("timestamp")


def get_oi_change(symbol: str, period: str = "5m", lookback: int = 6) -> tuple[float, float]:
    df = get_oi_history(symbol, period=period, limit=lookback + 1)
    if df.empty or len(df) < 2:
        return 0.0, 0.0
    oi_now  = df["sumOpenInterestValue"].iloc[-1]
    oi_prev = df["sumOpenInterestValue"].iloc[0]
    if oi_prev == 0:
        return 0.0, oi_now
    return (oi_now - oi_prev) / oi_prev * 100, oi_now


# ── CVD ───────────────────────────────────────────────────────────────────────

def compute_cvd(symbol: str, interval: str = "5m", limit: int = 50) -> pd.DataFrame:
    df = get_klines(symbol, interval=interval, limit=limit)
    if df.empty:
        return pd.DataFrame()
    df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote"]
    df["delta"]            = df["taker_buy_quote"] - df["taker_sell_quote"]
    df["cvd"]              = df["delta"].cumsum()
    return df


def get_cvd_signal(symbol: str, interval: str = "5m", limit: int = 20) -> int:
    df = compute_cvd(symbol, interval=interval, limit=limit)
    if df.empty:
        return 0
    recent = df["delta"].tail(3).sum()
    total  = df["quote_volume"].tail(3).sum()
    if recent >  total * 0.05: return  1
    if recent < -total * 0.05: return -1
    return 0


# ── Фандинг ───────────────────────────────────────────────────────────────────

def get_funding_rates(symbols: list[str] | None = None) -> pd.DataFrame:
    data = _get("/v5/market/tickers", {"category": "linear"})
    if not data:
        return pd.DataFrame()
    rows = []
    for t in data.get("list", []):
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if symbols and sym not in symbols:
            continue
        fr = t.get("fundingRate")
        if fr is not None:
            rows.append({"Symbol": sym, "Funding": float(fr) * 100})
    return pd.DataFrame(rows)
