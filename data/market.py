"""
Глобальные данные рынка: USDT Dominance, Total Market Cap.
Источник: CoinGecko (бесплатный API, лимит ~30 req/мин).
"""
import requests
import time
import threading

_cache: dict = {}
_lock = threading.Lock()

COINGECKO = "https://api.coingecko.com/api/v3"
TTL = 120  # секунд


def _get_cached(key: str, url: str, ttl: int = TTL) -> dict:
    with _lock:
        cached = _cache.get(key)
        if cached and time.time() - cached["ts"] < ttl:
            return cached["data"]

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            with _lock:
                _cache[key] = {"data": data, "ts": time.time()}
            return data
        # Rate limit — вернуть старый кэш если есть
        if r.status_code == 429 and cached:
            return cached["data"]
    except Exception:
        if cached:
            return cached["data"]

    return {}


def get_global_market() -> dict:
    """
    Возвращает:
      usdt_dominance    — % доминирования USDT
      btc_dominance     — % доминирования BTC
      total_market_cap  — общий MarCap в USD
      total_volume      — объём 24ч в USD
      market_cap_change_24h — изменение MarCap за 24ч в %
      active_cryptocurrencies
    """
    raw = _get_cached("global", f"{COINGECKO}/global")
    if not raw:
        return {}

    d = raw.get("data", {})
    pct = d.get("market_cap_percentage", {})

    return {
        "usdt_dominance":         pct.get("usdt", 0),
        "usdc_dominance":         pct.get("usdc", 0),
        "btc_dominance":          pct.get("btc", 0),
        "eth_dominance":          pct.get("eth", 0),
        "total_market_cap":       d.get("total_market_cap", {}).get("usd", 0),
        "total_volume":           d.get("total_volume", {}).get("usd", 0),
        "market_cap_change_24h":  d.get("market_cap_change_percentage_24h_usd", 0),
        "active_cryptocurrencies": d.get("active_cryptocurrencies", 0),
    }
