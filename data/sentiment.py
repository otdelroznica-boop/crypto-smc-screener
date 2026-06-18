"""
Сентимент рынка: Fear & Greed, trending coins, социальные сигналы.
Все источники — бесплатные публичные API.
"""
import requests, time

_cache: dict = {}


def _cached(key: str, url: str, ttl: int = 300, params: dict = None) -> dict | list:
    cached = _cache.get(key)
    if cached and time.time() - cached["ts"] < ttl:
        return cached["data"]
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            _cache[key] = {"data": data, "ts": time.time()}
            return data
    except Exception:
        pass
    return cached["data"] if cached else {}


# ═══════════════════════════════════════════════════════════
#  FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════

def get_fear_greed() -> dict:
    """
    Индекс страха и жадности от alternative.me.
    Возвращает: value (0-100), label, yesterday, week_avg.
    """
    raw = _cached("fng", "https://api.alternative.me/fng/?limit=7&format=json", ttl=3600)
    if not raw or "data" not in raw:
        return {}

    data = raw["data"]
    cur  = data[0]
    vals = [int(d["value"]) for d in data]

    label_map = {
        (0,  24):  "😱 Экстремальный страх",
        (25, 44):  "😰 Страх",
        (45, 55):  "😐 Нейтрально",
        (56, 74):  "😏 Жадность",
        (75, 100): "🤑 Экстремальная жадность",
    }
    v = int(cur["value"])
    label = next((l for (lo, hi), l in label_map.items() if lo <= v <= hi), "–")

    return {
        "value":     v,
        "label":     label,
        "yesterday": vals[1] if len(vals) > 1 else v,
        "week_avg":  int(sum(vals) / len(vals)),
        "history":   vals[:7],
    }


# ═══════════════════════════════════════════════════════════
#  TRENDING COINS (CoinGecko)
# ═══════════════════════════════════════════════════════════

def get_trending_coins() -> list[dict]:
    """
    Топ-7 трендовых монет за 24ч от CoinGecko.
    Возвращает список {symbol, name, rank, price_change_24h}.
    """
    raw = _cached("trending", "https://api.coingecko.com/api/v3/search/trending", ttl=600)
    if not raw or "coins" not in raw:
        return []

    result = []
    for item in raw["coins"][:7]:
        c = item.get("item", {})
        result.append({
            "symbol": c.get("symbol", "").upper(),
            "name":   c.get("name", ""),
            "rank":   c.get("market_cap_rank", 0),
            "score":  c.get("score", 0),
            "thumb":  c.get("thumb", ""),
        })
    return result


# ═══════════════════════════════════════════════════════════
#  ТОПОВЫЕ ГЕЙНЕРЫ / ЛУЗЕРЫ (CoinGecko)
# ═══════════════════════════════════════════════════════════

def get_top_movers(vs_currency: str = "usd", limit: int = 5) -> dict:
    """Топ гейнеры и лузеры за 24ч."""
    raw = _cached(
        "movers",
        "https://api.coingecko.com/api/v3/coins/markets",
        ttl=300,
        params={
            "vs_currency":    vs_currency,
            "order":          "market_cap_desc",
            "per_page":       100,
            "page":           1,
            "price_change_percentage": "24h",
        },
    )
    if not raw or not isinstance(raw, list):
        return {"gainers": [], "losers": []}

    valid = [c for c in raw if c.get("price_change_percentage_24h") is not None]
    gainers = sorted(valid, key=lambda c: c["price_change_percentage_24h"], reverse=True)[:limit]
    losers  = sorted(valid, key=lambda c: c["price_change_percentage_24h"])[:limit]

    def fmt(coins):
        return [{"symbol": c["symbol"].upper(),
                 "name":   c["name"],
                 "change": round(c["price_change_percentage_24h"], 2),
                 "volume": c.get("total_volume", 0)} for c in coins]

    return {"gainers": fmt(gainers), "losers": fmt(losers)}


# ═══════════════════════════════════════════════════════════
#  COIN METRICS (для конкретной монеты)
# ═══════════════════════════════════════════════════════════

def get_coin_sentiment(coingecko_id: str) -> dict:
    """
    Данные сообщества и разработки для монеты.
    Возвращает: sentiment_up%, sentiment_down%, community score, etc.
    """
    raw = _cached(
        f"coin_{coingecko_id}",
        f"https://api.coingecko.com/api/v3/coins/{coingecko_id}",
        ttl=600,
        params={"localization": "false", "tickers": "false",
                "market_data": "false", "community_data": "true",
                "developer_data": "false"},
    )
    if not raw:
        return {}

    sent_up   = raw.get("sentiment_votes_up_percentage", 0)
    sent_down = raw.get("sentiment_votes_down_percentage", 0)
    comm      = raw.get("community_data", {})

    return {
        "sentiment_up":       round(sent_up or 0, 1),
        "sentiment_down":     round(sent_down or 0, 1),
        "twitter_followers":  comm.get("twitter_followers", 0),
        "reddit_subscribers": comm.get("reddit_subscribers", 0),
        "telegram_users":     comm.get("telegram_channel_user_count", 0),
        "description":        raw.get("description", {}).get("en", "")[:300],
    }


# ═══════════════════════════════════════════════════════════
#  SYMBOL → CoinGecko ID
# ═══════════════════════════════════════════════════════════

_SYMBOL_MAP: dict = {}

def symbol_to_cg_id(symbol: str) -> str | None:
    """Конвертирует BTCUSDT → bitcoin."""
    global _SYMBOL_MAP
    clean = symbol.replace("USDT", "").lower()

    if not _SYMBOL_MAP:
        raw = _cached("cg_list", "https://api.coingecko.com/api/v3/coins/list",
                      ttl=86400)  # раз в сутки
        if isinstance(raw, list):
            for c in raw:
                _SYMBOL_MAP[c.get("symbol", "").lower()] = c.get("id", "")

    return _SYMBOL_MAP.get(clean)


# ═══════════════════════════════════════════════════════════
#  COMPOSITE SOCIAL SCORE
# ═══════════════════════════════════════════════════════════

def get_social_score(symbol: str) -> dict:
    """
    Агрегированный социальный сигнал по монете.
    """
    fg      = get_fear_greed()
    trend   = get_trending_coins()
    cg_id   = symbol_to_cg_id(symbol)
    coin_d  = get_coin_sentiment(cg_id) if cg_id else {}
    in_trend = any(t["symbol"] == symbol.replace("USDT","") for t in trend)

    score = 0
    notes = []

    fg_val = fg.get("value", 50)
    if fg_val < 30:
        score += 2
        notes.append(f"😱 Extreme Fear ({fg_val}) — исторически хорошая зона покупки")
    elif fg_val < 45:
        score += 1
        notes.append(f"😰 Fear ({fg_val}) — рынок в страхе, осторожный лонг")
    elif fg_val > 75:
        score -= 1
        notes.append(f"🤑 Extreme Greed ({fg_val}) — рынок перегрет, осторожно с лонгами")
    else:
        notes.append(f"😐 Нейтральный сентимент ({fg_val})")

    if in_trend:
        score += 1
        notes.append(f"🔥 Монета в топ-7 трендов CoinGecko")

    sent_up = coin_d.get("sentiment_up", 50)
    if sent_up > 70:
        score += 1
        notes.append(f"👍 Сентимент сообщества позитивный ({sent_up:.0f}%↑)")
    elif sent_up < 30:
        score -= 1
        notes.append(f"👎 Сентимент негативный ({sent_up:.0f}%↑)")

    return {
        "score":    score,
        "notes":    notes,
        "fg":       fg,
        "trending": trend,
        "coin":     coin_d,
        "in_trend": in_trend,
    }
