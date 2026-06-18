"""
Инвестиционный профиль монеты: токеномика, крупные держатели,
разлоки, рыночные метрики, on-chain прокси.
Источники: CoinGecko (free), CoinMarketCap (public), Binance.
"""
import requests
import time
from typing import Optional

_cache: dict = {}


def _get(key: str, url: str, ttl: int = 600, params: dict = None) -> dict:
    c = _cache.get(key)
    if c and time.time() - c["ts"] < ttl:
        return c["data"]
    try:
        r = requests.get(url, params=params, timeout=12,
                         headers={"accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            _cache[key] = {"data": data, "ts": time.time()}
            return data
    except Exception:
        pass
    return c["data"] if c else {}


def _cg_id(symbol: str) -> Optional[str]:
    """BTCUSDT → bitcoin."""
    clean = symbol.replace("USDT","").replace("BUSD","").lower()
    raw = _get("cg_list", "https://api.coingecko.com/api/v3/coins/list", ttl=86400)
    if not isinstance(raw, list): return None
    # точное совпадение символа
    hits = [c for c in raw if c.get("symbol","").lower() == clean]
    if not hits: return None
    # предпочитаем известные (id без "-" или самый короткий id)
    hits.sort(key=lambda c: (c["id"].count("-"), len(c["id"])))
    return hits[0]["id"]


def get_investment_profile(symbol: str) -> dict:
    """
    Полный инвестиционный профиль монеты.
    Возвращает словарь с разделами: market, tokenomics, community,
    holders_proxy, unlocks_note, dev, description.
    """
    cg_id = _cg_id(symbol)
    if not cg_id:
        return {"error": f"Монета {symbol} не найдена в CoinGecko"}

    data = _get(
        f"inv_{cg_id}", f"https://api.coingecko.com/api/v3/coins/{cg_id}",
        ttl=600,
        params={
            "localization": "false", "tickers": "false",
            "market_data": "true", "community_data": "true",
            "developer_data": "true", "sparkline": "false",
        },
    )
    if not data:
        return {"error": "CoinGecko rate limit или нет данных"}

    md  = data.get("market_data", {})
    com = data.get("community_data", {})
    dev = data.get("developer_data", {})

    def _v(d, *keys, default=None):
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d if d != {} else default

    price_usd   = _v(md, "current_price", "usd", default=0) or 0
    mc_usd      = _v(md, "market_cap", "usd", default=0) or 0
    vol_24h     = _v(md, "total_volume", "usd", default=0) or 0
    ath         = _v(md, "ath", "usd", default=0) or 0
    ath_date    = md.get("ath_date", {}).get("usd", "")[:10]
    ath_change  = _v(md, "ath_change_percentage", "usd", default=0) or 0
    atl         = _v(md, "atl", "usd", default=0) or 0
    atl_date    = md.get("atl_date", {}).get("usd", "")[:10]
    atl_change  = _v(md, "atl_change_percentage", "usd", default=0) or 0
    circ        = md.get("circulating_supply") or 0
    total_sup   = md.get("total_supply") or 0
    max_sup     = md.get("max_supply") or 0
    mc_rank     = data.get("market_cap_rank") or 0
    fdv         = _v(md, "fully_diluted_valuation", "usd", default=0) or 0
    price_chg_7 = _v(md, "price_change_percentage_7d", default=0) or 0
    price_chg_30= _v(md, "price_change_percentage_30d", default=0) or 0
    price_chg_1y= _v(md, "price_change_percentage_1y", default=0) or 0

    # Circulating ratio — прокси на «давление разлока»
    circ_ratio = circ / total_sup if total_sup > 0 else None

    # FDV / MC ratio — мера инфляционного давления
    fdv_mc_ratio = fdv / mc_usd if mc_usd > 0 else None

    # Vol/MC ratio — ликвидность
    vol_mc = vol_24h / mc_usd if mc_usd > 0 else None

    genesis = data.get("genesis_date") or "Неизвестно"
    categories = data.get("categories", [])
    sentiment_up = data.get("sentiment_votes_up_percentage") or 0

    desc = data.get("description", {}).get("en", "")
    desc = desc[:600].replace("<a href=","").replace("</a>","") if desc else ""

    # Ссылки
    links = data.get("links", {})
    homepage   = (links.get("homepage") or [""])[0]
    whitepaper = (links.get("whitepaper") or [""])[0]
    twitter    = links.get("twitter_screen_name", "")
    telegram   = links.get("telegram_channel_identifier", "")
    github     = (links.get("repos_url", {}).get("github") or [""])[0]

    # Community
    twitter_f  = com.get("twitter_followers") or 0
    reddit_sub = com.get("reddit_subscribers") or 0
    tg_users   = com.get("telegram_channel_user_count") or 0

    # Dev activity
    gh_stars   = dev.get("stars") or 0
    gh_forks   = dev.get("forks") or 0
    commits_4w = dev.get("commit_count_4_weeks") or 0
    contributors= dev.get("pull_request_contributors") or 0

    # ── Unlocks / Vesting — оценка по circ_ratio ──
    unlock_assessment = ""
    if circ_ratio is not None:
        if circ_ratio >= 0.95:
            unlock_assessment = "🟢 ~Полностью разлочено (95%+ в обращении). Давление продаж минимально."
        elif circ_ratio >= 0.70:
            unlock_assessment = f"🟡 {circ_ratio*100:.0f}% в обращении. Умеренный потенциал разлоков."
        elif circ_ratio >= 0.40:
            unlock_assessment = f"🟠 {circ_ratio*100:.0f}% в обращении. Значительная часть ещё заблокирована."
        else:
            unlock_assessment = f"🔴 {circ_ratio*100:.0f}% в обращении! Высокий риск инфляции от разлоков."

    # ── FDV / MC интерпретация ──
    fdv_note = ""
    if fdv_mc_ratio:
        if fdv_mc_ratio < 1.5:
            fdv_note = f"🟢 FDV/MC = {fdv_mc_ratio:.1f}x — низкое инфляционное давление."
        elif fdv_mc_ratio < 3:
            fdv_note = f"🟡 FDV/MC = {fdv_mc_ratio:.1f}x — умеренное давление от будущего предложения."
        else:
            fdv_note = f"🔴 FDV/MC = {fdv_mc_ratio:.1f}x — высокое инфляционное давление (много незаминченных токенов)."

    # ── ATH distance ──
    ath_note = ""
    if ath and price_usd:
        dist_from_ath = (ath - price_usd) / ath * 100
        if dist_from_ath < 10:
            ath_note = f"🔥 Цена вблизи ATH (-{dist_from_ath:.1f}%). Осторожно."
        elif dist_from_ath < 50:
            ath_note = f"📊 -{dist_from_ath:.1f}% от ATH ({ath:.4f})"
        else:
            ath_note = f"💎 -{dist_from_ath:.1f}% от ATH. Потенциальная точка входа."

    return {
        "cg_id": cg_id,
        "symbol": symbol,
        "market": {
            "price":        price_usd,
            "market_cap":   mc_usd,
            "fdv":          fdv,
            "volume_24h":   vol_24h,
            "mc_rank":      mc_rank,
            "ath":          ath,
            "ath_date":     ath_date,
            "ath_change":   ath_change,
            "atl":          atl,
            "atl_date":     atl_date,
            "atl_change":   atl_change,
            "change_7d":    price_chg_7,
            "change_30d":   price_chg_30,
            "change_1y":    price_chg_1y,
            "vol_mc_ratio": vol_mc,
        },
        "tokenomics": {
            "circ_supply":  circ,
            "total_supply": total_sup,
            "max_supply":   max_sup,
            "circ_ratio":   circ_ratio,
            "fdv_mc_ratio": fdv_mc_ratio,
            "genesis_date": genesis,
            "categories":   categories[:4],
        },
        "unlock": {
            "assessment":   unlock_assessment,
            "fdv_note":     fdv_note,
            "ath_note":     ath_note,
            "note":         "Точный вестинг-расписание: TokenUnlocks.io, Vestlab.io, Cryptorank.io",
        },
        "community": {
            "twitter_followers":  twitter_f,
            "reddit_subscribers": reddit_sub,
            "telegram_users":     tg_users,
            "sentiment_up":       sentiment_up,
            "twitter":            twitter,
            "telegram":           telegram,
        },
        "dev": {
            "github":       github,
            "stars":        gh_stars,
            "forks":        gh_forks,
            "commits_4w":   commits_4w,
            "contributors": contributors,
        },
        "links": {
            "homepage":   homepage,
            "whitepaper": whitepaper,
        },
        "description": desc,
        "holders_proxy": _get_holders_proxy(symbol, mc_usd, vol_24h, twitter_f),
    }


def _get_holders_proxy(symbol: str, mc: float, vol: float, twitter: int) -> dict:
    """
    Прокси-оценка числа держателей и качества распределения.
    Точные данные on-chain доступны через Nansen/Glassnode (платно).
    """
    notes = []

    if mc > 0:
        if mc > 10e9:
            notes.append("🐋 Large-cap (>$10B) — широкое институциональное присутствие.")
        elif mc > 1e9:
            notes.append("📊 Mid-cap ($1B-$10B) — mix розницы и институций.")
        elif mc > 100e6:
            notes.append("🌱 Small-cap ($100M-$1B) — преимущественно розничные инвесторы.")
        else:
            notes.append("⚠️ Micro-cap (<$100M) — высокий риск манипуляций.")

    if vol and mc:
        ratio = vol / mc
        if ratio > 0.5:
            notes.append("⚡ Очень высокий оборот (Vol/MC > 50%) — активная торговля.")
        elif ratio > 0.1:
            notes.append("✅ Хороший оборот (Vol/MC 10-50%) — нормальная ликвидность.")
        elif ratio > 0.02:
            notes.append("🟡 Умеренный оборот — средняя ликвидность.")
        else:
            notes.append("🔴 Низкий оборот — сложно выйти из позиции.")

    return {
        "notes": notes,
        "disclaimer": "Точное число холдеров: Etherscan/BSCscan/Solscan (on-chain), Nansen (платно).",
    }
