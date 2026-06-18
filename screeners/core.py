"""
Основная логика скрининга.
Собирает данные по топ N монетам и возвращает DataFrame с сигналами и Score.
"""
import time
import pandas as pd

from data.exchange import (
    get_futures_tickers,
    get_oi_change,
    get_cvd_signal,
    get_funding_rates,
)


# ── Score weights ─────────────────────────────────────────────────────────────
SCORE_PUMP        = 1   # резкое изменение цены
SCORE_OI_BULL     = 2   # ОИ растёт + цена растёт (лонги набирают)
SCORE_OI_SPIKE    = 1   # ОИ просто резко вырос (нейтрально)
SCORE_CVD_BULL    = 1   # CVD положительный
SCORE_FUNDING     = 1   # экстремальный фандинг (перегрев)


def run_screener(
    min_price_pct: float = 3.0,
    min_oi_pct:    float = 5.0,
    oi_period:     str   = "5m",
    oi_lookback:   int   = 6,
    cvd_interval:  str   = "5m",
    min_volume:    float = 5_000_000,
    top_n:         int   = 60,
    exclude_stablecoins: bool = True,
    funding_threshold:   float = 0.05,
    exchange:      str   = "binance",
) -> pd.DataFrame:
    """
    Запускает полный скрининг.

    Возвращает DataFrame со столбцами:
      Symbol, Price, Change24h, Volume24h, OI_Change, OI_USD,
      CVD, Funding, Score, Signals, Timestamp
    """

    # 1. Получить все тикеры
    tickers = get_futures_tickers(exchange=exchange, exclude_stablecoins=exclude_stablecoins)
    if tickers.empty:
        return pd.DataFrame()

    # 2. Фильтр по объёму + топ N
    tickers = tickers[tickers["Volume24h"] >= min_volume]
    tickers = tickers.nlargest(top_n, "Volume24h").copy()

    symbols = tickers["Symbol"].tolist()

    # 3. Фандинг (один запрос для всех)
    funding_df = get_funding_rates(symbols, exchange=exchange)
    if not funding_df.empty:
        tickers = tickers.merge(funding_df, on="Symbol", how="left")
    else:
        tickers["Funding"] = 0.0

    # 4. Пройти по монетам
    results = []

    for _, row in tickers.iterrows():
        symbol      = row["Symbol"]
        price       = row["Price"]
        change24h   = row["Change24h"]
        volume24h   = row["Volume24h"]
        funding     = row.get("Funding", 0.0) or 0.0

        score   = 0
        signals = []

        # ── Pump/Dump ──────────────────────────────────────────────────────
        if abs(change24h) >= min_price_pct:
            score += SCORE_PUMP
            arrow  = "↑" if change24h > 0 else "↓"
            signals.append(f"PUMP{arrow}{abs(change24h):.1f}%")

        # ── Открытый интерес ───────────────────────────────────────────────
        oi_change, oi_usd = get_oi_change(symbol, period=oi_period, lookback=oi_lookback, exchange=exchange)

        if abs(oi_change) >= min_oi_pct:
            if oi_change > 0 and change24h >= 0:
                # Лонги набирают позиции + цена растёт = сигнал
                score += SCORE_OI_BULL
                signals.append(f"OI↑{oi_change:.1f}%")
            elif oi_change > 0 and change24h < 0:
                # ОИ растёт, цена падает = шорты (не наш сетап)
                signals.append(f"OI↑ШОРТЫ")
            else:
                score += SCORE_OI_SPIKE
                signals.append(f"OI↓{oi_change:.1f}%")

        # ── CVD ────────────────────────────────────────────────────────────
        cvd_signal = get_cvd_signal(symbol, interval=cvd_interval, exchange=exchange)
        if cvd_signal > 0:
            score += SCORE_CVD_BULL
            signals.append("CVD↑")
        elif cvd_signal < 0:
            signals.append("CVD↓")

        # ── Фандинг ────────────────────────────────────────────────────────
        if abs(funding) >= funding_threshold:
            score += SCORE_FUNDING
            direction = "+" if funding > 0 else "-"
            signals.append(f"FUND{direction}{abs(funding):.3f}%")

        results.append({
            "Symbol":    symbol,
            "Price":     price,
            "Change24h": change24h,
            "Volume24h": volume24h,
            "OI_Change": round(oi_change, 2),
            "OI_USD":    oi_usd,
            "CVD":       "↑" if cvd_signal > 0 else ("↓" if cvd_signal < 0 else "–"),
            "Funding":   round(funding, 4),
            "Score":     score,
            "Signals":   " | ".join(signals) if signals else "–",
            "Timestamp": pd.Timestamp.now(),
        })

        time.sleep(0.08)  # Rate limiting Binance: ~12 req/sec

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df
