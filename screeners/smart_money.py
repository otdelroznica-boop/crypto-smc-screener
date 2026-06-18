"""
Smart Money Concepts (SMC).
Имплементация: ордер блоки, FVG (имбаланс), TTS (Stop Hunt), TDP, ликвидность,
свинг хай/лоу, брейкер блоки, named liquidity levels, RSI дивергенция, killzones.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import datetime


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    kind:       str          # "bullish" | "bearish"
    high:       float        # верхняя граница зоны
    low:        float        # нижняя граница зоны
    timestamp:  pd.Timestamp
    mitigated:  bool         # цена уже возвращалась в зону?
    strength:   float        # относительный объём свечи (1.0 = средний)
    has_fvg:    bool = False  # есть FVG после OB
    has_bos:    bool = False  # есть BOS после OB
    description: str = ""

@dataclass
class BreakerBlock:
    """OB, который цена пробила насквозь — теперь действует как противоположная зона."""
    kind:        str    # "bullish" (бывший медвежий OB) | "bearish" (бывший бычий OB)
    high:        float
    low:         float
    timestamp:   pd.Timestamp
    description: str = ""

@dataclass
class MitigationBlock:
    """Зона, где цена протестировала OB частично (50%) и отскочила."""
    kind:       str    # "bullish" | "bearish"
    high:       float
    low:        float
    mid:        float
    timestamp:  pd.Timestamp
    description: str = ""

@dataclass
class FVG:
    kind:       str          # "bullish" | "bearish"
    high:       float
    low:        float
    timestamp:  pd.Timestamp
    filled:     bool         # зазор закрыт?
    size_pct:   float        # размер зазора в % от цены

@dataclass
class SwingPoint:
    kind:       str          # "high" | "low"
    price:      float
    idx:        int
    timestamp:  pd.Timestamp
    strength:   int          # 3 (3-candle) или 5 (5-candle rule)

@dataclass
class StopHunt:
    kind:       str          # "bullish" (захват под лоу) | "bearish" (захват над хаем)
    swept_level: float       # уровень, который был захвачен
    candle_idx:  int
    timestamp:  pd.Timestamp
    reversal_confirmed: bool # следующая свеча подтвердила разворот?
    description: str = ""

@dataclass
class LiquidityZone:
    kind:       str          # "equal_highs" | "equal_lows"
    price:      float
    touches:    int
    timestamp:  pd.Timestamp
    swept:      bool

@dataclass
class NamedLiquidityLevel:
    """PMH, PWH, PDH, EQH, EQL, PDL, PWL, PML — именованные пулы ликвидности."""
    kind:       str   # "PDH" | "PDL" | "PWH" | "PWL" | "EQH" | "EQL" | "HOD" | "LOD"
    price:      float
    swept:      bool
    description: str = ""

@dataclass
class TDPLevel:
    """True Daily Profile — ключевые дневные уровни."""
    kind:       str          # "prev_high" | "prev_low" | "prev_close" | "open"
    price:      float
    description: str = ""

@dataclass
class RSIDivergence:
    kind:       str   # "bullish_classic" | "bearish_classic" | "bullish_hidden" | "bearish_hidden"
    price1:     float
    price2:     float
    rsi1:       float
    rsi2:       float
    idx1:       int
    idx2:       int
    description: str = ""

@dataclass
class KillzoneInfo:
    active_zone:  str    # "Asia" | "London" | "New York" | "London Close" | "Inactive"
    asia_range_high: float
    asia_range_low:  float
    current_utc:  str


# ═══════════════════════════════════════════════════════════
#  SWING HIGH / LOW  (PDF 1: 3-candle and 5-candle rules)
# ═══════════════════════════════════════════════════════════

def find_swing_points(df: pd.DataFrame, strength: int = 3,
                      lookback: int = 100) -> list[SwingPoint]:
    """
    Свинг Хай: центральная свеча — наивысший хай из N свечей по обе стороны.
    Свинг Лоу: центральная свеча — наинизший лоу из N свечей по обе стороны.
    strength=3 → 1 свеча с каждой стороны (3-candle rule)
    strength=5 → 2 свечи с каждой стороны (5-candle rule)
    """
    n = strength // 2  # сколько свечей по каждую сторону
    arr = df.tail(lookback).reset_index(drop=True)
    points = []
    for i in range(n, len(arr) - n):
        h_window = arr["high"].iloc[i-n:i+n+1]
        l_window = arr["low"].iloc[i-n:i+n+1]
        if arr["high"].iloc[i] == h_window.max() and list(h_window).count(h_window.max()) == 1:
            points.append(SwingPoint(
                kind="high",
                price=float(arr["high"].iloc[i]),
                idx=len(df) - lookback + i,
                timestamp=arr["timestamp"].iloc[i],
                strength=strength,
            ))
        if arr["low"].iloc[i] == l_window.min() and list(l_window).count(l_window.min()) == 1:
            points.append(SwingPoint(
                kind="low",
                price=float(arr["low"].iloc[i]),
                idx=len(df) - lookback + i,
                timestamp=arr["timestamp"].iloc[i],
                strength=strength,
            ))
    return points


# ═══════════════════════════════════════════════════════════
#  ORDER BLOCKS (улучшено по PDF 3)
# ═══════════════════════════════════════════════════════════

def find_order_blocks(df: pd.DataFrame, move_threshold: float = 0.008,
                      lookback_candles: int = 5) -> list[OrderBlock]:
    """
    Бычий OB = последняя медвежья свеча перед сильным ростом (BOS).
    Медвежий OB = последняя бычья свеча перед сильным падением (BOS).
    Дополнительно проверяем: есть ли FVG после OB.
    """
    obs = []
    avg_vol = df["quote_volume"].rolling(20).mean()
    fvgs = find_fvg(df, min_size_pct=0.05)

    # Вспомогательная: BOS вверх — цена пробила недавний свинг хай
    def _has_bos_up(after_idx: int) -> bool:
        if after_idx >= len(df) - 2:
            return False
        seg = df.iloc[after_idx:]
        ref_high = float(df["high"].iloc[after_idx])
        return bool((seg["high"] > ref_high).any())

    def _has_bos_dn(after_idx: int) -> bool:
        if after_idx >= len(df) - 2:
            return False
        seg = df.iloc[after_idx:]
        ref_low = float(df["low"].iloc[after_idx])
        return bool((seg["low"] < ref_low).any())

    def _has_fvg_after(ob_idx: int, kind: str) -> bool:
        for f in fvgs:
            # FVG должен быть после OB
            f_idx_approx = df[df["timestamp"] == f.timestamp].index
            if len(f_idx_approx) == 0:
                continue
            fi = f_idx_approx[0]
            if fi > ob_idx and f.kind == kind:
                return True
        return False

    for i in range(lookback_candles + 1, len(df) - 1):
        c = df.iloc[i]
        move = (c["close"] - c["open"]) / c["open"]

        # Сильная бычья свеча → ищем медвежий ОБ перед ней
        if move > move_threshold:
            for j in range(i-1, max(i-lookback_candles-1, 0), -1):
                prev = df.iloc[j]
                if prev["close"] < prev["open"]:  # медвежья
                    mid_high = max(prev["open"], prev["close"])
                    mid_low  = min(prev["open"], prev["close"])

                    future = df.iloc[i+1:]
                    mitigated = not future.empty and (
                        (future["low"] <= mid_high) & (future["high"] >= mid_low)
                    ).any()

                    rel_vol = float(df["quote_volume"].iloc[j]) / float(avg_vol.iloc[j]) \
                        if avg_vol.iloc[j] > 0 else 1.0

                    has_fvg = _has_fvg_after(j, "bullish")
                    has_bos = _has_bos_up(i)

                    obs.append(OrderBlock(
                        kind="bullish",
                        high=prev["high"],
                        low=mid_low,
                        timestamp=df["timestamp"].iloc[j],
                        mitigated=mitigated,
                        strength=round(rel_vol, 2),
                        has_fvg=has_fvg,
                        has_bos=has_bos,
                        description=(
                            f"Бычий ОБ ${mid_low:.4f}—${prev['high']:.4f} "
                            f"({'митигирован' if mitigated else 'свежий'}), "
                            f"объём ×{rel_vol:.1f}"
                            f"{' +FVG' if has_fvg else ''}"
                            f"{' +BOS' if has_bos else ''}"
                        ),
                    ))
                    break

        # Сильная медвежья свеча → ищем бычий ОБ перед ней
        elif move < -move_threshold:
            for j in range(i-1, max(i-lookback_candles-1, 0), -1):
                prev = df.iloc[j]
                if prev["close"] > prev["open"]:  # бычья
                    mid_high = max(prev["open"], prev["close"])
                    mid_low  = min(prev["open"], prev["close"])

                    future = df.iloc[i+1:]
                    mitigated = not future.empty and (
                        (future["high"] >= mid_low) & (future["low"] <= mid_high)
                    ).any()

                    rel_vol = float(df["quote_volume"].iloc[j]) / float(avg_vol.iloc[j]) \
                        if avg_vol.iloc[j] > 0 else 1.0

                    has_fvg = _has_fvg_after(j, "bearish")
                    has_bos = _has_bos_dn(i)

                    obs.append(OrderBlock(
                        kind="bearish",
                        high=mid_high,
                        low=prev["low"],
                        timestamp=df["timestamp"].iloc[j],
                        mitigated=mitigated,
                        strength=round(rel_vol, 2),
                        has_fvg=has_fvg,
                        has_bos=has_bos,
                        description=(
                            f"Медвежий ОБ ${prev['low']:.4f}—${mid_high:.4f} "
                            f"({'митигирован' if mitigated else 'свежий'}), "
                            f"объём ×{rel_vol:.1f}"
                            f"{' +FVG' if has_fvg else ''}"
                            f"{' +BOS' if has_bos else ''}"
                        ),
                    ))
                    break

    return sorted(obs, key=lambda x: x.timestamp)[-5:]


def find_breaker_blocks(df: pd.DataFrame, obs: list[OrderBlock]) -> list[BreakerBlock]:
    """
    Breaker Block (PDF 3/6): ОБ, который цена пробила насквозь.
    Бывший бычий OB → медвежий брейкер (сопротивление).
    Бывший медвежий OB → бычий брейкер (поддержка).
    """
    breakers = []
    cur = float(df["close"].iloc[-1])
    for ob in obs:
        if not ob.mitigated:
            continue
        # Бычий OB был полностью пробит вниз → медвежий брейкер
        if ob.kind == "bullish" and cur < ob.low:
            breakers.append(BreakerBlock(
                kind="bearish",
                high=ob.high, low=ob.low,
                timestamp=ob.timestamp,
                description=f"Медвежий брейкер ${ob.low:.4f}—${ob.high:.4f} (бывший бычий OB)",
            ))
        # Медвежий OB был полностью пробит вверх → бычий брейкер
        elif ob.kind == "bearish" and cur > ob.high:
            breakers.append(BreakerBlock(
                kind="bullish",
                high=ob.high, low=ob.low,
                timestamp=ob.timestamp,
                description=f"Бычий брейкер ${ob.low:.4f}—${ob.high:.4f} (бывший медвежий OB)",
            ))
    return breakers


def nearest_ob(obs: list[OrderBlock], price: float,
               kind: str = "bullish", max_dist_pct: float = 3.0) -> Optional[OrderBlock]:
    """Ближайший ОБ нужного типа к текущей цене."""
    candidates = [
        ob for ob in obs
        if ob.kind == kind and not ob.mitigated
        and abs((ob.high + ob.low)/2 - price) / price * 100 <= max_dist_pct
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda ob: abs((ob.high + ob.low)/2 - price))


# ═══════════════════════════════════════════════════════════
#  FAIR VALUE GAP (FVG) / ИМБАЛАНС  (PDF 2)
# ═══════════════════════════════════════════════════════════

def find_fvg(df: pd.DataFrame, min_size_pct: float = 0.1) -> list[FVG]:
    """
    FVG = разрыв между свечой[i].high и свечой[i+2].low (бычий)
    или свечой[i].low и свечой[i+2].high (медвежий).
    Триггерные зоны: 50% FVG или полное заполнение.
    """
    fvgs = []
    for i in range(len(df) - 2):
        c0, c2 = df.iloc[i], df.iloc[i+2]
        mid_price = (c0["close"] + c2["close"]) / 2

        # Бычий FVG
        if c2["low"] > c0["high"]:
            size_pct = (c2["low"] - c0["high"]) / mid_price * 100
            if size_pct >= min_size_pct:
                future = df.iloc[i+2:]
                filled = not future.empty and (future["low"] <= c2["low"]).any() and \
                         (future["high"] >= c0["high"]).any()
                fvgs.append(FVG(
                    kind="bullish", high=c2["low"], low=c0["high"],
                    timestamp=df["timestamp"].iloc[i+1],
                    filled=filled, size_pct=round(size_pct, 3),
                ))

        # Медвежий FVG
        elif c2["high"] < c0["low"]:
            size_pct = (c0["low"] - c2["high"]) / mid_price * 100
            if size_pct >= min_size_pct:
                future = df.iloc[i+2:]
                filled = not future.empty and (future["high"] >= c0["low"]).any() and \
                         (future["low"] <= c2["high"]).any()
                fvgs.append(FVG(
                    kind="bearish", high=c0["low"], low=c2["high"],
                    timestamp=df["timestamp"].iloc[i+1],
                    filled=filled, size_pct=round(size_pct, 3),
                ))

    return fvgs[-10:]


def fresh_fvg_near_price(fvgs: list[FVG], price: float,
                         kind: str = "bullish", max_dist_pct: float = 2.0) -> Optional[FVG]:
    candidates = [
        f for f in fvgs
        if f.kind == kind and not f.filled
        and abs((f.high + f.low)/2 - price) / price * 100 <= max_dist_pct
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda f: abs((f.high + f.low)/2 - price))


# ═══════════════════════════════════════════════════════════
#  STOP HUNT / TTS (Turtle Trap Setup)
# ═══════════════════════════════════════════════════════════

def find_stop_hunts(df: pd.DataFrame, lookback: int = 20,
                    min_revert_pct: float = 0.3) -> list[StopHunt]:
    """TTS = ложный пробой уровня + быстрый разворот обратно."""
    hunts = []

    for i in range(lookback, len(df) - 1):
        window = df.iloc[i-lookback:i]
        r_high = window["high"].max()
        r_low  = window["low"].min()

        c   = df.iloc[i]
        nxt = df.iloc[i+1]
        wick_top = c["high"] - max(c["open"], c["close"])
        wick_bot = min(c["open"], c["close"]) - c["low"]
        body     = abs(c["close"] - c["open"])

        if c["high"] > r_high and wick_top > body * 0.5:
            revert = r_high - c["close"]
            if revert / r_high >= min_revert_pct / 100:
                rev_confirmed = nxt["close"] < c["low"]
                hunts.append(StopHunt(
                    kind="bearish",
                    swept_level=r_high,
                    candle_idx=i,
                    timestamp=df["timestamp"].iloc[i],
                    reversal_confirmed=rev_confirmed,
                    description=(
                        f"TTS Медвежий: хвост захватил ликвидность выше ${r_high:.4f}, "
                        f"вернулся к ${c['close']:.4f}. "
                        f"{'✅ Разворот подтверждён' if rev_confirmed else '⏳ Ждём подтверждения'}."
                    ),
                ))

        elif c["low"] < r_low and wick_bot > body * 0.5:
            revert = c["close"] - r_low
            if revert / r_low >= min_revert_pct / 100:
                rev_confirmed = nxt["close"] > c["high"]
                hunts.append(StopHunt(
                    kind="bullish",
                    swept_level=r_low,
                    candle_idx=i,
                    timestamp=df["timestamp"].iloc[i],
                    reversal_confirmed=rev_confirmed,
                    description=(
                        f"TTS Бычий: хвост захватил ликвидность ниже ${r_low:.4f}, "
                        f"вернулся к ${c['close']:.4f}. "
                        f"{'✅ Разворот подтверждён' if rev_confirmed else '⏳ Ждём подтверждения'}."
                    ),
                ))

    return hunts[-5:]


# ═══════════════════════════════════════════════════════════
#  LIQUIDITY ZONES — Равные максимумы/минимумы
# ═══════════════════════════════════════════════════════════

def find_liquidity_zones(df: pd.DataFrame, tol_pct: float = 0.3,
                         min_touches: int = 2) -> list[LiquidityZone]:
    zones = []
    highs = df["high"].values
    lows  = df["low"].values
    ts    = df["timestamp"].values
    cur   = float(df["close"].iloc[-1])

    def cluster(prices, kind):
        visited = set()
        for i, p in enumerate(prices):
            if i in visited:
                continue
            touches = [j for j, q in enumerate(prices)
                       if abs(p - q) / p < tol_pct / 100]
            if len(touches) >= min_touches:
                avg_p = np.mean([prices[j] for j in touches])
                swept = (kind == "equal_highs" and cur > avg_p * 1.002) or \
                        (kind == "equal_lows"  and cur < avg_p * 0.998)
                zones.append(LiquidityZone(
                    kind=kind, price=round(avg_p, 6),
                    touches=len(touches),
                    timestamp=pd.Timestamp(ts[touches[-1]]),
                    swept=swept,
                ))
                visited.update(touches)

    cluster(highs, "equal_highs")
    cluster(lows,  "equal_lows")
    return sorted(zones, key=lambda z: abs(z.price - cur))[:6]


# ═══════════════════════════════════════════════════════════
#  NAMED LIQUIDITY LEVELS  (PDF 2: PMH/PWH/PDH/EQH/EQL)
# ═══════════════════════════════════════════════════════════

def find_named_liquidity_levels(df: pd.DataFrame) -> list[NamedLiquidityLevel]:
    """
    PDH/PDL: максимум/минимум предыдущего дня.
    PWH/PWL: максимум/минимум предыдущей недели (агрегация дневных свечей).
    EQH: равные максимумы (свинг хаи в пределах 0.3% друг от друга).
    EQL: равные минимумы (свинг лоу в пределах 0.3% друг от друга).
    HOD/LOD: максимум/минимум текущего дня.
    """
    if df.empty or len(df) < 2:
        return []

    levels = []
    cur = float(df["close"].iloc[-1])
    df2 = df.copy()
    df2["date"] = df2["timestamp"].dt.date
    days = sorted(df2["date"].unique())

    # PDH / PDL / HOD / LOD
    if len(days) >= 2:
        prev_day = days[-2]
        curr_day = days[-1]
        prev_data = df2[df2["date"] == prev_day]
        curr_data = df2[df2["date"] == curr_day]

        if not prev_data.empty:
            pdh = float(prev_data["high"].max())
            pdl = float(prev_data["low"].min())
            levels.append(NamedLiquidityLevel(
                "PDH", pdh, cur > pdh * 1.001,
                f"PDH — максимум вчерашнего дня (BSL выше ${pdh:.4f})"))
            levels.append(NamedLiquidityLevel(
                "PDL", pdl, cur < pdl * 0.999,
                f"PDL — минимум вчерашнего дня (SSL ниже ${pdl:.4f})"))

        if not curr_data.empty:
            hod = float(curr_data["high"].max())
            lod = float(curr_data["low"].min())
            levels.append(NamedLiquidityLevel(
                "HOD", hod, cur > hod * 1.001,
                f"HOD — максимум текущего дня"))
            levels.append(NamedLiquidityLevel(
                "LOD", lod, cur < lod * 0.999,
                f"LOD — минимум текущего дня"))

    # PWH / PWL (недельные)
    df2["week"] = df2["timestamp"].dt.isocalendar().week
    df2["year"] = df2["timestamp"].dt.year
    weeks = df2[["year", "week"]].drop_duplicates().sort_values(["year", "week"])
    if len(weeks) >= 2:
        pw_row = weeks.iloc[-2]
        pw_data = df2[(df2["year"] == pw_row["year"]) & (df2["week"] == pw_row["week"])]
        if not pw_data.empty:
            pwh = float(pw_data["high"].max())
            pwl = float(pw_data["low"].min())
            levels.append(NamedLiquidityLevel(
                "PWH", pwh, cur > pwh * 1.001,
                f"PWH — максимум прошлой недели (BSL выше ${pwh:.4f})"))
            levels.append(NamedLiquidityLevel(
                "PWL", pwl, cur < pwl * 0.999,
                f"PWL — минимум прошлой недели (SSL ниже ${pwl:.4f})"))

    # EQH / EQL через свинг хаи/лоу (3-candle rule)
    swings = find_swing_points(df, strength=3, lookback=100)
    swing_highs = [s for s in swings if s.kind == "high"]
    swing_lows  = [s for s in swings if s.kind == "low"]

    tol = 0.003  # 0.3%
    used = set()
    for i, sh in enumerate(swing_highs):
        if i in used:
            continue
        cluster = [j for j, s in enumerate(swing_highs)
                   if abs(s.price - sh.price) / sh.price < tol]
        if len(cluster) >= 2:
            avg_p = np.mean([swing_highs[j].price for j in cluster])
            levels.append(NamedLiquidityLevel(
                "EQH", round(avg_p, 6), cur > avg_p * 1.001,
                f"EQH — равные максимумы ${avg_p:.4f} ({len(cluster)} касания, BSL выше)"))
            used.update(cluster)

    used = set()
    for i, sl in enumerate(swing_lows):
        if i in used:
            continue
        cluster = [j for j, s in enumerate(swing_lows)
                   if abs(s.price - sl.price) / sl.price < tol]
        if len(cluster) >= 2:
            avg_p = np.mean([swing_lows[j].price for j in cluster])
            levels.append(NamedLiquidityLevel(
                "EQL", round(avg_p, 6), cur < avg_p * 0.999,
                f"EQL — равные минимумы ${avg_p:.4f} ({len(cluster)} касания, SSL ниже)"))
            used.update(cluster)

    return sorted(levels, key=lambda lv: abs(lv.price - cur))


# ═══════════════════════════════════════════════════════════
#  RSI DIVERGENCE  (PDF 4)
# ═══════════════════════════════════════════════════════════

def _calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    if len(gain) < period:
        return np.full(len(closes), 50.0)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    rs = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50.0
    return rsi


def find_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14,
                        lookback: int = 80) -> list[RSIDivergence]:
    """
    Классическая дивергенция: цена делает новый HH/LL, RSI — нет.
    Скрытая: RSI делает HH/LL, цена — нет (подтверждение тренда).
    Ищем по свинг точкам за последние lookback свечей.
    """
    if len(df) < rsi_period + 20:
        return []

    closes = df["close"].astype(float).values
    rsi_arr = _calc_rsi(closes, rsi_period)
    divs = []

    seg = df.tail(lookback).reset_index(drop=True)
    rsi_seg = rsi_arr[-lookback:]
    offset = len(df) - lookback

    # Свинг хаи/лоу в сегменте (3-candle)
    n = 1
    highs_idx = []
    lows_idx  = []
    for i in range(n, len(seg) - n):
        h_win = seg["high"].iloc[i-n:i+n+1]
        l_win = seg["low"].iloc[i-n:i+n+1]
        if seg["high"].iloc[i] == h_win.max():
            highs_idx.append(i)
        if seg["low"].iloc[i] == l_win.min():
            lows_idx.append(i)

    # Медвежья классическая: цена HH, RSI LH
    for k in range(1, len(highs_idx)):
        i1, i2 = highs_idx[k-1], highs_idx[k]
        p1, p2 = float(seg["high"].iloc[i1]), float(seg["high"].iloc[i2])
        r1, r2 = rsi_seg[i1], rsi_seg[i2]
        if p2 > p1 and r2 < r1 and r1 > 60:  # цена выше, RSI ниже → медвежья
            divs.append(RSIDivergence(
                kind="bearish_classic",
                price1=p1, price2=p2, rsi1=round(r1, 1), rsi2=round(r2, 1),
                idx1=offset+i1, idx2=offset+i2,
                description=f"Медвежья дивергенция: цена ↑ ${p1:.4f}→${p2:.4f}, RSI ↓ {r1:.0f}→{r2:.0f}",
            ))

    # Бычья классическая: цена LL, RSI HL
    for k in range(1, len(lows_idx)):
        i1, i2 = lows_idx[k-1], lows_idx[k]
        p1, p2 = float(seg["low"].iloc[i1]), float(seg["low"].iloc[i2])
        r1, r2 = rsi_seg[i1], rsi_seg[i2]
        if p2 < p1 and r2 > r1 and r1 < 40:  # цена ниже, RSI выше → бычья
            divs.append(RSIDivergence(
                kind="bullish_classic",
                price1=p1, price2=p2, rsi1=round(r1, 1), rsi2=round(r2, 1),
                idx1=offset+i1, idx2=offset+i2,
                description=f"Бычья дивергенция: цена ↓ ${p1:.4f}→${p2:.4f}, RSI ↑ {r1:.0f}→{r2:.0f}",
            ))

    # Скрытая бычья: цена HL, RSI LL (коррекция в восходящем тренде)
    for k in range(1, len(lows_idx)):
        i1, i2 = lows_idx[k-1], lows_idx[k]
        p1, p2 = float(seg["low"].iloc[i1]), float(seg["low"].iloc[i2])
        r1, r2 = rsi_seg[i1], rsi_seg[i2]
        if p2 > p1 and r2 < r1:  # цена выше, RSI ниже → скрытая бычья
            divs.append(RSIDivergence(
                kind="bullish_hidden",
                price1=p1, price2=p2, rsi1=round(r1, 1), rsi2=round(r2, 1),
                idx1=offset+i1, idx2=offset+i2,
                description=f"Скрытая бычья: цена HL ${p1:.4f}→${p2:.4f}, RSI ↓ {r1:.0f}→{r2:.0f}",
            ))

    # Скрытая медвежья: цена LH, RSI HH (коррекция в нисходящем тренде)
    for k in range(1, len(highs_idx)):
        i1, i2 = highs_idx[k-1], highs_idx[k]
        p1, p2 = float(seg["high"].iloc[i1]), float(seg["high"].iloc[i2])
        r1, r2 = rsi_seg[i1], rsi_seg[i2]
        if p2 < p1 and r2 > r1:
            divs.append(RSIDivergence(
                kind="bearish_hidden",
                price1=p1, price2=p2, rsi1=round(r1, 1), rsi2=round(r2, 1),
                idx1=offset+i1, idx2=offset+i2,
                description=f"Скрытая медвежья: цена LH ${p1:.4f}→${p2:.4f}, RSI ↑ {r1:.0f}→{r2:.0f}",
            ))

    return divs[-4:]  # последние 4 дивергенции


# ═══════════════════════════════════════════════════════════
#  KILLZONES & SESSIONS  (PDF 5)
# ═══════════════════════════════════════════════════════════

def get_killzone_info(df: pd.DataFrame) -> KillzoneInfo:
    """
    Killzone по UTC времени:
    Asia Range: 00:00–04:00 UTC (02:00–06:00 Киев UTC+2)
    London KZ:  07:00–11:00 UTC (09:00–13:00 Киев)
    NY KZ:      13:00–16:00 UTC (15:00–18:00 Киев)
    London Close KZ: 18:00–20:00 UTC (20:00–22:00 Киев)
    """
    now_utc = datetime.datetime.utcnow()
    hour = now_utc.hour

    if 0 <= hour < 4:
        zone = "Asia Range (аккумуляция)"
    elif 7 <= hour < 11:
        zone = "London KZ (09:00–13:00 Киев)"
    elif 11 <= hour < 13:
        zone = "London (после KZ)"
    elif 13 <= hour < 16:
        zone = "New York KZ (15:00–18:00 Киев)"
    elif 18 <= hour < 20:
        zone = "London Close KZ (20:00–22:00 Киев)"
    else:
        zone = "Неактивное время"

    # Азиатский рендж (свечи с 00:00 до 04:00 UTC текущего дня)
    today = pd.Timestamp.utcnow().normalize()
    asia_start = today
    asia_end   = today + pd.Timedelta(hours=4)
    df_ts = df.copy()
    df_ts["ts_utc"] = pd.to_datetime(df_ts["timestamp"], utc=True).dt.tz_localize(None) \
        if df_ts["timestamp"].dt.tz is None else pd.to_datetime(df_ts["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    asia_candles = df_ts[(df_ts["ts_utc"] >= asia_start) & (df_ts["ts_utc"] < asia_end)]

    asia_high = float(asia_candles["high"].max()) if not asia_candles.empty else 0.0
    asia_low  = float(asia_candles["low"].min())  if not asia_candles.empty else 0.0

    return KillzoneInfo(
        active_zone=zone,
        asia_range_high=asia_high,
        asia_range_low=asia_low,
        current_utc=now_utc.strftime("%H:%M UTC"),
    )


# ═══════════════════════════════════════════════════════════
#  THREE DRIVES PATTERN  (PDF 7)
# ═══════════════════════════════════════════════════════════

@dataclass
class SMCSetup:
    """Готовый SMC сетап для сканера."""
    name:        str
    kind:        str   # "bullish" | "bearish"
    confidence:  float # 0..1
    description: str
    key_levels:  list  # горизонтальные уровни [entry, stop, target]
    key_points:  list  # [(idx, price), ...] для рисования
    confirmed:   bool  = False
    pattern_shapes: list = field(default_factory=list)  # для TV chart


def detect_three_drives(df: pd.DataFrame, lookback: int = 150) -> list[SMCSetup]:
    """
    Three Drives Pattern (PDF 7): три последовательных экстремума (drive 1, drive 2, drive 3).
    Медвежий: HH1 → pullback A → HH2 → pullback B → HH3 (completion zone).
    Бычий: LL1 → bounce A → LL2 → bounce B → LL3.
    Drive legs должны быть сопоставимы по размеру (±40%).
    Между drives — коррекция не менее 38%.
    """
    results = []
    n = len(df)
    arr = df.tail(lookback).reset_index(drop=True)
    offset = n - lookback

    h = arr["high"].astype(float).values
    l = arr["low"].astype(float).values
    c = arr["close"].astype(float).values

    # swing highs and lows (3-candle)
    sw = 1
    highs_idx, lows_idx = [], []
    for i in range(sw, len(arr) - sw):
        if h[i] == max(h[i-sw:i+sw+1]):
            highs_idx.append(i)
        if l[i] == min(l[i-sw:i+sw+1]):
            lows_idx.append(i)

    def _argmin_seg(a, b):
        seg = l[a:b]; idx = int(np.argmin(seg)) + a if len(seg) else a
        return idx, l[idx]

    def _argmax_seg(a, b):
        seg = h[a:b]; idx = int(np.argmax(seg)) + a if len(seg) else a
        return idx, h[idx]

    def _try_bearish_drives(hi_list):
        for k in range(2, len(hi_list)):
            i3 = hi_list[k]
            if i3 < len(arr) - 30:
                continue
            i1, i2 = hi_list[k-2], hi_list[k-1]
            if not (h[i1] < h[i2] < h[i3]):
                continue
            if i2 <= i1 + 1 or i3 <= i2 + 1: continue
            # Pullback A и B
            pa_idx, low_a = _argmin_seg(i1+1, i2)
            pb_idx, low_b = _argmin_seg(i2+1, i3)
            leg1 = h[i1] - low_a
            leg2 = h[i2] - low_b
            if leg1 <= 0 or leg2 <= 0: continue
            retrace_a = (h[i1] - low_a) / max(h[i2] - low_a, 1e-9)
            if retrace_a < 0.20: continue
            ratio = leg2 / leg1
            if not (0.45 < ratio < 2.2): continue

            conf = 0.65
            if 0.75 < ratio < 1.35: conf += 0.12
            if i3 >= len(arr) - 20: conf += 0.10

            target = low_a
            stop   = h[i3] * 1.005
            # Форма: drive1_high → pullback_A_low → drive2_high → pullback_B_low → drive3_high
            kp = [
                (offset+i1, h[i1]),
                (offset+pa_idx, low_a),
                (offset+i2, h[i2]),
                (offset+pb_idx, low_b),
                (offset+i3, h[i3]),
            ]
            results.append(SMCSetup(
                name="Three Drives ↓",
                kind="bearish",
                confidence=min(conf, 0.90),
                description=f"3 Drives медвежий: пики ${h[i1]:.4f}→${h[i2]:.4f}→${h[i3]:.4f}, цель ${target:.4f}",
                key_levels=[h[i3], stop, target],
                key_points=kp,
                confirmed=(c[-1] < h[i3]),
            ))
        return results

    def _try_bullish_drives(lo_list):
        bull = []
        for k in range(2, len(lo_list)):
            i3 = lo_list[k]
            if i3 < len(arr) - 30:
                continue
            i1, i2 = lo_list[k-2], lo_list[k-1]
            if not (l[i1] > l[i2] > l[i3]):
                continue
            if i2 <= i1 + 1 or i3 <= i2 + 1: continue
            ha_idx, high_a = _argmax_seg(i1+1, i2)
            hb_idx, high_b = _argmax_seg(i2+1, i3)
            leg1 = high_a - l[i1]
            leg2 = high_b - l[i2]
            if leg1 <= 0 or leg2 <= 0: continue
            ratio = leg2 / leg1
            if not (0.45 < ratio < 2.2): continue

            conf = 0.65
            if 0.75 < ratio < 1.35: conf += 0.12
            if i3 >= len(arr) - 20: conf += 0.10

            target = high_a
            stop   = l[i3] * 0.995
            # Форма: drive1_low → bounce_A_high → drive2_low → bounce_B_high → drive3_low
            kp = [
                (offset+i1, l[i1]),
                (offset+ha_idx, high_a),
                (offset+i2, l[i2]),
                (offset+hb_idx, high_b),
                (offset+i3, l[i3]),
            ]
            bull.append(SMCSetup(
                name="Three Drives ↑",
                kind="bullish",
                confidence=min(conf, 0.90),
                description=f"3 Drives бычий: лои ${l[i1]:.4f}→${l[i2]:.4f}→${l[i3]:.4f}, цель ${target:.4f}",
                key_levels=[l[i3], stop, target],
                key_points=kp,
                confirmed=(c[-1] > l[i3]),
            ))
        return bull

    results += _try_bearish_drives(highs_idx)
    results += _try_bullish_drives(lows_idx)
    # Вернуть самый свежий (последний по key_points)
    results.sort(key=lambda x: x.key_points[-1][0] if x.key_points else 0)
    return results[-2:] if results else []


# ═══════════════════════════════════════════════════════════
#  THREE TAP SETUP  (PDF 7)
# ═══════════════════════════════════════════════════════════

def detect_three_tap(df: pd.DataFrame, tol_pct: float = 0.018,
                     lookback: int = 150, recency: int = 20,
                     sw: int = 1) -> list[SMCSetup]:
    """
    Three Tap Setup (PDF 7): цена три раза тестирует одну и ту же зону.
    Третий тест с ослабевающим импульсом = высоковероятный разворот.

    Параметры:
      tol_pct  — допуск «одного уровня» (1.8% по умолчанию)
      lookback — сколько баров анализировать (глобальный vs локальный)
      recency  — третий тап должен быть в последних N барах
      sw       — сила свинга (1=локальный, 2=глобальный)
    """
    results = []
    n = len(df)
    arr = df.tail(lookback).reset_index(drop=True)
    offset = n - lookback

    h   = arr["high"].astype(float).values
    l   = arr["low"].astype(float).values
    c   = arr["close"].astype(float).values
    op  = arr["open"].astype(float).values

    lows_idx, highs_idx = [], []
    for i in range(sw, len(arr) - sw):
        if l[i] == min(l[i-sw:i+sw+1]) and list(l[i-sw:i+sw+1]).count(l[i]) == 1:
            lows_idx.append(i)
        if h[i] == max(h[i-sw:i+sw+1]) and list(h[i-sw:i+sw+1]).count(h[i]) == 1:
            highs_idx.append(i)

    def _body(idx):
        return abs(c[idx] - op[idx])

    def _argmax_in(arr_1d, a, b):
        seg = arr_1d[a:b]
        if len(seg) == 0: return a, arr_1d[a]
        idx = int(np.argmax(seg)) + a
        return idx, arr_1d[idx]

    def _argmin_in(arr_1d, a, b):
        seg = arr_1d[a:b]
        if len(seg) == 0: return a, arr_1d[a]
        idx = int(np.argmin(seg)) + a
        return idx, arr_1d[idx]

    # ── Бычий Three Tap: три лоу в одной зоне поддержки ──
    for k in range(2, len(lows_idx)):
        i3 = lows_idx[k]
        if i3 < len(arr) - recency:      # третий тап слишком старый
            continue
        i1, i2 = lows_idx[k-2], lows_idx[k-1]
        # Тапы должны идти в хронологическом порядке с нужными промежутками
        if not (i1 < i2 < i3) or (i2 - i1) < 2 or (i3 - i2) < 2:
            continue
        p1, p2, p3 = l[i1], l[i2], l[i3]
        ref = (p1 + p2 + p3) / 3
        if ref == 0: continue
        # Все три лоу в пределах tol_pct
        if max(abs(p1-ref)/ref, abs(p2-ref)/ref, abs(p3-ref)/ref) > tol_pct:
            continue
        # Между тапами — ощутимые отскоки ≥ 0.8%
        b12_idx, b12_v = _argmax_in(h, i1+1, i2)
        b23_idx, b23_v = _argmax_in(h, i2+1, i3)
        if (b12_v - ref) / ref < 0.008 or (b23_v - ref) / ref < 0.008:
            continue
        # Ослабление: тело третьего тапа ≤ тела первого
        weakening = _body(i3) <= _body(i1) * 1.3

        zone_high = max(p1, p2, p3) * 1.002
        zone_low  = min(p1, p2, p3) * 0.998
        # Цель = ширина отскока от зоны
        avg_bounce = ((b12_v - ref) + (b23_v - ref)) / 2
        target = ref + avg_bounce * 2
        stop   = zone_low * 0.995

        conf = 0.70
        if weakening: conf += 0.10
        # Чем равнее лои — тем выше уверенность
        spread = (max(p1,p2,p3) - min(p1,p2,p3)) / ref
        if spread < 0.005: conf += 0.10
        # Третий тап в последних 10 барах — подтверждённый сетап
        very_fresh = (i3 >= len(arr) - 10)
        if very_fresh: conf += 0.05

        # ── Гребёнка: показываем каждый тап как спуск к зоне ──
        # Форма: tap1_low → bounce12 → tap2_low → bounce23 → tap3_low
        kp = [
            (offset + i1, p1),
            (offset + b12_idx, b12_v),
            (offset + i2, p2),
            (offset + b23_idx, b23_v),
            (offset + i3, p3),
        ]

        results.append(SMCSetup(
            name="Three Tap ↑",
            kind="bullish",
            confidence=min(conf, 0.92),
            description=(
                f"Three Tap поддержка ({sw=}): "
                f"${p1:.4f}/{p2:.4f}/{p3:.4f} ≈ ${ref:.4f}, "
                f"bounce {(b12_v-ref)/ref*100:.1f}%/{(b23_v-ref)/ref*100:.1f}%, "
                f"цель ${target:.4f}"
            ),
            key_levels=[zone_high, stop, target],
            key_points=kp,
            confirmed=very_fresh and (c[-1] > zone_high),
        ))

    # ── Медвежий Three Tap: три хая в одной зоне сопротивления ──
    for k in range(2, len(highs_idx)):
        i3 = highs_idx[k]
        if i3 < len(arr) - recency:
            continue
        i1, i2 = highs_idx[k-2], highs_idx[k-1]
        if not (i1 < i2 < i3) or (i2 - i1) < 2 or (i3 - i2) < 2:
            continue
        p1, p2, p3 = h[i1], h[i2], h[i3]
        ref = (p1 + p2 + p3) / 3
        if ref == 0: continue
        if max(abs(p1-ref)/ref, abs(p2-ref)/ref, abs(p3-ref)/ref) > tol_pct:
            continue
        d12_idx, d12_v = _argmin_in(l, i1+1, i2)
        d23_idx, d23_v = _argmin_in(l, i2+1, i3)
        if (ref - d12_v) / ref < 0.008 or (ref - d23_v) / ref < 0.008:
            continue
        weakening = _body(i3) <= _body(i1) * 1.3

        zone_high = max(p1, p2, p3) * 1.002
        zone_low  = min(p1, p2, p3) * 0.998
        avg_drop  = ((ref - d12_v) + (ref - d23_v)) / 2
        target    = ref - avg_drop * 2
        stop      = zone_high * 1.005

        conf = 0.70
        if weakening: conf += 0.10
        spread = (max(p1,p2,p3) - min(p1,p2,p3)) / ref
        if spread < 0.005: conf += 0.10
        very_fresh = (i3 >= len(arr) - 10)
        if very_fresh: conf += 0.05

        # Гребёнка: tap1_high → dip12 → tap2_high → dip23 → tap3_high
        kp = [
            (offset + i1, p1),
            (offset + d12_idx, d12_v),
            (offset + i2, p2),
            (offset + d23_idx, d23_v),
            (offset + i3, p3),
        ]

        results.append(SMCSetup(
            name="Three Tap ↓",
            kind="bearish",
            confidence=min(conf, 0.92),
            description=(
                f"Three Tap сопротивление ({sw=}): "
                f"${p1:.4f}/{p2:.4f}/{p3:.4f} ≈ ${ref:.4f}, "
                f"drop {(ref-d12_v)/ref*100:.1f}%/{(ref-d23_v)/ref*100:.1f}%, "
                f"цель ${target:.4f}"
            ),
            key_levels=[zone_low, stop, target],
            key_points=kp,
            confirmed=very_fresh and (c[-1] < zone_low),
        ))

    results.sort(key=lambda x: x.key_points[-1][0] if x.key_points else 0)
    return results[-3:] if results else []


def _tf_tap_params(tf: str) -> dict:
    """Параметры Three Tap по таймфрейму: lookback, recency, sw."""
    tf = (tf or "1h").lower()
    if tf in ("1m", "3m", "5m"):
        return dict(lookback=120, recency=15, sw=1)
    elif tf in ("15m", "30m"):
        return dict(lookback=150, recency=20, sw=1)
    elif tf in ("1h",):
        return dict(lookback=200, recency=25, sw=2)
    elif tf in ("4h",):
        return dict(lookback=200, recency=20, sw=2)
    elif tf in ("1d", "3d"):
        return dict(lookback=300, recency=15, sw=3)
    else:
        return dict(lookback=150, recency=20, sw=1)


# ═══════════════════════════════════════════════════════════
#  RANGE + DEVIATION  (PDF 6)
# ═══════════════════════════════════════════════════════════

def detect_range_deviation(df: pd.DataFrame, min_range_bars: int = 15,
                           lookback: int = 100) -> list[SMCSetup]:
    """
    Range + Deviation (PDF 6): консолидация → ложный пробой (девиация) → возврат внутрь → движение.
    Bullish: девиация вниз (sweep SSL) → цена возвращается выше range_low.
    Bearish: девиация вверх (sweep BSL) → цена падает ниже range_high.
    """
    results = []
    n = len(df)
    arr = df.tail(lookback).reset_index(drop=True)
    offset = n - lookback

    h = arr["high"].astype(float).values
    l = arr["low"].astype(float).values
    c = arr["close"].astype(float).values

    # Ищем рендж: 15+ баров где high/low стабильны (ATR мал)
    for start in range(0, len(arr) - min_range_bars - 5):
        seg_h = h[start:start + min_range_bars]
        seg_l = l[start:start + min_range_bars]
        rng_high = float(np.max(seg_h))
        rng_low  = float(np.min(seg_l))
        rng_size = rng_high - rng_low
        if rng_size <= 0: continue
        mid = (rng_high + rng_low) / 2
        # Рендж узкий относительно последующего движения
        atr_range = float(np.mean(seg_h - seg_l))
        if rng_size > atr_range * 4:  # слишком широкий — не рендж
            continue

        end_range = start + min_range_bars
        if end_range >= len(arr) - 3:
            continue

        after_h = h[end_range:]
        after_l = l[end_range:]
        after_c = c[end_range:]
        if len(after_h) == 0:
            continue

        # Bullish devitation: цена пробила вниз (< rng_low) и закрылась обратно выше
        for j in range(len(after_l) - 1):
            if after_l[j] < rng_low * 0.998 and after_c[j] > rng_low:
                dev_idx = end_range + j
                target  = rng_high
                stop    = after_l[j] * 0.995
                conf    = 0.72
                if after_c[-1] > rng_low: conf += 0.08  # последняя свеча выше рэнджа
                results.append(SMCSetup(
                    name="Deviation ↑",
                    kind="bullish",
                    confidence=min(conf, 0.88),
                    description=f"Range deviation бычий: рендж ${rng_low:.4f}—${rng_high:.4f}, sweep до ${after_l[j]:.4f}, цель ${target:.4f}",
                    key_levels=[rng_low, stop, target],
                    key_points=[(offset+start, rng_high), (offset+end_range-1, rng_low),
                                (offset+dev_idx, after_l[j])],
                    confirmed=(after_c[-1] > rng_low),
                ))
                break  # один сетап на рендж

        # Bearish deviation: пробой вверх и возврат ниже
        for j in range(len(after_h) - 1):
            if after_h[j] > rng_high * 1.002 and after_c[j] < rng_high:
                dev_idx = end_range + j
                target  = rng_low
                stop    = after_h[j] * 1.005
                conf    = 0.72
                if after_c[-1] < rng_high: conf += 0.08
                results.append(SMCSetup(
                    name="Deviation ↓",
                    kind="bearish",
                    confidence=min(conf, 0.88),
                    description=f"Range deviation медвежий: рендж ${rng_low:.4f}—${rng_high:.4f}, sweep до ${after_h[j]:.4f}, цель ${target:.4f}",
                    key_levels=[rng_high, stop, target],
                    key_points=[(offset+start, rng_low), (offset+end_range-1, rng_high),
                                (offset+dev_idx, after_h[j])],
                    confirmed=(after_c[-1] < rng_high),
                ))
                break

    # Вернуть только свежие (последний по key_points)
    results.sort(key=lambda x: x.key_points[-1][0] if x.key_points else 0)
    return results[-2:] if results else []


# ═══════════════════════════════════════════════════════════
#  QUALITY SMC SETUP  (PDF 7: OB+FVG+BOS+liquidity sweep)
# ═══════════════════════════════════════════════════════════

def detect_quality_smc_setup(df: pd.DataFrame) -> list[SMCSetup]:
    """
    Качественный SMC сетап (PDF 7 критерии):
    Лонг: структура восходящая + sweep SSL + бычий OB с FVG + BOS подтверждён + цена в дисконте.
    Шорт: структура нисходящая + sweep BSL + медвежий OB с FVG + BOS подтверждён + цена в премиуме.
    """
    results = []
    cur = float(df["close"].iloc[-1])
    obs = find_order_blocks(df)
    hunts = find_stop_hunts(df)
    swings = find_swing_points(df, strength=3, lookback=80)

    if not obs:
        return results

    # Структура рынка: последние 3 свинг-хая/лоя
    highs = [s for s in swings if s.kind == "high"]
    lows  = [s for s in swings if s.kind == "low"]

    # Восходящий тренд: каждый следующий хай и лоу выше предыдущего
    bull_structure = (
        len(highs) >= 2 and len(lows) >= 2 and
        highs[-1].price > highs[-2].price and
        lows[-1].price  > lows[-2].price
    )
    bear_structure = (
        len(highs) >= 2 and len(lows) >= 2 and
        highs[-1].price < highs[-2].price and
        lows[-1].price  < lows[-2].price
    )

    # Недавний Stop Hunt
    recent_bull_hunt = any(
        sh.kind == "bullish" and sh.reversal_confirmed
        for sh in hunts
    )
    recent_bear_hunt = any(
        sh.kind == "bearish" and sh.reversal_confirmed
        for sh in hunts
    )

    for ob in obs:
        ob_mid = (ob.high + ob.low) / 2
        dist_pct = abs(cur - ob_mid) / cur * 100

        # ── Качественный бычий сетап ──
        if ob.kind == "bullish" and ob.has_fvg and ob.has_bos and not ob.mitigated:
            if dist_pct > 3.0: continue
            # Цена в дисконте: ниже 50% от последнего свинг-диапазона
            in_discount = (len(highs) > 0 and len(lows) > 0 and
                           cur < (highs[-1].price + lows[-1].price) / 2)
            conf = 0.65
            if bull_structure: conf += 0.12
            if recent_bull_hunt: conf += 0.10
            if in_discount: conf += 0.08
            if ob.strength > 1.5: conf += 0.05

            target = highs[-1].price if highs else cur * 1.05
            stop   = ob.low * 0.995
            results.append(SMCSetup(
                name="SMC Quality ↑",
                kind="bullish",
                confidence=min(conf, 0.95),
                description=(
                    f"Качественный лонг: бычий OB ${ob.low:.4f}—${ob.high:.4f} +FVG+BOS"
                    f"{', sweep SSL' if recent_bull_hunt else ''}"
                    f"{', bull structure' if bull_structure else ''}"
                    f", dist {dist_pct:.1f}%"
                ),
                key_levels=[ob_mid, stop, target],
                key_points=[(df.index[-1], ob.low), (df.index[-1], ob.high)],
                confirmed=bull_structure and recent_bull_hunt,
            ))

        # ── Качественный медвежий сетап ──
        elif ob.kind == "bearish" and ob.has_fvg and ob.has_bos and not ob.mitigated:
            if dist_pct > 3.0: continue
            in_premium = (len(highs) > 0 and len(lows) > 0 and
                          cur > (highs[-1].price + lows[-1].price) / 2)
            conf = 0.65
            if bear_structure: conf += 0.12
            if recent_bear_hunt: conf += 0.10
            if in_premium: conf += 0.08
            if ob.strength > 1.5: conf += 0.05

            target = lows[-1].price if lows else cur * 0.95
            stop   = ob.high * 1.005
            results.append(SMCSetup(
                name="SMC Quality ↓",
                kind="bearish",
                confidence=min(conf, 0.95),
                description=(
                    f"Качественный шорт: медвежий OB ${ob.low:.4f}—${ob.high:.4f} +FVG+BOS"
                    f"{', sweep BSL' if recent_bear_hunt else ''}"
                    f"{', bear structure' if bear_structure else ''}"
                    f", dist {dist_pct:.1f}%"
                ),
                key_levels=[ob_mid, stop, target],
                key_points=[(df.index[-1], ob.low), (df.index[-1], ob.high)],
                confirmed=bear_structure and recent_bear_hunt,
            ))

    return results


# ═══════════════════════════════════════════════════════════
#  BREAKER BLOCK SETUP  (PDF 3/6)
# ═══════════════════════════════════════════════════════════

def detect_breaker_setup(df: pd.DataFrame) -> list[SMCSetup]:
    """
    Breaker Block сетап: цена возвращается в BB зону → контртрендовый вход.
    Бычий BB: бывший медвежий OB → теперь поддержка, цена ретестирует.
    """
    results = []
    cur = float(df["close"].iloc[-1])
    obs = find_order_blocks(df)
    bbs = find_breaker_blocks(df, obs)

    for bb in bbs:
        bb_mid = (bb.high + bb.low) / 2
        dist_pct = abs(cur - bb_mid) / cur * 100
        if dist_pct > 2.5: continue

        if bb.kind == "bullish":
            target = bb.high * 1.03
            stop   = bb.low  * 0.995
            conf   = 0.72 + (0.10 if dist_pct < 1.0 else 0)
            results.append(SMCSetup(
                name="Breaker Block ↑",
                kind="bullish",
                confidence=min(conf, 0.88),
                description=f"Бычий Breaker Block ${bb.low:.4f}—${bb.high:.4f}, ретест, цель ${target:.4f}",
                key_levels=[bb_mid, stop, target],
                key_points=[(df.index[-1], bb.low), (df.index[-1], bb.high)],
                confirmed=(cur > bb.low and cur < bb.high),
            ))
        elif bb.kind == "bearish":
            target = bb.low  * 0.97
            stop   = bb.high * 1.005
            conf   = 0.72 + (0.10 if dist_pct < 1.0 else 0)
            results.append(SMCSetup(
                name="Breaker Block ↓",
                kind="bearish",
                confidence=min(conf, 0.88),
                description=f"Медвежий Breaker Block ${bb.low:.4f}—${bb.high:.4f}, ретест, цель ${target:.4f}",
                key_levels=[bb_mid, stop, target],
                key_points=[(df.index[-1], bb.low), (df.index[-1], bb.high)],
                confirmed=(cur > bb.low and cur < bb.high),
            ))

    return results


# ═══════════════════════════════════════════════════════════
#  MASTER SMC SCANNER
# ═══════════════════════════════════════════════════════════

def scan_smc_setups(df: pd.DataFrame, setup_types: list[str] | None = None,
                    tf: str = "1h") -> list[SMCSetup]:
    """
    Главный сканер SMC сетапов.
    setup_types: "quality_ob", "breaker", "three_drives", "three_tap", "deviation"
    tf: таймфрейм (влияет на lookback и recency в Three Tap / Three Drives)
    """
    if setup_types is None:
        setup_types = ["quality_ob", "breaker", "three_drives", "three_tap", "deviation"]

    results = []
    tap_p = _tf_tap_params(tf)

    try:
        if "quality_ob"   in setup_types:
            results += detect_quality_smc_setup(df)
        if "breaker"      in setup_types:
            results += detect_breaker_setup(df)
        if "three_drives" in setup_types:
            # Локальный + глобальный поиск
            results += detect_three_drives(df, lookback=tap_p["lookback"])
            if tap_p["sw"] > 1:
                results += detect_three_drives(df, lookback=min(len(df)-1, tap_p["lookback"]+100))
        if "three_tap" in setup_types:
            # Локальный поиск (sw=1, tight recency)
            results += detect_three_tap(df, sw=1,
                                        lookback=tap_p["lookback"],
                                        recency=tap_p["recency"])
            # Глобальный поиск (sw=2, шире, для более крупных зон)
            if len(df) >= 80:
                results += detect_three_tap(df, sw=2,
                                            lookback=min(len(df)-1, tap_p["lookback"]+100),
                                            recency=tap_p["recency"]+10,
                                            tol_pct=0.025)
        if "deviation"    in setup_types:
            results += detect_range_deviation(df)
    except Exception:
        pass

    # Дедупликация: убираем дубли с одинаковым name+direction+last_kp (±5 баров)
    seen = []
    deduped = []
    for r in results:
        sig = (r.name, r.kind,
               round(r.key_points[-1][0] / 5) if r.key_points else 0)
        if sig not in seen:
            seen.append(sig)
            deduped.append(r)
    return deduped


# ═══════════════════════════════════════════════════════════
#  TDP — True Daily Profile
# ═══════════════════════════════════════════════════════════

def get_tdp_levels(df_1h: pd.DataFrame) -> list[TDPLevel]:
    """PDH, PDL, PDC, открытие текущего дня."""
    if df_1h.empty or len(df_1h) < 25:
        return []

    df = df_1h.copy()
    df["date"] = df["timestamp"].dt.date
    days = sorted(df["date"].unique())

    if len(days) < 2:
        return []

    prev_day = days[-2]
    curr_day = days[-1]

    prev_data = df[df["date"] == prev_day]
    curr_data = df[df["date"] == curr_day]

    if prev_data.empty or curr_data.empty:
        return []

    return [
        TDPLevel("prev_high",  float(prev_data["high"].max()),
                 "PDH — максимум предыдущего дня"),
        TDPLevel("prev_low",   float(prev_data["low"].min()),
                 "PDL — минимум предыдущего дня"),
        TDPLevel("prev_close", float(prev_data["close"].iloc[-1]),
                 "PDC — закрытие предыдущего дня"),
        TDPLevel("open",       float(curr_data["open"].iloc[0]),
                 "Открытие текущего дня"),
    ]
