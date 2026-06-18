"""
Поиск 18 классических графических паттернов + свечные + гармоники.
─────────────────────────────────────────────────────────────────
Медвежьи разворотные : Double Top, Triple Top, H&S, Rising Wedge,
                       Rounded Top, Inverted Cup&Handle, Diamond Top,
                       Bearish Broadening Triangle, Trend Change ↓
Бычьи разворотные   : Double Bottom, Triple Bottom, Inverse H&S, Falling Wedge,
                       Rounded Bottom, Cup&Handle, Cup (no handle),
                       Diamond Bottom, Bullish Broadening Triangle, Trend Change ↑
Продолжение         : Ascending/Descending/Symmetrical Triangle
Свечные             : Bullish/Bearish Engulfing, Hammer/Pin Bar
Гармонические       : Gartley, Butterfly, Bat, Crab, Shark
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatternResult:
    name:        str
    found:       bool
    direction:   str          # "bullish" | "bearish" | "neutral"
    confidence:  float        # 0.0 – 1.0
    confirmed:   bool         # пробой уровня подтверждён
    key_levels:  list = field(default_factory=list)
    key_points:  list = field(default_factory=list)  # [{price, idx}] abs index in df
    description: str  = ""
    candle_idx:  int  = -1


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _pivots(s: np.ndarray, order: int = 5):
    peaks, troughs = [], []
    n = len(s)
    for i in range(order, n - order):
        win = s[i - order: i + order + 1]
        if s[i] >= win.max(): peaks.append(i)
        if s[i] <= win.min(): troughs.append(i)
    return peaks, troughs


def _linreg(xs, ys):
    if len(xs) < 2:
        return 0.0, float(ys[0]) if ys else 0.0
    m, b = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return float(m), float(b)


# ──────────────────────────────────────────────
#  1. DOUBLE TOP / BOTTOM
# ──────────────────────────────────────────────

def _prior_uptrend(arr, pivot_idx, lookback=20, min_rise=0.04) -> bool:
    """Цена перед pivot выросла минимум на min_rise от своего лоу."""
    start = max(0, pivot_idx - lookback)
    seg = arr[start:pivot_idx + 1]
    return (seg[-1] - seg.min()) / (seg.min() + 1e-10) >= min_rise


def _prior_downtrend(arr, pivot_idx, lookback=20, min_fall=0.04) -> bool:
    """Цена перед pivot упала минимум на min_fall от своего хая."""
    start = max(0, pivot_idx - lookback)
    seg = arr[start:pivot_idx + 1]
    return (seg.max() - seg[-1]) / (seg.max() + 1e-10) >= min_fall


def detect_double_top(df: pd.DataFrame, window=100, tol=0.035) -> Optional[PatternResult]:
    """
    Double Top (M-pattern):
    - 2 свинг хая на одном уровне ±tol, между ними отскок вниз ≥2%
    - Перед первым хаем восходящий тренд ≥4%
    - Второй хай в последних 25 барах
    """
    n = min(len(df), window)
    h = df["high"].astype(float).values[-n:]
    l = df["low"].astype(float).values[-n:]
    c = df["close"].astype(float).values
    off = len(df) - n

    order = 3
    peaks = [i for i in range(order, n - order)
             if h[i] == max(h[i-order:i+order+1]) and h[i] > h[i-1] and h[i] > h[i+1]]

    for idx2 in range(len(peaks) - 1, 0, -1):
        p2 = peaks[idx2]
        if p2 < n - 25:
            break
        for idx1 in range(idx2 - 1, -1, -1):
            p1 = peaks[idx1]
            # Два пика на одном уровне
            if abs(h[p1] - h[p2]) / (h[p1] + 1e-10) > tol: continue
            # Второй пик не выше первого
            if h[p2] > h[p1] * 1.005: continue
            # Отскок вниз между пиками ≥2%
            valley_l = float(np.min(l[p1:p2+1]))
            pullback = (h[p1] - valley_l) / h[p1]
            if pullback < 0.02: continue
            # Восходящий тренд перед первым пиком
            if not _prior_uptrend(h, p1, lookback=max(p1, 10), min_rise=0.04): continue

            seg = l[p1+1:p2] if p2 > p1+1 else l[p1:p2+1]
            if len(seg) == 0: continue
            neck_rel = p1 + 1 + int(np.argmin(seg))
            neck = float(l[neck_rel])
            height = float(h[p1]) - neck
            target = neck - height
            confirmed = float(c[-1]) < neck
            conf = min(0.65 + 0.20 * confirmed + 0.10 * (1 - abs(h[p1]-h[p2])/(h[p1]*tol+1e-10)), 0.95)
            kpts = [
                {"price": float(h[p1]), "idx": off + p1},
                {"price": neck,          "idx": off + neck_rel},
                {"price": float(h[p2]), "idx": off + p2},
                {"price": neck,          "idx": len(df) - 1},
            ]
            return PatternResult(
                "Double Top", True, "bearish", round(conf, 2), confirmed,
                [float(h[p1]), float(h[p2]), neck],
                kpts,
                f"Double Top. Neck={neck:.4f}. Target={target:.4f}. "
                f"{'Confirmed' if confirmed else 'Wait breakdown'}"
            )
    return None


def detect_double_bottom(df: pd.DataFrame, window=100, tol=0.035) -> Optional[PatternResult]:
    """
    Double Bottom (W-pattern):
    - 2 свинг лоу на одном уровне ±tol, между ними отскок вверх ≥2%
    - Перед первым лоем нисходящий тренд ≥4%
    - Второй лой в последних 25 барах
    """
    n = min(len(df), window)
    l = df["low"].astype(float).values[-n:]
    h = df["high"].astype(float).values[-n:]
    c = df["close"].astype(float).values
    off = len(df) - n

    order = 3
    troughs = [i for i in range(order, n - order)
               if l[i] == min(l[i-order:i+order+1]) and l[i] < l[i-1] and l[i] < l[i+1]]

    for idx2 in range(len(troughs) - 1, 0, -1):
        t2 = troughs[idx2]
        if t2 < n - 25:
            break
        for idx1 in range(idx2 - 1, -1, -1):
            t1 = troughs[idx1]
            # Два дна на одном уровне
            if abs(l[t1] - l[t2]) / (l[t1] + 1e-10) > tol: continue
            # Второе дно не ниже первого
            if l[t2] < l[t1] * (1 - 0.005): continue
            # Отскок между дна ≥2%
            peak_h = float(np.max(h[t1:t2+1]))
            bounce = (peak_h - l[t1]) / (l[t1] + 1e-10)
            if bounce < 0.02: continue
            # Нисходящий тренд перед первым дном
            if not _prior_downtrend(l, t1, lookback=max(t1, 10), min_fall=0.04): continue

            seg = h[t1+1:t2] if t2 > t1+1 else h[t1:t2+1]
            if len(seg) == 0: continue
            neck_rel = t1 + 1 + int(np.argmax(seg))
            neck = float(h[neck_rel])
            height = neck - float(l[t1])
            target = neck + height
            confirmed = float(c[-1]) > neck
            conf = min(0.65 + 0.20 * confirmed + 0.10 * (1 - abs(l[t1]-l[t2])/(l[t1]*tol+1e-10)), 0.95)
            kpts = [
                {"price": float(l[t1]), "idx": off + t1},
                {"price": neck,          "idx": off + neck_rel},
                {"price": float(l[t2]), "idx": off + t2},
                {"price": neck,          "idx": len(df) - 1},
            ]
            return PatternResult(
                "Double Bottom", True, "bullish", round(conf, 2), confirmed,
                [float(l[t1]), float(l[t2]), neck],
                kpts,
                f"Double Bottom. Neck={neck:.4f}. Target={target:.4f}. "
                f"{'Confirmed' if confirmed else 'Wait breakout'}"
            )
    return None


# ──────────────────────────────────────────────
#  2. TRIPLE TOP / BOTTOM
# ──────────────────────────────────────────────

def detect_triple_top(df: pd.DataFrame, window=150, tol=0.04) -> Optional[PatternResult]:
    """
    Triple Top (TradingView-style):
    - 3 свинг хая на одном уровне (±tol), между ними 2 отскока вниз (≥2%)
    - Перед первым хаем восходящий тренд ≥5%
    - Третий хай должен быть свежим (в последних 30% окна)
    - Нечлайн = минимум из двух промежуточных лоу
    - Подтверждение = закрытие ниже нечлайна
    """
    n = min(len(df), window)
    h = df["high"].astype(float).values[-n:]
    l = df["low"].astype(float).values[-n:]
    c = df["close"].astype(float).values
    off = len(df) - n

    # Свинг хаи с 3-свечным правилом (order=3 для чувствительности)
    order = 3
    peaks = [i for i in range(order, n - order)
             if h[i] == max(h[i-order:i+order+1]) and h[i] > h[i-1] and h[i] > h[i+1]]

    # Нужно хотя бы 3 пика
    for idx3 in range(len(peaks) - 1, 1, -1):
        p3 = peaks[idx3]
        # p3 должен быть в последних 30 барах
        if p3 < n - 30:
            break
        for idx2 in range(idx3 - 1, 0, -1):
            p2 = peaks[idx2]
            # Между p2 и p3 должен быть значимый провал
            valley23_low = float(np.min(l[p2:p3+1]))
            bounce23 = (h[p2] - valley23_low) / (h[p2] + 1e-10)
            if bounce23 < 0.02: continue
            for idx1 in range(idx2 - 1, -1, -1):
                p1 = peaks[idx1]
                # Между p1 и p2 должен быть значимый провал
                valley12_low = float(np.min(l[p1:p2+1]))
                bounce12 = (h[p1] - valley12_low) / (h[p1] + 1e-10)
                if bounce12 < 0.02: continue

                # Три пика на одном уровне ±tol
                avg_h = (h[p1] + h[p2] + h[p3]) / 3
                if any(abs(h[p] - avg_h) / (avg_h + 1e-10) > tol for p in [p1, p2, p3]): continue

                # Восходящий тренд перед первым пиком
                if not _prior_uptrend(h, p1, lookback=max(p1, 10), min_rise=0.05): continue

                # Нечлайн = минимум из двух промежуточных долин (только МЕЖДУ хаями)
                seg12 = l[p1+1:p2] if p2 > p1+1 else l[p1:p2+1]
                seg23 = l[p2+1:p3] if p3 > p2+1 else l[p2:p3+1]
                if len(seg12) == 0 or len(seg23) == 0: continue
                valley12_idx = p1 + 1 + int(np.argmin(seg12))
                valley23_idx = p2 + 1 + int(np.argmin(seg23))
                neck12 = float(l[valley12_idx])
                neck23 = float(l[valley23_idx])
                neck = min(neck12, neck23)
                height = avg_h - neck
                target = neck - height

                confirmed = float(c[-1]) < neck
                conf = min(0.72 + 0.18 * confirmed + 0.05 * (bounce12 + bounce23), 0.94)

                kpts = [
                    {"price": float(h[p1]),  "idx": off + p1},
                    {"price": neck12,         "idx": off + valley12_idx},
                    {"price": float(h[p2]),  "idx": off + p2},
                    {"price": neck23,         "idx": off + valley23_idx},
                    {"price": float(h[p3]),  "idx": off + p3},
                    {"price": neck,           "idx": len(df) - 1},
                ]
                return PatternResult(
                    "Triple Top", True, "bearish", round(conf, 2), confirmed,
                    [float(avg_h), neck, target],
                    kpts,
                    f"Triple Top. Neck={neck:.4f}. Target={target:.4f}. "
                    f"{'Confirmed' if confirmed else 'Wait breakout'}"
                )
    return None


def detect_triple_bottom(df: pd.DataFrame, window=150, tol=0.04) -> Optional[PatternResult]:
    """
    Triple Bottom (TradingView-style):
    - 3 свинг лоу на одном уровне (±tol), между ними 2 отскока вверх (≥2%)
    - Перед первым лоу нисходящий тренд ≥5%
    - Третий лой должен быть свежим (в последних 30 барах)
    - Нечлайн = максимум из двух промежуточных хаев
    - Подтверждение = закрытие выше нечлайна
    """
    n = min(len(df), window)
    l = df["low"].astype(float).values[-n:]
    h = df["high"].astype(float).values[-n:]
    c = df["close"].astype(float).values
    off = len(df) - n

    order = 3
    troughs = [i for i in range(order, n - order)
               if l[i] == min(l[i-order:i+order+1]) and l[i] < l[i-1] and l[i] < l[i+1]]

    for idx3 in range(len(troughs) - 1, 1, -1):
        t3 = troughs[idx3]
        # t3 должен быть в последних 30 барах
        if t3 < n - 30:
            break
        for idx2 in range(idx3 - 1, 0, -1):
            t2 = troughs[idx2]
            # Между t2 и t3 должен быть значимый отскок
            peak23_high = float(np.max(h[t2:t3+1]))
            bounce23 = (peak23_high - l[t2]) / (l[t2] + 1e-10)
            if bounce23 < 0.02: continue
            for idx1 in range(idx2 - 1, -1, -1):
                t1 = troughs[idx1]
                # Между t1 и t2 должен быть значимый отскок
                peak12_high = float(np.max(h[t1:t2+1]))
                bounce12 = (peak12_high - l[t1]) / (l[t1] + 1e-10)
                if bounce12 < 0.02: continue

                # Три лоу на одном уровне ±tol
                avg_l = (l[t1] + l[t2] + l[t3]) / 3
                if any(abs(l[t] - avg_l) / (avg_l + 1e-10) > tol for t in [t1, t2, t3]): continue

                # Нисходящий тренд перед первым лоем
                if not _prior_downtrend(l, t1, lookback=max(t1, 10), min_fall=0.05): continue

                # Нечлайн = максимум из двух промежуточных отскоков (только МЕЖДУ лоями)
                seg12 = h[t1+1:t2] if t2 > t1+1 else h[t1:t2+1]
                seg23 = h[t2+1:t3] if t3 > t2+1 else h[t2:t3+1]
                if len(seg12) == 0 or len(seg23) == 0: continue
                peak12_idx = t1 + 1 + int(np.argmax(seg12))
                peak23_idx = t2 + 1 + int(np.argmax(seg23))
                neck12 = float(h[peak12_idx])
                neck23 = float(h[peak23_idx])
                neck = max(neck12, neck23)
                height = neck - avg_l
                target = neck + height

                confirmed = float(c[-1]) > neck
                conf = min(0.72 + 0.18 * confirmed + 0.05 * (bounce12 + bounce23), 0.94)

                kpts = [
                    {"price": float(l[t1]), "idx": off + t1},
                    {"price": neck12,        "idx": off + peak12_idx},
                    {"price": float(l[t2]), "idx": off + t2},
                    {"price": neck23,        "idx": off + peak23_idx},
                    {"price": float(l[t3]), "idx": off + t3},
                    {"price": neck,          "idx": len(df) - 1},
                ]
                return PatternResult(
                    "Triple Bottom", True, "bullish", round(conf, 2), confirmed,
                    [float(avg_l), neck, target],
                    kpts,
                    f"Triple Bottom. Neck={neck:.4f}. Target={target:.4f}. "
                    f"{'Confirmed' if confirmed else 'Wait breakout'}"
                )
    return None


# ──────────────────────────────────────────────
#  3. HEAD & SHOULDERS (прямая + перевёрнутая)
# ──────────────────────────────────────────────

def detect_head_shoulders(df: pd.DataFrame, window=100) -> Optional[PatternResult]:
    h = df["high"].values[-window:]
    l = df["low"].values[-window:]
    c = df["close"].values
    peaks, _ = _pivots(h, order=5)
    if len(peaks) < 3: return None
    ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]
    if not (h[hd] > h[ls] and h[hd] > h[rs]): return None
    if abs(h[ls] - h[rs]) / (h[hd] + 1e-10) > 0.07: return None
    neck = float(np.min(l[ls:rs + 1]))
    confirmed = float(c[-1]) < neck
    height = float(h[hd]) - neck
    conf = min(0.70 + 0.15 * confirmed, 0.93)
    # Правило: перед левым плечом восходящий тренд
    if not _prior_uptrend(h, ls, lookback=max(ls, 15), min_rise=0.04): return None
    off = len(df) - window
    n1 = off + ls + int(np.argmin(l[ls:hd+1]))
    n2 = off + hd + int(np.argmin(l[hd:rs+1]))
    kpts = [{"price": float(h[ls]), "idx": off+ls}, {"price": neck, "idx": n1},
            {"price": float(h[hd]), "idx": off+hd}, {"price": neck, "idx": n2},
            {"price": float(h[rs]), "idx": off+rs}, {"price": neck, "idx": len(df)-1}]
    return PatternResult("Head & Shoulders", True, "bearish", conf, confirmed,
        [float(h[ls]), float(h[hd]), float(h[rs]), neck], kpts,
        f"ГиП. Голова={h[hd]:.4f} Нек={neck:.4f} Цель≈{neck - height:.4f}")


def detect_inverse_head_shoulders(df: pd.DataFrame, window=100) -> Optional[PatternResult]:
    l = df["low"].values[-window:]
    h = df["high"].values[-window:]
    c = df["close"].values
    _, troughs = _pivots(l, order=5)
    if len(troughs) < 3: return None
    ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]
    if not (l[hd] < l[ls] and l[hd] < l[rs]): return None
    if abs(l[ls] - l[rs]) / (abs(l[hd]) + 1e-10) > 0.07: return None
    neck = float(np.max(h[ls:rs + 1]))
    confirmed = float(c[-1]) > neck
    height = neck - float(l[hd])
    # Правило: перед левым плечом нисходящий тренд
    if not _prior_downtrend(l, ls, lookback=max(ls, 15), min_fall=0.04): return None
    off = len(df) - window
    n1 = off + ls + int(np.argmax(h[ls:hd+1]))
    n2 = off + hd + int(np.argmax(h[hd:rs+1]))
    kpts = [{"price": float(l[ls]), "idx": off+ls}, {"price": neck, "idx": n1},
            {"price": float(l[hd]), "idx": off+hd}, {"price": neck, "idx": n2},
            {"price": float(l[rs]), "idx": off+rs}, {"price": neck, "idx": len(df)-1}]
    return PatternResult("Inverse H&S", True, "bullish", min(0.70 + 0.15 * confirmed, 0.93), confirmed,
        [float(l[ls]), float(l[hd]), float(l[rs]), neck], kpts,
        f"Перевёрнутая ГиП. Голова={l[hd]:.4f} Нек={neck:.4f} Цель≈{neck + height:.4f}")


# ──────────────────────────────────────────────
#  4. WEDGES (клинья)
# ──────────────────────────────────────────────

def detect_wedge(df: pd.DataFrame, window=60) -> Optional[PatternResult]:
    h = df["high"].values[-window:]
    l = df["low"].values[-window:]
    c = df["close"].values
    peaks, _   = _pivots(h, order=4)
    _, troughs = _pivots(l, order=4)
    if len(peaks) < 3 or len(troughs) < 3: return None
    m_h, _ = _linreg(peaks[-3:],   [h[p] for p in peaks[-3:]])
    m_l, _ = _linreg(troughs[-3:], [l[t] for t in troughs[-3:]])
    if m_h > 0 and m_l > 0 and m_h < m_l:
        confirmed = float(c[-1]) < float(l[-3])
        return PatternResult("Rising Wedge", True, "bearish", min(0.65 + 0.2 * confirmed, 0.90), confirmed,
            [float(h[peaks[-1]]), float(l[troughs[-1]])], "Восходящий клин (медвежий). Ждём пробой вниз.")
    if m_h < 0 and m_l < 0 and m_l < m_h:
        confirmed = float(c[-1]) > float(h[-3])
        return PatternResult("Falling Wedge", True, "bullish", min(0.65 + 0.2 * confirmed, 0.90), confirmed,
            [float(h[peaks[-1]]), float(l[troughs[-1]])], "Нисходящий клин (бычий). Ждём пробой вверх.")
    return None


# ──────────────────────────────────────────────
#  5. TRIANGLES
# ──────────────────────────────────────────────

def detect_triangle(df: pd.DataFrame, window=60) -> Optional[PatternResult]:
    h = df["high"].values[-window:]
    l = df["low"].values[-window:]
    c = df["close"].values
    peaks, _   = _pivots(h, order=4)
    _, troughs = _pivots(l, order=4)
    if len(peaks) < 3 or len(troughs) < 3: return None
    m_h, b_h = _linreg(peaks[-3:],   [h[p] for p in peaks[-3:]])
    m_l, b_l = _linreg(troughs[-3:], [l[t] for t in troughs[-3:]])
    if m_h < -1e-4 and m_l > 1e-4:
        confirmed = float(c[-1]) > float(h[-5]) or float(c[-1]) < float(l[-5])
        return PatternResult("Symmetrical Triangle", True, "neutral", min(0.65 + 0.1 * confirmed, 0.85), confirmed,
            [float(h[peaks[-1]]), float(l[troughs[-1]])], "Симметричный треугольник. Ждём пробой.")
    if abs(m_h) < 1e-4 and m_l > 1e-4:
        resistance = float(np.mean([h[p] for p in peaks[-3:]]))
        confirmed  = float(c[-1]) > resistance
        return PatternResult("Ascending Triangle", True, "bullish", 0.70 + 0.15 * confirmed, confirmed,
            [resistance, float(l[troughs[-1]])], f"Восходящий треугольник. Сопр ≈ {resistance:.4f}")
    if abs(m_l) < 1e-4 and m_h < -1e-4:
        support   = float(np.mean([l[t] for t in troughs[-3:]]))
        confirmed = float(c[-1]) < support
        return PatternResult("Descending Triangle", True, "bearish", 0.70 + 0.15 * confirmed, confirmed,
            [float(h[peaks[-1]]), support], f"Нисходящий треугольник. Поддержка ≈ {support:.4f}")
    return None


# ──────────────────────────────────────────────
#  6. BROADENING / EXPANDING TRIANGLE
# ──────────────────────────────────────────────

def detect_broadening_triangle(df: pd.DataFrame, window=60) -> Optional[PatternResult]:
    h = df["high"].values[-window:]
    l = df["low"].values[-window:]
    c = df["close"].values
    peaks, _   = _pivots(h, order=4)
    _, troughs = _pivots(l, order=4)
    if len(peaks) < 3 or len(troughs) < 3: return None
    m_h, _ = _linreg(peaks[-3:],   [h[p] for p in peaks[-3:]])
    m_l, _ = _linreg(troughs[-3:], [l[t] for t in troughs[-3:]])
    if m_h > 1e-4 and m_l < -1e-4:
        mid  = (float(h[peaks[-1]]) + float(l[troughs[-1]])) / 2
        bull = float(c[-1]) < mid
        name = "Bullish Broadening Triangle" if bull else "Bearish Broadening Triangle"
        return PatternResult(name, True, "bullish" if bull else "bearish",
            min(0.60 + 0.1 * abs(m_h - m_l) / (abs(m_h) + 1e-9), 0.80), False,
            [float(h[peaks[-1]]), float(l[troughs[-1]])],
            f"{'Бычий' if bull else 'Медвежий'} расходящийся треугольник.")
    return None


# ──────────────────────────────────────────────
#  7. ROUNDED TOP / BOTTOM
# ──────────────────────────────────────────────

def detect_rounded(df: pd.DataFrame, window=80) -> Optional[PatternResult]:
    c   = df["close"].values[-window:]
    xs  = np.arange(len(c), dtype=float)
    mid = len(c) // 2
    try:
        coeffs = np.polyfit(xs, c.astype(float), 2)
    except Exception:
        return None
    a = coeffs[0]
    fitted  = np.polyval(coeffs, xs)
    r2_den  = float(np.sum((c - c.mean()) ** 2))
    r2      = 1 - float(np.sum((fitted - c) ** 2)) / (r2_den + 1e-10)
    if r2 < 0.50 or abs(a) < 1e-8: return None
    conf = min(0.50 + 0.4 * r2, 0.88)
    if a < 0:
        return PatternResult("Rounded Top", True, "bearish", conf, float(c[-1]) < float(c[mid]),
            [float(np.max(c))], f"Круглая вершина. R²={r2:.2f}")
    else:
        return PatternResult("Rounded Bottom", True, "bullish", conf, float(c[-1]) > float(c[mid]),
            [float(np.min(c))], f"Круглое дно. R²={r2:.2f}")


# ──────────────────────────────────────────────
#  8. CUP & HANDLE / CUP WITHOUT HANDLE
# ──────────────────────────────────────────────

def detect_cup_handle(df: pd.DataFrame, window=120) -> Optional[PatternResult]:
    if len(df) < window: return None
    c       = df["close"].values[-window:].astype(float)
    cup_end = int(len(c) * 0.75)
    cup     = c[:cup_end]
    try:
        coeffs = np.polyfit(np.arange(cup_end, dtype=float), cup, 2)
    except Exception:
        return None
    if coeffs[0] < 1e-8: return None
    rim   = float(max(cup[0], cup[-1]))
    bot   = float(np.min(cup))
    depth = (rim - bot) / (rim + 1e-10)
    if depth < 0.05 or depth > 0.55: return None
    handle         = c[cup_end:]
    has_handle     = len(handle) > 5
    handle_retrace = (float(np.max(handle)) - float(np.min(handle))) / (rim - bot + 1e-10) if has_handle else 0
    confirmed      = float(c[-1]) > rim
    conf           = min(0.65 + 0.15 * confirmed + 0.05 * has_handle, 0.90)
    if has_handle and handle_retrace < 0.55:
        return PatternResult("Cup & Handle", True, "bullish", conf, confirmed, [rim, bot],
            f"Чаша с ручкой. Глубина={depth*100:.0f}%. {'↑ подтверждён' if confirmed else 'Ждём пробой рима'}")
    return PatternResult("Cup (no handle)", True, "bullish", conf - 0.05, confirmed, [rim, bot],
        f"Чаша без ручки. Глубина={depth*100:.0f}%. {'↑ подтверждён' if confirmed else 'Ждём пробой'}")


def detect_inverted_cup_handle(df: pd.DataFrame, window=120) -> Optional[PatternResult]:
    if len(df) < window: return None
    c       = df["close"].values[-window:].astype(float)
    cup_end = int(len(c) * 0.75)
    cup     = c[:cup_end]
    try:
        coeffs = np.polyfit(np.arange(cup_end, dtype=float), cup, 2)
    except Exception:
        return None
    if coeffs[0] > -1e-8: return None
    rim   = float(min(cup[0], cup[-1]))
    top_  = float(np.max(cup))
    depth = (top_ - rim) / (rim + 1e-10)
    if depth < 0.05: return None
    confirmed = float(c[-1]) < rim
    return PatternResult("Inverted Cup & Handle", True, "bearish", min(0.60 + 0.2 * confirmed, 0.85), confirmed,
        [rim, top_], f"Перевёрнутая чаша. {'↓ подтверждён' if confirmed else 'Ждём пробой вниз'}")


# ──────────────────────────────────────────────
#  9. DIAMOND TOP / BOTTOM
# ──────────────────────────────────────────────

def detect_diamond(df: pd.DataFrame, window=100) -> Optional[PatternResult]:
    h = df["high"].values[-window:].astype(float)
    l = df["low"].values[-window:].astype(float)
    c = df["close"].values
    n   = len(h)
    mid = n // 2
    peaks_l, _ = _pivots(h[:mid], order=3)
    peaks_r, _ = _pivots(h[mid:], order=3)
    _, tro_l   = _pivots(l[:mid], order=3)
    _, tro_r   = _pivots(l[mid:], order=3)
    if not (len(peaks_l) >= 2 and len(peaks_r) >= 2 and len(tro_l) >= 2 and len(tro_r) >= 2):
        return None
    m_hl, _ = _linreg(peaks_l[-2:], [h[p] for p in peaks_l[-2:]])
    m_hr, _ = _linreg([p + mid for p in peaks_r[-2:]], [h[mid + p] for p in peaks_r[-2:]])
    m_ll, _ = _linreg(tro_l[-2:],   [l[t] for t in tro_l[-2:]])
    m_lr, _ = _linreg([t + mid for t in tro_r[-2:]], [l[mid + t] for t in tro_r[-2:]])
    if not (m_hl > 0 and m_ll < 0 and m_hr < 0 and m_lr > 0): return None
    price_mid = (float(np.max(h)) + float(np.min(l))) / 2
    is_top    = float(c[-1]) < price_mid
    confirmed = float(c[-1]) < float(l[-5]) if is_top else float(c[-1]) > float(h[-5])
    if is_top:
        return PatternResult("Diamond Top", True, "bearish", 0.65 + 0.2 * confirmed, confirmed,
            [float(np.max(h[mid-5:mid+5]))], "Бриллиант вершины. Медвежий разворот.")
    return PatternResult("Diamond Bottom", True, "bullish", 0.65 + 0.2 * confirmed, confirmed,
        [float(np.min(l[mid-5:mid+5]))], "Бриллиант дна. Бычий разворот.")


# ──────────────────────────────────────────────
#  10. TREND CHANGE
# ──────────────────────────────────────────────

def detect_trend_change(df: pd.DataFrame, window=60) -> Optional[PatternResult]:
    h = df["high"].values[-window:]
    l = df["low"].values[-window:]
    c = df["close"].values
    peaks, _   = _pivots(h, order=4)
    _, troughs = _pivots(l, order=4)
    if len(peaks) < 3 or len(troughs) < 3: return None
    h3 = [h[p] for p in peaks[-3:]]
    l3 = [l[t] for t in troughs[-3:]]
    if h3[0] > h3[1] > h3[2] and l3[-1] > l3[-2]:
        confirmed = float(c[-1]) > h3[-1]
        return PatternResult("Trend Change ↑ (HH→HL)", True, "bullish", 0.65 + 0.2 * confirmed, confirmed,
            [h3[-1], l3[-1]], "Снижение хаёв → рост лоёв. Бычья смена тренда.")
    if l3[0] < l3[1] < l3[2] and h3[-1] < h3[-2]:
        confirmed = float(c[-1]) < l3[-1]
        return PatternResult("Trend Change ↓ (LL→LH)", True, "bearish", 0.65 + 0.2 * confirmed, confirmed,
            [h3[-1], l3[-1]], "Рост лоёв → снижение хаёв. Медвежья смена тренда.")
    return None


# ──────────────────────────────────────────────
#  11. CANDLE PATTERNS
# ──────────────────────────────────────────────

def detect_engulfing(df: pd.DataFrame) -> Optional[PatternResult]:
    if len(df) < 2: return None
    o1, c1 = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
    o2, c2 = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    body1, body2 = abs(c1 - o1), abs(c2 - o2)
    if body2 < body1 * 1.2: return None
    if c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1:
        return PatternResult("Bullish Engulfing", True, "bullish", 0.72, True, [c2],
            "Бычье поглощение: медвежья свеча поглощена бычьей.")
    if c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1:
        return PatternResult("Bearish Engulfing", True, "bearish", 0.72, True, [c2],
            "Медвежье поглощение: бычья свеча поглощена медвежьей.")
    return None


def detect_pin_bar(df: pd.DataFrame, ratio=0.3) -> Optional[PatternResult]:
    if len(df) < 2: return None
    o, h, l, c = (float(df[x].iloc[-1]) for x in ("open", "high", "low", "close"))
    body  = abs(c - o)
    total = h - l
    if total < 1e-10: return None
    upper = h - max(o, c)
    lower = min(o, c) - l
    if body / total < ratio and lower > upper * 2 and lower > total * 0.5:
        return PatternResult("Hammer / Pin Bar ↑", True, "bullish", 0.68, True, [l],
            "Молот: длинная нижняя тень. Бычий сигнал.")
    if body / total < ratio and upper > lower * 2 and upper > total * 0.5:
        return PatternResult("Shooting Star / Pin Bar ↓", True, "bearish", 0.68, True, [h],
            "Падающая звезда: длинная верхняя тень. Медвежий сигнал.")
    return None


# ──────────────────────────────────────────────
#  12. HARMONIC PATTERNS
# ──────────────────────────────────────────────

_HARM = {
    "Gartley":   {"XB":(0.600,0.618),"AC":(0.382,0.886),"BD":(1.130,1.618),"XD":(0.786,0.786)},
    "Butterfly": {"XB":(0.786,0.786),"AC":(0.382,0.886),"BD":(1.618,2.618),"XD":(1.272,1.272)},
    "Bat":       {"XB":(0.382,0.500),"AC":(0.382,0.886),"BD":(1.618,2.618),"XD":(0.886,0.886)},
    "Crab":      {"XB":(0.382,0.618),"AC":(0.382,0.886),"BD":(2.240,3.618),"XD":(1.618,1.618)},
    "Shark":     {"XB":(0.446,0.618),"AC":(1.130,1.618),"BD":(1.618,2.240),"XD":(0.886,1.130)},
}


def _zz(arr, pct=2.0):
    pivs, n = [], len(arr)
    if n < 10: return pivs
    last_dir, last_idx, last_val = None, 0, arr[0]
    for i in range(1, n):
        chg = (arr[i] - last_val) / (last_val + 1e-10) * 100
        if chg >= pct and last_dir != 'H':
            if last_dir == 'L': pivs.append((last_idx, last_val, 'L'))
            last_dir, last_idx, last_val = 'H', i, arr[i]
        elif chg <= -pct and last_dir != 'L':
            if last_dir == 'H': pivs.append((last_idx, last_val, 'H'))
            last_dir, last_idx, last_val = 'L', i, arr[i]
        elif last_dir == 'H' and arr[i] > last_val: last_idx, last_val = i, arr[i]
        elif last_dir == 'L' and arr[i] < last_val: last_idx, last_val = i, arr[i]
    pivs.append((last_idx, last_val, last_dir or 'H'))
    return pivs


def _in_range(val, lo, hi, tol=0.06):
    return lo * (1 - tol) <= val <= hi * (1 + tol)


def detect_abc_correction(df: pd.DataFrame, min_leg_pct: float = 2.5) -> Optional[PatternResult]:
    """
    ABC Zigzag correction (Elliott Wave).

    Медвежий ABC (коррекция вниз после роста):
        A = свинг ХАЙ (начало первой волны вниз)
        B = свинг ЛОУ  (конец волны A, начало ретреса B)
        C = свинг ХАЙ  (конец ретреса B, начало финальной волны C вниз)
           C < A по цене; B ретрейсит A на 38.2–78.6%; C = 0.618–1.618 × длина A
        Цель = конец C − длина A.

    Бычий ABC (коррекция вверх после падения):
        A = свинг ЛОУ
        B = свинг ХАЙ
        C = свинг ЛОУ > A; аналогичные пропорции.
        Цель = конец C + длина A.
    """
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    n = len(df)
    if n < 30:
        return None

    order = max(3, n // 40)   # адаптивный размер окна для свингов

    # Свинг хаи и лоу по правилу order-свечей с каждой стороны
    s_highs = [i for i in range(order, n - order)
               if h[i] == max(h[i-order:i+order+1])]
    s_lows  = [i for i in range(order, n - order)
               if l[i] == min(l[i-order:i+order+1])]

    # Объединяем и сортируем по времени
    pivots = sorted(
        [("H", i, h[i]) for i in s_highs] +
        [("L", i, l[i]) for i in s_lows],
        key=lambda x: x[1],
    )

    # Убираем подряд идущие одного типа (берём экстремум)
    merged = []
    for kind, idx, price in pivots:
        if merged and merged[-1][0] == kind:
            # Оставляем более экстремальный
            if (kind == "H" and price > merged[-1][2]) or (kind == "L" and price < merged[-1][2]):
                merged[-1] = (kind, idx, price)
        else:
            merged.append((kind, idx, price))

    if len(merged) < 3:
        return None

    def _try_abc(kind_a, kind_b, kind_c, is_bearish):
        """Ищем паттерн ABC в массиве merged, начиная с конца."""
        for i in range(len(merged) - 2, 1, -1):
            if merged[i][0] != kind_c:
                continue
            # Ищем B перед C
            for j in range(i - 1, 0, -1):
                if merged[j][0] != kind_b:
                    continue
                # Ищем A перед B
                for k in range(j - 1, -1, -1):
                    if merged[k][0] != kind_a:
                        continue

                    a_kind, a_idx, a_price = merged[k]
                    b_kind, b_idx, b_price = merged[j]
                    c_kind, c_idx, c_price = merged[i]

                    # Минимальный размер ноги A
                    leg_a = abs(b_price - a_price)
                    if leg_a / (a_price + 1e-10) * 100 < min_leg_pct:
                        continue

                    # Ретрейс B: 38.2–78.6% от ноги A
                    leg_b = abs(c_price - b_price)
                    retr_b = leg_b / (leg_a + 1e-10)
                    if not 0.30 <= retr_b <= 0.88:
                        continue

                    # Минимальный размер ноги C (0.5–1.618 × A)
                    leg_c_ratio = leg_b / (leg_a + 1e-10)  # примерно leg_c = leg_b direction continued
                    # Нога C должна продолжаться за конец B
                    if is_bearish and c_price >= a_price:
                        continue   # C должен быть ниже A
                    if not is_bearish and c_price <= a_price:
                        continue   # C должен быть выше A

                    # Нога C должна быть >= min_leg_pct%
                    if leg_b / (b_price + 1e-10) * 100 < min_leg_pct:
                        continue

                    # Уверенность: лучше если retr_b ≈ 0.618 (золотое сечение)
                    conf = 0.65 + 0.20 * (1 - abs(retr_b - 0.618) / 0.3)

                    # Цель (проекция C-волны) = C + length_A
                    leg_a_size = abs(b_price - a_price)
                    if is_bearish:
                        target = c_price - leg_a_size        # цель вниз
                        sl     = c_price + leg_a_size * 0.15
                    else:
                        target = c_price + leg_a_size
                        sl     = c_price - leg_a_size * 0.15

                    # Только если паттерн свежий (C-точка близко к концу данных)
                    if c_idx < n - max(10, n // 10):
                        continue

                    kpts = [
                        {"price": a_price, "idx": a_idx},
                        {"price": b_price, "idx": b_idx},
                        {"price": c_price, "idx": c_idx},
                    ]
                    dir_str = "bearish" if is_bearish else "bullish"
                    rtr_pct  = retr_b * 100
                    desc = (
                        f"ABC Zigzag ({'bear' if is_bearish else 'bull'}): "
                        f"A={a_price:.4f} B={b_price:.4f} C={c_price:.4f}. "
                        f"B retrace: {rtr_pct:.0f}%. Target: {target:.4f}."
                    )
                    return PatternResult(
                        name="ABC Correction",
                        found=True,
                        direction=dir_str,
                        confidence=min(round(conf, 2), 0.90),
                        confirmed=True,
                        key_levels=[float(c_price), float(target), float(sl)],
                        key_points=kpts,
                        description=desc,
                    )
        return None

    # Пробуем оба направления
    result = _try_abc("H", "L", "H", is_bearish=True)   # медвежий ABC
    if result:
        return result
    return _try_abc("L", "H", "L", is_bearish=False)    # бычий ABC


def detect_harmonics(df: pd.DataFrame, zigzag_pct=2.0) -> Optional[PatternResult]:
    c    = df["close"].values.astype(float)
    pivs = _zz(c, zigzag_pct)
    if len(pivs) < 5: return None
    for i in range(len(pivs) - 4):
        X, A, B, C, D = [pivs[i + j][1] for j in range(5)]
        XA = abs(A - X)
        if XA < 1e-10: continue
        AB = abs(B - A); BC = abs(C - B); CD = abs(D - C); XD = abs(D - X)
        bull = A > X
        for name, r in _HARM.items():
            try:
                ok = (_in_range(AB / XA, *r["XB"]) and
                      _in_range(BC / (AB + 1e-10), *r["AC"]) and
                      _in_range(CD / (BC + 1e-10), *r["BD"]) and
                      _in_range(XD / XA, *r["XD"]))
            except Exception:
                ok = False
            if ok:
                return PatternResult(f"Harmonic {name}", True,
                    "bullish" if not bull else "bearish", 0.75, False,
                    [float(D)], f"Гармонический паттерн {name}. D={D:.4f}")
    return None


# ──────────────────────────────────────────────
#  MAIN: scan_patterns / scan_selected
# ──────────────────────────────────────────────

_DETECTORS = [
    detect_double_top,   detect_double_bottom,
    detect_triple_top,   detect_triple_bottom,
    detect_head_shoulders, detect_inverse_head_shoulders,
    detect_wedge,
    detect_triangle,
    detect_broadening_triangle,
    detect_rounded,
    detect_cup_handle,   detect_inverted_cup_handle,
    detect_diamond,
    detect_trend_change,
    detect_engulfing,    detect_pin_bar,
    detect_abc_correction,
    detect_harmonics,
]


def scan_patterns(df: pd.DataFrame, zigzag_pct: float = 2.0) -> list:
    results = []
    for fn in _DETECTORS:
        try:
            r = fn(df)
            if r and r.found:
                results.append(r)
        except Exception:
            pass
    return results


def scan_selected(df: pd.DataFrame, selected: list, zigzag_pct: float = 2.0) -> list:
    """
    selected — список ключевых слов (lowercase), например ['double_bottom','triangle','harmonic'].
    """
    all_pats = scan_patterns(df, zigzag_pct)
    if not selected:
        return all_pats
    result = []
    for p in all_pats:
        name_lc = p.name.lower().replace(" ", "_").replace("&", "")
        if any(k in name_lc for k in selected):
            result.append(p)
    return result
