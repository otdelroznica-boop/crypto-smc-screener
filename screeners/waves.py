"""
Волновой анализ — автоматическое определение опорных точек (ZigZag)
и разметка волн Эллиотта на свечных данных.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Pivot:
    idx:       int
    timestamp: pd.Timestamp
    price:     float
    kind:      str   # "H" | "L"
    label:     str = ""   # "1","2","3","4","5","A","B","C" или ""


# ═══════════════════════════════════════════════════════════
#  ZigZag — поиск значимых опорных точек
# ═══════════════════════════════════════════════════════════

def find_pivots(df: pd.DataFrame, pct: float = 3.0) -> list[Pivot]:
    """
    Классический ZigZag алгоритм.
    pct — минимальное % движение для нового пивота.
    """
    if df.empty or len(df) < 5:
        return []

    highs = df["high"].values
    lows  = df["low"].values
    ts    = df["timestamp"].values

    pivots: list[Pivot] = []

    # Определить начальное направление
    direction = "up" if df["close"].iloc[-1] > df["close"].iloc[0] else "down"
    last_price = lows[0] if direction == "up" else highs[0]
    last_idx   = 0

    for i in range(1, len(df)):
        if direction == "up":
            if highs[i] > last_price:
                last_price = highs[i]
                last_idx   = i
            elif lows[i] < last_price * (1 - pct / 100):
                pivots.append(Pivot(last_idx, pd.Timestamp(ts[last_idx]), last_price, "H"))
                direction  = "down"
                last_price = lows[i]
                last_idx   = i
        else:
            if lows[i] < last_price:
                last_price = lows[i]
                last_idx   = i
            elif highs[i] > last_price * (1 + pct / 100):
                pivots.append(Pivot(last_idx, pd.Timestamp(ts[last_idx]), last_price, "L"))
                direction  = "up"
                last_price = highs[i]
                last_idx   = i

    # Добавить последнюю точку
    if direction == "up":
        pivots.append(Pivot(last_idx, pd.Timestamp(ts[last_idx]), last_price, "H"))
    else:
        pivots.append(Pivot(last_idx, pd.Timestamp(ts[last_idx]), last_price, "L"))

    return pivots


# ═══════════════════════════════════════════════════════════
#  Разметка волн Эллиотта
# ═══════════════════════════════════════════════════════════

def _label_impulse(pivots: list[Pivot]) -> bool:
    """
    Пытается разметить 5-волновой импульс на последних точках.
    Правила Эллиотта:
      — Волна 2 не отыгрывает более 100% волны 1
      — Волна 3 — не самая короткая из 1,3,5
      — Волна 4 не перекрывает территорию волны 1
    Возвращает True если разметка прошла.
    """
    if len(pivots) < 5:
        return False

    pts = pivots[-5:]

    # Проверить чередование H/L
    kinds = [p.kind for p in pts]
    alt_up   = ["L","H","L","H","L"]   # начало снизу
    alt_down = ["H","L","H","L","H"]   # начало сверху

    if kinds not in (alt_up, alt_down):
        return False

    p = [pt.price for pt in pts]
    going_up = (kinds == alt_up)

    if going_up:
        w1 = p[1] - p[0]  # высота волны 1
        w2 = p[1] - p[2]  # глубина волны 2
        w3 = p[3] - p[2]  # высота волны 3
        w4 = p[3] - p[4]  # глубина волны 4
        w5_start = p[4]
        # Правило 1: Волна 2 < 100% волны 1
        if w2 > w1:           return False
        # Правило 2: Волна 3 не самая короткая
        # Правило 3: Волна 4 не заходит в зону волны 1
        if p[4] < p[1]:       return False
    else:
        w1 = p[0] - p[1]
        w2 = p[2] - p[1]
        w3 = p[2] - p[3]
        w4 = p[4] - p[3]
        if w2 > w1:           return False
        if p[4] > p[1]:       return False

    labels = ["1","2","3","4","5"]
    for pt, lbl in zip(pts, labels):
        pt.label = lbl

    return True


def _label_corrective(pivots: list[Pivot]) -> bool:
    """
    Пытается разметить ABC-коррекцию на последних 3 точках.
    """
    if len(pivots) < 3:
        return False

    pts   = pivots[-3:]
    kinds = [p.kind for p in pts]

    if kinds not in (["H","L","H"], ["L","H","L"]):
        return False

    for pt, lbl in zip(pts, ["A","B","C"]):
        pt.label = lbl

    return True


def analyze_waves(df: pd.DataFrame, pct: float = 3.0) -> dict:
    """
    Главная функция.
    Возвращает:
      pivots      — список Pivot с разметкой
      wave_type   — "impulse" | "corrective" | "unknown"
      description — текстовое описание ситуации
      fib_levels  — словарь уровней Фибоначчи от последнего движения
    """
    pivots = find_pivots(df, pct=pct)

    if len(pivots) < 3:
        return {
            "pivots": pivots,
            "wave_type": "unknown",
            "description": "Недостаточно данных для волнового анализа. Снизьте % ZigZag.",
            "fib_levels": {},
        }

    wave_type = "unknown"
    if _label_impulse(pivots):
        wave_type = "impulse"
    elif _label_corrective(pivots):
        wave_type = "corrective"

    # Описание
    last = pivots[-1]
    prev = pivots[-2]

    if wave_type == "impulse":
        last_wave = pivots[-1].label
        if last_wave == "5":
            desc = "🔢 Завершение 5-волнового импульса. Ожидается ABC-коррекция."
        elif last_wave == "3":
            desc = "🚀 Идёт 3-я волна (самая сильная). Тренд продолжается."
        else:
            desc = f"🔢 Импульс: последняя волна {last_wave}."
    elif wave_type == "corrective":
        last_wave = pivots[-1].label
        if last_wave == "C":
            desc = "📉 Завершение ABC-коррекции. Возможно начало нового импульса."
        else:
            desc = f"📉 Коррекция: волна {last_wave}."
    else:
        desc = f"Текущее движение: {'вверх' if last.kind == 'H' else 'вниз'}. Разметка не найдена."

    # Уровни Фибоначчи от последнего полного движения
    fib = _fib_levels(prev.price, last.price)

    return {
        "pivots":      pivots,
        "wave_type":   wave_type,
        "description": desc,
        "fib_levels":  fib,
    }


# ═══════════════════════════════════════════════════════════
#  Фибоначчи
# ═══════════════════════════════════════════════════════════

FIB_RATIOS = {
    "0.0":   0.000,
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.786": 0.786,
    "1.0":   1.000,
    "1.272": 1.272,
    "1.618": 1.618,
}


def _fib_levels(start: float, end: float) -> dict[str, float]:
    diff = end - start
    return {k: end - v * diff for k, v in FIB_RATIOS.items()}


def fib_from_pivots(pivots: list[Pivot]) -> dict[str, float]:
    """Уровни Фибоначчи от последних двух опорных точек."""
    if len(pivots) < 2:
        return {}
    return _fib_levels(pivots[-2].price, pivots[-1].price)
