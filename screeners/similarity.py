"""
Историческое сравнение паттернов — поиск похожих ситуаций в истории.
Принцип: берём последние N свечей как шаблон, ищем аналоги в истории,
смотрим что было дальше → предсказываем направление.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimilarPattern:
    idx:          int         # позиция в истории
    timestamp:    pd.Timestamp
    correlation:  float       # схожесть 0–1
    next_change_pct: float    # % движение после паттерна
    direction:    str         # "up" | "down"
    candles_after: list       # нормализованные свечи после


@dataclass
class SimilarityResult:
    matches:        list[SimilarPattern]
    up_probability: float          # вероятность роста (0–1)
    avg_move_pct:   float          # среднее движение %
    median_move:    float
    predicted_path: list[float]    # средняя траектория следующих N свечей
    confidence:     float          # уверенность предсказания 0–1
    description:    str


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score нормализация."""
    std = arr.std()
    if std < 1e-10:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def _pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Корреляция Пирсона между двумя нормализованными последовательностями."""
    if len(a) != len(b):
        return 0.0
    try:
        r = np.corrcoef(a, b)[0, 1]
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        return 0.0


def find_similar_patterns(
    klines:        pd.DataFrame,
    template_len:  int   = 30,   # размер шаблона (последних N свечей)
    predict_len:   int   = 20,   # сколько свечей вперёд предсказывать
    top_n:         int   = 10,   # топ N похожих
    min_corr:      float = 0.70, # мин. корреляция
) -> SimilarityResult:
    """
    Находит похожие исторические паттерны и строит прогноз.
    """
    if klines is None or len(klines) < template_len + predict_len + 10:
        return SimilarityResult(
            matches=[], up_probability=0.5, avg_move_pct=0.0,
            median_move=0.0, predicted_path=[], confidence=0.0,
            description="Недостаточно исторических данных для сравнения.",
        )

    df = klines.reset_index(drop=True)
    closes = df["close"].values

    # Шаблон = последние template_len свечей
    template_raw = closes[-(template_len):]
    template_norm = _normalize(template_raw)

    # Sliding window по всей истории (кроме последних template_len + predict_len)
    matches: list[SimilarPattern] = []
    search_end = len(df) - template_len - predict_len

    for i in range(0, search_end, 1):
        window = closes[i: i + template_len]
        if len(window) < template_len:
            break

        window_norm = _normalize(window)
        corr = _pattern_similarity(template_norm, window_norm)

        if corr < min_corr:
            continue

        # Что было дальше
        after_start = i + template_len
        after_slice = closes[after_start: after_start + predict_len]
        if len(after_slice) < predict_len:
            continue

        base_price   = closes[i + template_len - 1]
        end_price    = closes[after_start + predict_len - 1]
        next_change  = (end_price - base_price) / base_price * 100

        # Нормализованные свечи после (для усреднения)
        after_norm = (after_slice - base_price) / base_price * 100

        matches.append(SimilarPattern(
            idx=i,
            timestamp=df["timestamp"].iloc[i],
            correlation=round(corr, 3),
            next_change_pct=round(next_change, 3),
            direction="up" if next_change > 0 else "down",
            candles_after=after_norm.tolist(),
        ))

    # Сортируем по корреляции, берём топ N
    matches = sorted(matches, key=lambda m: m.correlation, reverse=True)[:top_n]

    if not matches:
        return SimilarityResult(
            matches=[], up_probability=0.5, avg_move_pct=0.0,
            median_move=0.0, predicted_path=[], confidence=0.0,
            description=f"Похожих паттернов не найдено (мин. корреляция {min_corr:.0%}).",
        )

    # Статистика
    changes       = [m.next_change_pct for m in matches]
    up_count      = sum(1 for c in changes if c > 0)
    up_prob       = up_count / len(matches)
    avg_move      = float(np.mean(changes))
    median_move   = float(np.median(changes))

    # Средняя траектория
    paths = [m.candles_after for m in matches if len(m.candles_after) == predict_len]
    if paths:
        predicted_path = list(np.mean(paths, axis=0))
    else:
        predicted_path = []

    # Уверенность = согласованность направлений
    up_share   = up_prob
    dn_share   = 1 - up_prob
    confidence = max(up_share, dn_share) * np.mean([m.correlation for m in matches])
    confidence = round(min(confidence, 1.0), 2)

    dominant = "ВВЕРХ 📈" if up_prob >= 0.5 else "ВНИЗ 📉"
    desc = (
        f"Найдено **{len(matches)}** похожих паттернов "
        f"(мин. корреляция {min_corr:.0%}). "
        f"**{up_count}/{len(matches)}** дали рост. "
        f"Прогноз: **{dominant}** "
        f"({up_prob*100:.0f}% вероятность). "
        f"Среднее движение: **{avg_move:+.2f}%**, медиана: {median_move:+.2f}%."
    )

    return SimilarityResult(
        matches=matches,
        up_probability=up_prob,
        avg_move_pct=avg_move,
        median_move=median_move,
        predicted_path=predicted_path,
        confidence=confidence,
        description=desc,
    )
