"""
Полный анализ позиции — синтез всех модулей.
Вызывается когда пользователь нажимает "Анализ" на конкретной монете.
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from screeners.patterns    import scan_patterns, PatternResult
from screeners.smart_money import (
    find_order_blocks, find_fvg, find_stop_hunts, find_liquidity_zones,
    get_tdp_levels, nearest_ob, fresh_fvg_near_price,
)
from screeners.waves       import analyze_waves


# ═══════════════════════════════════════════════════════════
#  RESULT DATA CLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class AnalysisReport:
    symbol:         str
    direction:      str              # "long" | "short" | "wait"
    verdict:        str              # "ВХОДИМ" | "ЖДЁМ" | "ПРОПУСКАЕМ"
    confidence_pct: int              # 0-100

    # Секции анализа
    patterns:       list             # PatternResult
    order_blocks:   list             # OrderBlock
    fvgs:           list             # FVG
    stop_hunts:     list             # StopHunt
    liquidity:      list             # LiquidityZone
    tdp_levels:     list             # TDPLevel
    wave_result:    dict

    # Факторы
    factors_ok:     list[str] = field(default_factory=list)
    factors_no:     list[str] = field(default_factory=list)
    factors_warn:   list[str] = field(default_factory=list)

    # Торговый план
    entry_zone:     tuple[float, float] = (0.0, 0.0)
    stop_loss:      float = 0.0
    take_profit_1:  float = 0.0
    take_profit_2:  float = 0.0
    rr_ratio:       float = 0.0

    # Текст
    summary:        str = ""


# ═══════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════

def generate_analysis(
    symbol:       str,
    klines_15m:   pd.DataFrame,
    klines_1h:    pd.DataFrame,
    screener_row: Optional[pd.Series],
    market_data:  dict,
    wave_pct:     float = 3.0,
) -> AnalysisReport:
    """
    Полный анализ монеты. Возвращает AnalysisReport.
    """

    cur_price = float(klines_15m["close"].iloc[-1]) if not klines_15m.empty else 0.0
    factors_ok:   list[str] = []
    factors_no:   list[str] = []
    factors_warn: list[str] = []

    # ── 1. Паттерны ───────────────────────────────────────
    patterns_15m = scan_patterns(klines_15m, zigzag_pct=2.0)
    patterns_1h  = scan_patterns(klines_1h,  zigzag_pct=wave_pct)

    # Объединить, отдать приоритет 1h (более значимые)
    all_patterns = patterns_1h + patterns_15m
    # Убрать дубли по типу
    seen_names = set()
    patterns_unique = []
    for p in all_patterns:
        if p.name not in seen_names:
            patterns_unique.append(p)
            seen_names.add(p.name)

    # Определить доминирующее направление из паттернов
    long_score  = sum(p.confidence for p in patterns_unique if p.direction == "long")
    short_score = sum(p.confidence for p in patterns_unique if p.direction == "short")

    if patterns_unique:
        for p in patterns_unique:
            icon = "✅" if p.confirmed else "⏳"
            factors_ok.append(f"{icon} {p.name} ({p.direction}) — {p.description[:80]}")
    else:
        factors_no.append("❌ Чёткие ТА-паттерны не найдены")

    # ── 2. Smart Money — ОБ ──────────────────────────────
    obs_15m = find_order_blocks(klines_15m)
    obs_1h  = find_order_blocks(klines_1h)
    all_obs = obs_1h + obs_15m

    near_bull_ob = nearest_ob(all_obs, cur_price, "bullish", max_dist_pct=2.5)
    near_bear_ob = nearest_ob(all_obs, cur_price, "bearish", max_dist_pct=2.5)

    if near_bull_ob:
        factors_ok.append(
            f"✅ Бычий ОБ ${near_bull_ob.low:.4f}—${near_bull_ob.high:.4f} "
            f"(сила ×{near_bull_ob.strength:.1f}, {'свежий' if not near_bull_ob.mitigated else 'митигирован'})"
        )
    else:
        factors_warn.append("⚠️ Бычий ордер блок не найден в радиусе 2.5%")

    if near_bear_ob:
        factors_warn.append(
            f"⚠️ Медвежий ОБ рядом ${near_bear_ob.low:.4f}—${near_bear_ob.high:.4f} "
            f"(может давить на цену)"
        )

    # ── 3. FVG / Имбаланс ────────────────────────────────
    fvgs_15m  = find_fvg(klines_15m)
    fvgs_1h   = find_fvg(klines_1h)
    all_fvgs  = fvgs_1h + fvgs_15m

    near_bull_fvg = fresh_fvg_near_price(all_fvgs, cur_price, "bullish", 2.0)
    near_bear_fvg = fresh_fvg_near_price(all_fvgs, cur_price, "bearish", 2.0)

    if near_bull_fvg:
        factors_ok.append(
            f"✅ Бычий FVG (имбаланс) ${near_bull_fvg.low:.4f}—${near_bull_fvg.high:.4f} "
            f"({near_bull_fvg.size_pct:.2f}%, {'закрыт' if near_bull_fvg.filled else 'открыт'})"
        )
    else:
        factors_no.append("❌ Незакрытый бычий FVG не найден")

    if near_bear_fvg:
        factors_warn.append(
            f"⚠️ Медвежий FVG ${near_bear_fvg.low:.4f}—${near_bear_fvg.high:.4f} (нависает)"
        )

    # ── 4. TTS — Stop Hunt ────────────────────────────────
    hunts = find_stop_hunts(klines_15m)
    recent_hunt = next(
        (h for h in reversed(hunts)
         if (len(klines_15m) - h.candle_idx) <= 10),
        None
    )
    if recent_hunt:
        if recent_hunt.kind == "bullish":
            factors_ok.append(f"✅ TTS Бычий захват ликвидности: {recent_hunt.description[:80]}")
        else:
            factors_warn.append(f"⚠️ TTS Медвежий захват ликвидности: {recent_hunt.description[:80]}")
    else:
        factors_warn.append("⚠️ Свежего TTS не обнаружено (ликвидность не снята)")

    # ── 5. TDP — Daily Levels ─────────────────────────────
    tdp_levels = get_tdp_levels(klines_1h)
    if tdp_levels:
        for lvl in tdp_levels:
            dist = abs(lvl.price - cur_price) / cur_price * 100
            if dist < 1.5:
                factors_ok.append(f"✅ TDP: цена у ключевого уровня {lvl.description} (${lvl.price:.4f}, {dist:.1f}%)")
            else:
                factors_warn.append(f"⚠️ TDP {lvl.description}: ${lvl.price:.4f} ({dist:.1f}% от цены)")
    else:
        factors_warn.append("⚠️ TDP уровни не определены (нужны данные 1h)")

    # ── 6. Ликвидность ────────────────────────────────────
    liq_zones = find_liquidity_zones(klines_1h)
    unswept_below = [z for z in liq_zones if z.kind == "equal_lows"
                     and not z.swept and z.price < cur_price]
    if unswept_below:
        factors_warn.append(
            f"⚠️ Ликвидность под ценой (не снята): ${unswept_below[0].price:.4f} "
            f"({unswept_below[0].touches} касаний) — может притянуть цену"
        )
    else:
        factors_ok.append("✅ Ликвидность под ценой снята (чистое движение вверх)")

    # ── 7. Волновой анализ ────────────────────────────────
    wave_result = analyze_waves(klines_1h, pct=wave_pct)
    wave_type   = wave_result.get("wave_type", "unknown")
    wave_desc   = wave_result.get("description", "")
    wave_pivots = wave_result.get("pivots", [])

    labeled_waves = [p for p in wave_pivots if p.label]
    last_wave_label = labeled_waves[-1].label if labeled_waves else ""

    if wave_type == "impulse":
        if last_wave_label in ("1", "3"):
            factors_ok.append(f"✅ Волны: {wave_desc} — заходная волна (продолжение)")
        elif last_wave_label == "5":
            factors_warn.append(f"⚠️ Волны: 5-я волна — возможная коррекция впереди")
        elif last_wave_label in ("2", "4"):
            factors_ok.append(f"✅ Волны: коррекционная волна {last_wave_label} — готовимся к продолжению")
    elif wave_type == "corrective":
        if last_wave_label == "C":
            factors_ok.append(f"✅ Волны: ABC коррекция завершается — возможен новый импульс")
        else:
            factors_warn.append(f"⚠️ Волны: коррекция в процессе ({wave_desc})")
    else:
        factors_warn.append(f"⚠️ Волны: разметка не определена")

    # ── 8. Моментум (из screener_row) ────────────────────
    oi_change   = float(screener_row.get("OI_Change", 0)) if screener_row is not None else 0
    change_24h  = float(screener_row.get("Change24h", 0)) if screener_row is not None else 0
    cvd         = str(screener_row.get("CVD", "–"))        if screener_row is not None else "–"
    funding     = float(screener_row.get("Funding", 0))    if screener_row is not None else 0

    if oi_change > 5 and change_24h > 0:
        factors_ok.append(f"✅ OI↑{oi_change:+.1f}% + Цена↑{change_24h:+.1f}% — лонги набирают")
    elif oi_change > 5 and change_24h < 0:
        factors_no.append(f"❌ OI↑ но Цена↓ — шорты набирают (не наш сетап для лонга)")
    elif oi_change < -5:
        factors_warn.append(f"⚠️ OI↓{oi_change:.1f}% — делеверидж / позиции закрываются")
    else:
        factors_warn.append(f"⚠️ OI нейтральный ({oi_change:+.1f}%)")

    if cvd == "↑":
        factors_ok.append("✅ CVD↑ — агрессивные покупки (тейкеры покупают)")
    elif cvd == "↓":
        factors_no.append("❌ CVD↓ — агрессивные продажи")
    else:
        factors_warn.append("⚠️ CVD нейтральный")

    if abs(funding) < 0.01:
        factors_warn.append(f"⚠️ Фандинг нейтральный ({funding:+.4f}%) — слабое подтверждение")
    elif funding > 0.03:
        factors_warn.append(f"⚠️ Фандинг перегрет ({funding:+.4f}%) — лонги переполнены")
    elif 0 < funding <= 0.03:
        factors_ok.append(f"✅ Фандинг умеренно позитивный ({funding:+.4f}%) — лонг-перевес")

    # ── 9. Рыночный контекст ─────────────────────────────
    usdt_d = market_data.get("usdt_dominance", 0)
    mc_chg = market_data.get("market_cap_change_24h", 0)

    if usdt_d < 4.8:
        factors_ok.append(f"✅ USDT.D = {usdt_d:.2f}% (низко) — деньги в крипте")
    elif usdt_d > 6.0:
        factors_no.append(f"❌ USDT.D = {usdt_d:.2f}% (высоко) — деньги уходят в стейблы")
    else:
        factors_warn.append(f"⚠️ USDT.D = {usdt_d:.2f}% — нейтральная зона")

    if mc_chg > 1:
        factors_ok.append(f"✅ Total MarCap +{mc_chg:.1f}% за 24ч — рынок растёт")
    elif mc_chg < -2:
        factors_no.append(f"❌ Total MarCap {mc_chg:.1f}% за 24ч — рынок падает")
    else:
        factors_warn.append(f"⚠️ Total MarCap {mc_chg:+.1f}% — нейтрально")

    # ── 10. Финальный вердикт ─────────────────────────────
    ok_count   = len(factors_ok)
    no_count   = len(factors_no)
    warn_count = len(factors_warn)
    total      = ok_count + no_count + warn_count

    # Доминирующее направление
    direction = "long"
    if short_score > long_score:
        direction = "short"
    if ok_count == 0:
        direction = "wait"

    raw_conf = (ok_count * 1.0 + warn_count * 0.3) / max(total, 1)

    # Штраф за противоречия
    if no_count > 0:
        raw_conf *= (1 - 0.15 * no_count)
    if direction == "short" and oi_change > 0 and change_24h < 0:
        raw_conf *= 0.8   # шортовый сетап с OI↑ — более рискованный

    confidence_pct = int(min(max(raw_conf * 100, 0), 100))

    if confidence_pct >= 65 and no_count <= 2:
        verdict = "✅ ВХОДИМ"
    elif confidence_pct >= 45:
        verdict = "⏳ ЖДЁМ"
    else:
        verdict = "🚫 ПРОПУСКАЕМ"

    # ── 11. Торговый план ─────────────────────────────────
    entry_zone = (cur_price, cur_price)
    stop_loss  = cur_price * 0.97
    tp1 = tp2 = 0.0

    # Приоритет: бычий ОБ / FVG для зоны входа
    if near_bull_ob and direction == "long":
        entry_zone = (near_bull_ob.low, near_bull_ob.high)
        stop_loss  = near_bull_ob.low * 0.992
    elif near_bull_fvg and direction == "long":
        entry_zone = (near_bull_fvg.low, near_bull_fvg.high)
        stop_loss  = near_bull_fvg.low * 0.992
    elif near_bear_ob and direction == "short":
        entry_zone = (near_bear_ob.low, near_bear_ob.high)
        stop_loss  = near_bear_ob.high * 1.008

    # Фибоначчи для тейк-профита
    fib_levels = wave_result.get("fib_levels", {})
    if fib_levels and direction == "long":
        tp1 = fib_levels.get("0.618", cur_price * 1.03)
        tp2 = fib_levels.get("1.0",   cur_price * 1.06)
    elif fib_levels and direction == "short":
        tp1 = fib_levels.get("0.618", cur_price * 0.97)
        tp2 = fib_levels.get("1.0",   cur_price * 0.94)
    else:
        tp1 = cur_price * (1.03 if direction == "long" else 0.97)
        tp2 = cur_price * (1.06 if direction == "long" else 0.94)

    # R:R
    entry_mid = (entry_zone[0] + entry_zone[1]) / 2
    risk      = abs(entry_mid - stop_loss)
    reward    = abs(tp1 - entry_mid)
    rr        = round(reward / risk, 1) if risk > 0 else 0

    # ── Итог ─────────────────────────────────────────────
    summary_lines = [
        f"**{verdict}** по {symbol} | Уверенность: {confidence_pct}% | R:R ≈ 1:{rr}",
        f"Подтверждающих: {ok_count} | Противоречий: {no_count} | Предупреждений: {warn_count}",
    ]

    return AnalysisReport(
        symbol=symbol, direction=direction,
        verdict=verdict, confidence_pct=confidence_pct,
        patterns=patterns_unique,
        order_blocks=all_obs, fvgs=all_fvgs,
        stop_hunts=hunts, liquidity=liq_zones, tdp_levels=tdp_levels,
        wave_result=wave_result,
        factors_ok=factors_ok, factors_no=factors_no, factors_warn=factors_warn,
        entry_zone=entry_zone, stop_loss=stop_loss,
        take_profit_1=tp1, take_profit_2=tp2, rr_ratio=rr,
        summary=" | ".join(summary_lines),
    )
