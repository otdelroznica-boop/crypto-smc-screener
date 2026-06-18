"""
Trading Screener Pro — главный дашборд.
Запуск: python -m streamlit run dashboard.py --server.headless true
"""
import os
from datetime import datetime
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from streamlit_autorefresh import st_autorefresh

from data.exchange       import get_klines, compute_cvd, get_oi_history
from data.market         import get_global_market
from data.sentiment      import get_fear_greed, get_trending_coins, get_top_movers, get_social_score
from data.investment     import get_investment_profile
from screeners.core      import run_screener
from screeners.waves     import analyze_waves, FIB_RATIOS
from screeners.patterns  import scan_patterns
from screeners.smart_money import find_order_blocks, find_fvg, find_stop_hunts, \
                                  find_liquidity_zones, get_tdp_levels, \
                                  find_swing_points, find_breaker_blocks, \
                                  find_named_liquidity_levels, find_rsi_divergence, \
                                  get_killzone_info, scan_smc_setups
from screeners.full_analysis import generate_analysis
from screeners.similarity import find_similar_patterns
from alerts.telegram     import TelegramAlert
from ui.tv_chart         import render_tv_chart, ALL_TIMEFRAMES, TF_HIERARCHY
import base64

BASE_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════════════════════════
#  SPARKLINE — миниатюрный SVG превью паттерна
# ═══════════════════════════════════════════════════════════

def _sparkline_svg(closes: list, key_levels: list = None,
                   direction: str = "bullish",
                   w: int = 160, h: int = 52) -> str:
    """Генерирует SVG спарклайн и возвращает data URI для ImageColumn."""
    if not closes or len(closes) < 3:
        return ""
    mn, mx = min(closes), max(closes)
    rng = mx - mn or (mn * 0.01) or 1e-10

    def py(price):
        return h - 4 - int((price - mn) / rng * (h - 8))

    def px(i):
        return 2 + int(i / (len(closes) - 1) * (w - 4))

    color = "#a6e3a1" if direction == "bullish" else "#f38ba8"
    pts = " ".join(f"{px(i)},{py(c)}" for i, c in enumerate(closes))

    # Горизонтальные уровни паттерна
    level_svg = ""
    for lvl in (key_levels or [])[:3]:
        if lvl and mn <= lvl <= mx:
            y = py(lvl)
            level_svg += (f'<line x1="2" y1="{y}" x2="{w-2}" y2="{y}" '
                          f'stroke="#f9e2af" stroke-width="0.8" stroke-dasharray="3,2"/>')

    # Заливка под линией
    fill_pts = f"2,{h-2} {pts} {w-2},{h-2}"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect width="{w}" height="{h}" rx="3" fill="#0f0f1a"/>'
        f'{level_svg}'
        f'<polygon points="{fill_pts}" fill="{color}" fill-opacity="0.12"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"'
        f' stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{px(len(closes)-1)}" cy="{py(closes[-1])}" r="2.5" fill="{color}"/>'
        f'</svg>'
    )
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"

# ═══════════════════════════════════════════════════════════
#  КОНФИГ
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_config():
    with open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

# ═══════════════════════════════════════════════════════════
#  PAGE
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Trading Screener Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
[data-testid="metric-container"] {
    background:#1e1e2e; border:1px solid #313244;
    border-radius:8px; padding:10px 14px;
}
.screener-title { font-size:1.6rem; font-weight:700; margin:0; }
.verdict-box {
    border-radius:10px; padding:14px 18px; margin:8px 0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
if "signal_counter"   not in st.session_state: st.session_state.signal_counter   = defaultdict(int)
if "alert_bot"        not in st.session_state: st.session_state.alert_bot        = None
if "h_lines"          not in st.session_state: st.session_state.h_lines          = []
if "fib_range"        not in st.session_state: st.session_state.fib_range        = None
if "analysis_sym"     not in st.session_state: st.session_state.analysis_sym     = None
if "chart_sym"        not in st.session_state: st.session_state.chart_sym        = None
if "chart_tf"         not in st.session_state: st.session_state.chart_tf         = "15m"
if "pat_highlight"    not in st.session_state: st.session_state.pat_highlight     = None
if "inv_analysis_sym" not in st.session_state: st.session_state.inv_analysis_sym = None

# ═══════════════════════════════════════════════════════════
#  САЙДБАР
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Настройки")

    exchange_label = st.selectbox(
        "Биржа",
        ["Binance Futures", "Bybit Futures"],
        help="Binance — максимальная ликвидность. Bybit — альтернатива с похожим API.",
    )
    exchange = "bybit" if "Bybit" in exchange_label else "binance"

    st.divider()
    st.markdown("### 🔍 Скринеры")
    pump_on    = st.toggle("Pump / Dump",          True,  help="Резкое % изменение цены за 24ч.")
    oi_on      = st.toggle("Открытый интерес",     True,  help="OI↑+Цена↑ = лонги набирают.")
    cvd_on     = st.toggle("CVD",                  True,  help="Кумулятивная дельта: покупки vs продажи.")
    funding_on = st.toggle("Фандинг",              True,  help="Экстремальный фандинг = перегрев позиций.")

    st.divider()
    st.markdown("### 🎛️ Пороги")
    min_price_pct = st.slider("Мин. изм. цены %",    0.5, 20.0, float(cfg["screeners"]["pump"]["min_price_change"]),    0.5,
        help="Мин. % изменение цены за 24ч для сигнала PUMP/DUMP.")
    min_oi_pct    = st.slider("Мин. изм. ОИ %",      1.0, 30.0, float(cfg["screeners"]["open_interest"]["min_oi_change"]), 0.5,
        help="Мин. % рост OI за выбранный период.")
    funding_thr   = st.slider("Мин. |фандинг| %",   0.01,  0.2, float(cfg["screeners"]["funding"]["alert_above"]), 0.01,
        help="Экстремальное значение ставки фандинга (норма = 0.01%).")

    st.divider()
    st.markdown("### 📐 Периоды")
    oi_period   = st.selectbox("Период ОИ", ["5m","15m","30m","1h","4h"],
        index=["5m","15m","30m","1h","4h"].index(cfg["screeners"]["open_interest"]["oi_period"]),
        help="Период для расчёта изменения OI. 5m = быстрые всплески, 4h = накопление.")
    oi_lookback = st.slider("Окно ОИ (свечи)", 2, 24, int(cfg["screeners"]["open_interest"]["oi_lookback"]),
        help="Сравниваем OI сейчас vs N периодов назад. Период×Окно = временной горизонт.")
    cvd_interval = st.selectbox("Период CVD", ["1m","3m","5m","15m","1h"], index=2,
        help="Таймфрейм свечей для CVD. Меньше = реакция на краткосрочный поток.")

    st.divider()
    st.markdown("### 🌐 Фильтры")
    min_volume_m   = st.number_input("Мин. объём 24ч ($M)", 1, 500, 5,
        help="Фильтр по ликвидности. $5M+ = нормальная торговля.")
    top_n          = st.slider("Топ N монет", 10, 200, int(cfg["market"]["top_n"]), 10,
        help="Сканируем топ N по объёму. Больше = дольше (~1с/10 монет).")
    min_confidence = st.slider("Мин. уверенность анализа %", 0, 90, 0, 5,
        help="Показывать только пары где полный анализ даёт уверенность ≥ X%. "
             "0 = все пары. 50+ = только сильные сетапы (работает медленнее).")
    exclude_sc     = st.toggle("Исключить стейблкоины", True)

    st.divider()
    st.markdown("### 📬 Telegram")
    tg_token    = st.text_input("Bot Token", type="password", value=cfg["alerts"]["telegram"]["bot_token"],
        help="Токен от @BotFather. Создать: @BotFather → /newbot → скопировать токен.")
    tg_chat_id  = st.text_input("Chat ID",              value=cfg["alerts"]["telegram"]["chat_id"],
        help="ID чата. Узнать: написать @userinfobot в Telegram.")
    tg_enabled  = st.toggle("Уведомления",              value=cfg["alerts"]["telegram"]["enabled"])
    tg_min_score= st.slider("Мин. Score для алерта", 1, 5, int(cfg["scoring"]["min_score_alert"]),
        help="Score 3+ = хороший сигнал, 4-5 = очень сильный.")
    if st.button("🔌 Тест Telegram", use_container_width=True):
        bot = TelegramAlert(tg_token, tg_chat_id)
        st.success("✅ OK") if bot.test_connection() else st.error("❌ Ошибка")

    st.divider()
    st.markdown("### 🔄 Обновление")
    refresh_sec = st.slider("Интервал (сек)", 15, 300, int(cfg["update_interval"]), 15,
        help="15-30 сек = почти реальное время. 60-120 сек = умеренная нагрузка на API.")
    auto_ref    = st.toggle("Авто-обновление", True)

# ═══════════════════════════════════════════════════════════
#  AUTO-REFRESH
# ═══════════════════════════════════════════════════════════
if auto_ref:
    st_autorefresh(interval=refresh_sec * 1000, key="main_refresh")

# ═══════════════════════════════════════════════════════════
#  ШАПКА
# ═══════════════════════════════════════════════════════════
c1, c2, c3 = st.columns([5, 2, 1])
with c1:
    st.markdown('<p class="screener-title">📊 Trading Screener Pro</p>', unsafe_allow_html=True)
    st.caption(f"**{exchange_label}** · {datetime.now().strftime('%H:%M:%S')}")
with c3:
    if st.button("🔄", use_container_width=True, help="Принудительное обновление данных"):
        st.cache_data.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════
#  РЫНОЧНЫЙ ОБЗОР + СЕНТИМЕНТ
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=120)
def _mkt():  return get_global_market()
@st.cache_data(ttl=1800)
def _fg():   return get_fear_greed()
@st.cache_data(ttl=600)
def _trend(): return get_trending_coins()

mkt   = _mkt()
fg    = _fg()
trend = _trend()

usdt_d = mkt.get("usdt_dominance", 0)
btc_d  = mkt.get("btc_dominance", 0)
tot_mc = mkt.get("total_market_cap", 0)
mc_chg = mkt.get("market_cap_change_24h", 0)
fg_val = fg.get("value", 50)

bull_t = cfg["usdt_dominance"]["bullish_below"]
bear_t = cfg["usdt_dominance"]["bearish_above"]
if   usdt_d < bull_t: mkt_sig = "🟢 БЫЧИЙ"
elif usdt_d > bear_t: mkt_sig = "🔴 МЕДВЕЖИЙ"
else:                 mkt_sig = "🟡 НЕЙТРАЛЬНЫЙ"

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("USDT.D",         f"{usdt_d:.2f}%",          help="↓ падает = деньги в крипте")
m2.metric("BTC.D",          f"{btc_d:.2f}%",           help="↑ растёт = альты слабее")
m3.metric("Total MarCap",   f"${tot_mc/1e9:.1f}B",     delta=f"{mc_chg:+.2f}%")
m4.metric("F&G Index",      f"{fg_val} · {fg.get('label','–')}",
          help="Fear & Greed Index. <30 = страх (покупай). >75 = жадность (осторожно).")
m5.metric("Сигнал рынка",   mkt_sig,                   help="На основе USDT.D")
m6.metric("Трендовых монет",f"{len(trend)} 🔥",         help="Монеты в топ-7 трендов CoinGecko")

st.divider()

# ═══════════════════════════════════════════════════════════
#  СКРИНИНГ
# ═══════════════════════════════════════════════════════════
@st.cache_data(ttl=refresh_sec, show_spinner=False)
def _scan(_mp,_mo,_op,_ol,_ci,_mv,_tn,_es,_ft,_ex):
    return run_screener(
        min_price_pct=_mp, min_oi_pct=_mo, oi_period=_op,
        oi_lookback=_ol, cvd_interval=_ci, min_volume=_mv*1_000_000,
        top_n=_tn, exclude_stablecoins=_es, funding_threshold=_ft, exchange=_ex,
    )

with st.spinner("⏳ Сканирую рынок..."):
    df_sig = _scan(min_price_pct, min_oi_pct, oi_period, oi_lookback,
                   cvd_interval, min_volume_m, top_n, exclude_sc, funding_thr, exchange)

# Telegram
if tg_enabled and tg_token and tg_chat_id and not df_sig.empty:
    if not st.session_state.alert_bot:
        st.session_state.alert_bot = TelegramAlert(tg_token, tg_chat_id)
    bot = st.session_state.alert_bot
    for _, row in df_sig[df_sig["Score"] >= tg_min_score].iterrows():
        st.session_state.signal_counter[row["Symbol"]] += 1
        bot.send_signal(row, exchange=exchange_label, timeframe=oi_period,
                        signal_count=st.session_state.signal_counter[row["Symbol"]])

# ═══════════════════════════════════════════════════════════
#  VA + SMC ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════

def _render_va_smc_panel(kl: pd.DataFrame, sym: str, tf: str,
                         obs: list, fvgs: list, tdp_levels: list):
    """Volume Analysis + Smart Money Concepts панель под графиком."""
    c = kl["close"].astype(float)
    h = kl["high"].astype(float)
    l = kl["low"].astype(float)
    v = kl["quote_volume"].astype(float)
    cur = float(c.iloc[-1])

    # ── Volume Profile / Value Area (последние 100 свечей) ──
    seg_c = c.iloc[-100:]
    seg_h = h.iloc[-100:]
    seg_l = l.iloc[-100:]
    seg_v = v.iloc[-100:]
    price_min = float(seg_l.min())
    price_max = float(seg_h.max())
    bins = 40
    edges = [price_min + (price_max - price_min) / bins * i for i in range(bins + 1)]
    vol_hist = [0.0] * bins
    for i in range(len(seg_c)):
        mid = (float(seg_h.iloc[i]) + float(seg_l.iloc[i])) / 2
        bi  = min(int((mid - price_min) / (price_max - price_min + 1e-10) * bins), bins - 1)
        vol_hist[bi] += float(seg_v.iloc[i])
    total_vol = sum(vol_hist) or 1
    poc_idx   = vol_hist.index(max(vol_hist))
    poc_price = (edges[poc_idx] + edges[poc_idx + 1]) / 2
    va_vol = max(vol_hist); va_up = poc_idx; va_dn = poc_idx
    while va_vol < total_vol * 0.70 and (va_up < bins - 1 or va_dn > 0):
        add_up = vol_hist[va_up + 1] if va_up < bins - 1 else 0
        add_dn = vol_hist[va_dn - 1] if va_dn > 0 else 0
        if add_up >= add_dn and va_up < bins - 1:
            va_up += 1; va_vol += add_up
        elif va_dn > 0:
            va_dn -= 1; va_vol += add_dn
        else:
            break
    vah = (edges[va_up] + edges[va_up + 1]) / 2
    val = (edges[va_dn] + edges[va_dn + 1]) / 2

    # ── Market Structure ──
    def _ms_label(arr_h, arr_l):
        highs = []
        lows  = []
        for i in range(2, len(arr_h) - 2):
            if arr_h[i] == max(arr_h[max(0,i-2):i+3]): highs.append((i, arr_h[i]))
            if arr_l[i] == min(arr_l[max(0,i-2):i+3]): lows.append((i, arr_l[i]))
        if len(highs) < 2 or len(lows) < 2: return "Неопределена", "#585b70"
        hh = highs[-1][1] > highs[-2][1]
        hl = lows[-1][1]  > lows[-2][1]
        lh = highs[-1][1] < highs[-2][1]
        ll = lows[-1][1]  < lows[-2][1]
        if hh and hl:  return "Бычья (HH+HL)", "#a6e3a1"
        if lh and ll:  return "Медвежья (LH+LL)", "#f38ba8"
        if hh and ll:  return "Расширение (HH+LL)", "#f9e2af"
        if lh and hl:  return "Сжатие (LH+HL)", "#89b4fa"
        return "Боковик", "#cba6f7"

    h_arr = h.values[-50:]
    l_arr = l.values[-50:]
    ms_label, ms_color = _ms_label(h_arr, l_arr)

    # ── Подсчёт OB/FVG ──
    bull_fvg = [f for f in fvgs if f.kind == "bullish" and f.high > cur * 0.9]
    bear_fvg = [f for f in fvgs if f.kind == "bearish" and f.low  < cur * 1.1]
    bull_ob  = [o for o in obs  if o.kind == "bullish" and o.high > cur * 0.85]
    bear_ob  = [o for o in obs  if o.kind == "bearish" and o.low  < cur * 1.15]

    # Качественные OB (с FVG + BOS)
    hq_bull_ob = [o for o in bull_ob if o.has_fvg and o.has_bos and not o.mitigated]
    hq_bear_ob = [o for o in bear_ob if o.has_fvg and o.has_bos and not o.mitigated]

    # ── Breaker Blocks ──
    breakers   = find_breaker_blocks(kl, obs)
    bull_bb    = [b for b in breakers if b.kind == "bullish"]
    bear_bb    = [b for b in breakers if b.kind == "bearish"]

    # ── Named Liquidity Levels ──
    named_liq = find_named_liquidity_levels(kl)
    bsl_levels = [lv for lv in named_liq if lv.kind in ("PDH","PWH","HOD","EQH") and not lv.swept]
    ssl_levels = [lv for lv in named_liq if lv.kind in ("PDL","PWL","LOD","EQL") and not lv.swept]

    # ── Liquidity sweeps ──
    wicks_up = int(((h - kl["open"].astype(float).clip(upper=c)).clip(lower=0) >
                    (c - l).clip(lower=0) * 1.5).iloc[-20:].sum())
    wicks_dn = int(((kl["open"].astype(float).clip(lower=c) - l).clip(lower=0) >
                    (h - c).clip(lower=0) * 1.5).iloc[-20:].sum())

    # ── RSI Divergence ──
    rsi_divs = find_rsi_divergence(kl)
    bull_divs = [d for d in rsi_divs if "bullish" in d.kind]
    bear_divs = [d for d in rsi_divs if "bearish" in d.kind]

    # ── Killzone ──
    try:
        kz_info = get_killzone_info(kl)
    except Exception:
        kz_info = None

    # ── Swing Points ──
    swings_3 = find_swing_points(kl, strength=3, lookback=60)
    sw_highs = [s for s in swings_3 if s.kind == "high"]
    sw_lows  = [s for s in swings_3 if s.kind == "low"]
    last_sh  = sw_highs[-1].price if sw_highs else None
    last_sl  = sw_lows[-1].price  if sw_lows  else None

    # ── ATR ──
    atr = float((h - l).iloc[-14:].mean())
    atr_pct = atr / cur * 100

    # ── BOS note ──
    bos_note = ""
    highs_list = [h_arr[i] for i in range(2,len(h_arr)-2) if h_arr[i]==max(h_arr[max(0,i-2):i+3])]
    if len(highs_list) >= 2:
        bos_note = "🔼 BOS вверх" if highs_list[-1] > highs_list[-2] else "🔽 BOS вниз"

    # ════════════════════════════════════════════
    #  РЕНДЕР
    # ════════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 📊 Volume Analysis + Smart Money Concepts")

    # Строка 1: VP / Market Structure / Ликвидность
    va1, va2, va3 = st.columns(3)
    with va1:
        st.markdown("**📦 Volume Profile (100 свечей)**")
        poc_pos = "выше цены" if poc_price > cur else "ниже цены"
        in_va   = val <= cur <= vah
        st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.85rem">
<b style="color:#cba6f7">POC</b>: <b>${poc_price:.4f}</b> <span style="color:#585b70">({poc_pos})</span><br>
<b style="color:#a6e3a1">VAH</b>: <b>${vah:.4f}</b> &nbsp;|&nbsp; <b style="color:#f38ba8">VAL</b>: <b>${val:.4f}</b><br>
<hr style="border-color:#313244;margin:5px 0">
Цена {'<b style="color:#a6e3a1">внутри VA</b>' if in_va else '<b style="color:#f9e2af">вне VA</b>'} (70% объёма)<br>
<span style="color:#585b70;font-size:.8rem">ATR: ${atr:.4f} ({atr_pct:.1f}%/свеча)</span>
</div>""", unsafe_allow_html=True)

    with va2:
        st.markdown("**🏗️ Market Structure**")
        sh_txt = f"${last_sh:.4f}" if last_sh else "—"
        sl_txt = f"${last_sl:.4f}" if last_sl else "—"
        st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.85rem">
<b style="color:{ms_color}">{ms_label}</b> &nbsp; <span style="color:#cdd6f4">{bos_note}</span><br>
Свинг Хай: <b>${sh_txt}</b> &nbsp;|&nbsp; Свинг Лоу: <b>${sl_txt}</b><br>
<hr style="border-color:#313244;margin:5px 0">
🟢 Бычьи OB: <b>{len(bull_ob)}</b> (качест.: <b style="color:#a6e3a1">{len(hq_bull_ob)}</b>)<br>
🔴 Медвежьи OB: <b>{len(bear_ob)}</b> (качест.: <b style="color:#f38ba8">{len(hq_bear_ob)}</b>)<br>
📊 FVG бычьи: <b>{len(bull_fvg)}</b> &nbsp;|&nbsp; медвежьи: <b>{len(bear_fvg)}</b><br>
🔷 Breaker бычий: <b>{len(bull_bb)}</b> &nbsp;|&nbsp; медвежий: <b>{len(bear_bb)}</b>
</div>""", unsafe_allow_html=True)

    with va3:
        st.markdown("**💧 Ликвидность**")
        liq_bias = "🐂 Больше свипов лонг-ликв." if wicks_dn > wicks_up else (
                   "🐻 Больше свипов шорт-ликв." if wicks_up > wicks_dn else "⚖️ Баланс")
        nearest_ob_obj = None
        if bull_ob:
            nearest_ob_obj = min(bull_ob, key=lambda o: abs((o.high+o.low)/2 - cur))
        elif bear_ob:
            nearest_ob_obj = min(bear_ob, key=lambda o: abs((o.high+o.low)/2 - cur))
        ob_txt = ""
        if nearest_ob_obj:
            ob_dist = abs((nearest_ob_obj.high+nearest_ob_obj.low)/2 - cur) / cur * 100
            quality = " ✅+FVG+BOS" if nearest_ob_obj.has_fvg and nearest_ob_obj.has_bos else ""
            ob_txt = (f"Ближайший {'🟢' if nearest_ob_obj.kind=='bullish' else '🔴'} OB: "
                      f"<b>${nearest_ob_obj.low:.4f}–${nearest_ob_obj.high:.4f}</b> "
                      f"({ob_dist:.1f}%{quality})")
        bsl_txt = " | ".join(f"<b style='color:#f38ba8'>{lv.kind}</b> ${lv.price:.4f}" for lv in bsl_levels[:2])
        ssl_txt = " | ".join(f"<b style='color:#a6e3a1'>{lv.kind}</b> ${lv.price:.4f}" for lv in ssl_levels[:2])
        st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.85rem">
<b>{liq_bias}</b><br>
Свипов лонг (последние 20): <b>{wicks_dn}</b> &nbsp;|&nbsp; шорт: <b>{wicks_up}</b><br>
<hr style="border-color:#313244;margin:5px 0">
{ob_txt or "OB рядом не найден"}<br>
BSL пулы: {bsl_txt or "—"}<br>
SSL пулы: {ssl_txt or "—"}
</div>""", unsafe_allow_html=True)

    # Строка 2: RSI Дивергенция / Killzone / Named Liquidity
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**📉 RSI Дивергенция**")
        div_html = ""
        for d in bull_divs[-2:]:
            label = "Бычья" if d.kind == "bullish_classic" else "Скрытая бычья"
            div_html += f'<span style="color:#a6e3a1">⬆ {label}</span>: цена {d.price1:.4f}→{d.price2:.4f}, RSI {d.rsi1}→{d.rsi2}<br>'
        for d in bear_divs[-2:]:
            label = "Медвежья" if d.kind == "bearish_classic" else "Скрытая медвежья"
            div_html += f'<span style="color:#f38ba8">⬇ {label}</span>: цена {d.price1:.4f}→{d.price2:.4f}, RSI {d.rsi1}→{d.rsi2}<br>'
        st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.82rem">
{div_html or '<span style="color:#585b70">Дивергенций не найдено</span>'}
</div>""", unsafe_allow_html=True)

    with d2:
        st.markdown("**⏰ Сессии & Killzone**")
        if kz_info:
            zone_color = "#a6e3a1" if "London" in kz_info.active_zone or "New York" in kz_info.active_zone else (
                         "#f9e2af" if "Asia" in kz_info.active_zone else "#585b70")
            asia_rng = ""
            if kz_info.asia_range_high > 0:
                asia_rng = (f"Asia Range: ${kz_info.asia_range_low:.4f}–${kz_info.asia_range_high:.4f}<br>"
                            f"<span style='color:#585b70;font-size:.78rem'>BSL выше ренджа, SSL ниже</span>")
            st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.82rem">
<b style="color:{zone_color}">⏱ {kz_info.active_zone}</b>
<span style="color:#585b70"> ({kz_info.current_utc})</span><br>
<hr style="border-color:#313244;margin:5px 0">
{asia_rng or '<span style="color:#585b70">Азиатский рендж не определён</span>'}<br>
<span style="color:#585b70;font-size:.78rem">
London KZ: 09:00–13:00 | NY KZ: 15:00–18:00 | LC: 20:00–22:00 (Киев)
</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.82rem;color:#585b70">Killzone недоступна</div>', unsafe_allow_html=True)

    with d3:
        st.markdown("**🎯 Named Liquidity Levels**")
        lev_rows = ""
        for lv in sorted(named_liq, key=lambda x: abs(x.price - cur))[:6]:
            dist = (lv.price - cur) / cur * 100
            swept_mark = " ✓" if lv.swept else ""
            color = "#f38ba8" if dist > 0 else "#a6e3a1"
            kind_icon = "↑" if dist > 0 else "↓"
            lev_rows += (f'<span style="color:{color}">{kind_icon} <b>{lv.kind}</b></span> '
                         f'${lv.price:.4f} <span style="color:#585b70">({dist:+.1f}%{swept_mark})</span><br>')
        st.markdown(f"""
<div style="background:#1e1e2e;border-radius:8px;padding:10px 14px;font-size:.82rem">
{lev_rows or '<span style="color:#585b70">Уровни не найдены</span>'}
</div>""", unsafe_allow_html=True)

    # TDP уровни
    if tdp_levels:
        near_tdp = sorted(tdp_levels, key=lambda t: abs(t.price - cur))[:4]
        tdp_html = " &nbsp;|&nbsp; ".join(
            f'<span style="color:#f9e2af">{t.kind}</span> <b>${t.price:.4f}</b>'
            f' <span style="color:#585b70">({(t.price-cur)/cur*100:+.1f}%)</span>'
            for t in near_tdp)
        st.markdown(
            f'<div style="background:#1e1e2e;border-radius:6px;padding:7px 14px;'
            f'font-size:.82rem;margin-top:6px">⚡ <b>TDP уровни:</b> {tdp_html}</div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ФУНКЦИЯ ПОЛНОГО АНАЛИЗА (должна быть до вкладок)
# ═══════════════════════════════════════════════════════════

def _render_pattern_detail(ph: dict, exchange: str, oi_period: str, mkt: dict, df_sig: pd.DataFrame):
    """
    Показывает чарт с выделенным паттерном + трейдинговый + инвестиционный анализ.
    ph — словарь строки из таблицы scan_df.
    """
    sym      = ph.get("Монета","")
    pat_name = ph.get("Паттерн","")
    pat_dir  = ph.get("Направление","")
    pat_conf = ph.get("Уверенность","")
    pat_desc = ph.get("Описание","")
    pat_conf_num = int(pat_conf.replace("%","")) if pat_conf else 0

    st.markdown("---")
    dir_color = "#a6e3a1" if "Лонг" in pat_dir else ("#f38ba8" if "Шорт" in pat_dir else "#f9e2af")
    confirmed_mark = '✅' if ph.get('Подтверждён') == '✅' else '⏳'

    st.markdown(
        f"""<div style="background:#1e1e2e;border-left:4px solid {dir_color};
        border-radius:6px;padding:8px 16px;margin:6px 0;display:flex;align-items:center;gap:16px">
        <span style="font-size:1.1rem;font-weight:700;color:{dir_color}">{pat_dir} {pat_name}</span>
        <span style="color:#cdd6f4"><b>{sym}</b></span>
        <span style="color:#585b70">·</span>
        <span style="color:{dir_color}"><b>{pat_conf}</b></span>
        <span style="color:#585b70">·</span>
        <span>{confirmed_mark} {'Подтверждён' if confirmed_mark == '✅' else 'Ожидание'}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Кнопки действий
    scan_tf = ph.get("ТФ", "1h")  # берём ТФ из результатов сканера
    ba1,ba2,ba3,ba4 = st.columns(4)
    if ba1.button(f"📈 На главный график", key="ph_chart"):
        st.session_state.chart_sym = sym
        st.session_state.chart_tf  = scan_tf
        st.rerun()
    show_trade = ba2.button("🎯 Трейдинг-анализ", key="ph_trade")
    show_invest = ba3.button("💼 Инвест-анализ",  key="ph_invest")
    if ba4.button("✖ Закрыть", key="ph_close"):
        st.session_state.pat_highlight    = None
        st.session_state.inv_analysis_sym = None
        st.rerun()

    # ── Загружаем данные ──
    @st.cache_data(ttl=30)
    def _ph_klines(s, tf, ex): return get_klines(s, tf, 300, exchange=ex)
    @st.cache_data(ttl=60)
    def _ph_oi(s, op, ex):     return get_oi_history(s, op, 60, exchange=ex)

    kl   = _ph_klines(sym, scan_tf, exchange)
    oi_h = _ph_oi(sym, oi_period, exchange)

    # ── Паттерн на чарте — key_levels + key_points (фигура) ──
    pat_hlines = []
    pat_shapes = []
    colors_lvl = ["#f9e2af","#89b4fa","#a6e3a1","#f38ba8","#cba6f7"]
    shape_color = "#f38ba8" if "Шорт" in pat_dir or "↓" in pat_name else "#a6e3a1"

    if not kl.empty:
        # SMC сетапы передают _kp и _kl напрямую
        # После r_df.iloc[n].to_dict() отсутствующие колонки становятся NaN (float) — фильтруем
        _ext_kl = ph.get("_kl")
        _ext_kp = ph.get("_kp")
        if not isinstance(_ext_kl, (list, tuple)):
            _ext_kl = None
        if not isinstance(_ext_kp, (list, tuple)):
            _ext_kp = None

        if _ext_kl:
            for ci, lvl in enumerate(_ext_kl):
                try:
                    fv = float(lvl)
                    if fv > 0:
                        pat_hlines.append({
                            "price": fv,
                            "color": colors_lvl[ci % len(colors_lvl)],
                            "label": f"{pat_name} L{ci+1}",
                        })
                except (TypeError, ValueError):
                    pass
        if _ext_kp and len(_ext_kp) >= 2:
            shape_pts = []
            for (idx, price) in _ext_kp:
                if 0 <= idx < len(kl):
                    shape_pts.append({
                        "price": float(price),
                        "time":  int(kl["timestamp"].iloc[idx].timestamp()),
                    })
            if len(shape_pts) >= 2:
                pat_shapes.append({"points": shape_pts, "color": shape_color, "label": pat_name})

        # Паттерны из patterns.py
        if not _ext_kp:
            try:
                from screeners.patterns import scan_patterns as _sp
                found = [p for p in _sp(kl) if p.name == pat_name]
                if found:
                    p_obj = found[0]
                    for ci, lvl in enumerate(p_obj.key_levels or []):
                        if lvl and lvl > 0:
                            pat_hlines.append({
                                "price": float(lvl),
                                "color": colors_lvl[ci % len(colors_lvl)],
                                "label": f"{pat_name} L{ci+1}",
                            })
                    if p_obj.key_points:
                        sc = "#f38ba8" if "bearish" in p_obj.direction else "#a6e3a1"
                        shape_pts = []
                        for kp in p_obj.key_points:
                            idx = kp.get("idx", -1)
                            if 0 <= idx < len(kl):
                                shape_pts.append({
                                    "price": float(kp["price"]),
                                    "time":  int(kl["timestamp"].iloc[idx].timestamp()),
                                })
                        if len(shape_pts) >= 2:
                            pat_shapes.append({"points": shape_pts, "color": sc, "label": pat_name})
            except Exception:
                pass

    # SMC оверлеи
    obs_v  = find_order_blocks(kl) if not kl.empty else []
    fvgs_v = find_fvg(kl)          if not kl.empty else []
    tdp_v  = get_tdp_levels(kl)    if not kl.empty else []

    st.markdown(f"#### 📈 {sym} · {scan_tf} — паттерн **{pat_name}** на графике &nbsp; <span style='font-size:.8rem;color:#585b70'>📸 Скриншот · ⛶ Полный экран · Log шкала — в тулбаре графика</span>", unsafe_allow_html=True)
    render_tv_chart(
        kl, sym, scan_tf, height=680,
        overlays={
            "obs":            obs_v,
            "fvgs":           fvgs_v,
            "tdp":            tdp_v,
            "h_lines":        pat_hlines + st.session_state.h_lines,
            "pattern_shapes": pat_shapes,
        },
        show_cvd=True, show_oi=oi_h,
    )

    # ── VA + SMC АНАЛИТИКА ──
    if not kl.empty:
        _render_va_smc_panel(kl, sym, scan_tf, obs_v, fvgs_v, tdp_v)

    # ── ТРЕЙДИНГ-АНАЛИЗ ──
    row = df_sig[df_sig["Symbol"] == sym] if not df_sig.empty else pd.DataFrame()
    if not row.empty and (show_trade or pat_conf_num >= 65):
        st.markdown("#### 🎯 Трейдинговый анализ")
        try:
            k1h = get_klines(sym, "1h", 200, exchange=exchange)
            rep = generate_analysis(sym, kl, k1h, row.iloc[0], mkt)
            vc  = {"✅ ВХОДИМ":"#a6e3a1","⏳ ЖДЁМ":"#f9e2af","🚫 ПРОПУСКАЕМ":"#f38ba8"}.get(rep.verdict,"#cdd6f4")
            cb  = "█"*(rep.confidence_pct//10)+"░"*(10-rep.confidence_pct//10)
            fmt = lambda x: f"${x:.4f}" if x < 10 else f"${x:.2f}"
            st.markdown(
                f"""<div style="background:#1e1e2e;border:1px solid {vc};
                border-radius:8px;padding:10px 16px;margin:6px 0">
                <b style="color:{vc}">{rep.verdict}</b>
                &nbsp; Уверенность <b>{rep.confidence_pct}%</b>
                <code style="color:{vc}">{cb}</code>
                &nbsp; R:R <b>1:{rep.rr_ratio}</b>
                &nbsp; Направление <b style="color:{vc}">{rep.direction.upper()}</b>
                </div>""", unsafe_allow_html=True,
            )
            tc1,tc2 = st.columns(2)
            with tc1:
                if rep.factors_ok:
                    st.markdown("**✅ За**")
                    for f in rep.factors_ok: st.markdown(f"&nbsp;&nbsp;{f}")
                if rep.factors_no:
                    st.markdown("**❌ Против**")
                    for f in rep.factors_no: st.markdown(f"&nbsp;&nbsp;{f}")
            with tc2:
                entry_mid = (rep.entry_zone[0]+rep.entry_zone[1])/2
                plan = pd.DataFrame([
                    ("Зона входа",  f"{fmt(rep.entry_zone[0])} — {fmt(rep.entry_zone[1])}"),
                    ("Стоп-лосс",   f"{fmt(rep.stop_loss)} ({(rep.stop_loss/entry_mid-1)*100:+.1f}%)"),
                    ("Цель 1",      f"{fmt(rep.take_profit_1)} ({(rep.take_profit_1/entry_mid-1)*100:+.1f}%)"),
                    ("Цель 2",      f"{fmt(rep.take_profit_2)} ({(rep.take_profit_2/entry_mid-1)*100:+.1f}%)"),
                    ("R/R",         f"1:{rep.rr_ratio}"),
                ], columns=["Параметр","Значение"])
                st.dataframe(plan, use_container_width=True, hide_index=True, height=200)
        except Exception as e:
            st.warning(f"Трейдинг-анализ недоступен: {e}")

    # ── ИНВЕСТ-АНАЛИЗ ──
    if show_invest:
        _render_investment_panel(sym)


def _render_investment_panel(sym: str):
    """Инвестиционный профиль монеты."""
    st.markdown("#### 💼 Инвестиционный анализ")
    with st.spinner(f"Загружаю инвестиционный профиль {sym}..."):
        @st.cache_data(ttl=600)
        def _inv(s): return get_investment_profile(s)
        inv = _inv(sym)

    if "error" in inv:
        st.warning(inv["error"])
        return

    fmt_m = lambda x: (f"${x/1e9:.2f}B" if x >= 1e9 else
                        f"${x/1e6:.1f}M" if x >= 1e6 else
                        f"${x:,.0f}")
    fmt_p = lambda x: (f"${x:,.4f}" if x < 10 else
                        f"${x:,.2f}" if x < 1000 else
                        f"${x:,.0f}")
    fmt_n = lambda x: (f"{x/1e9:.1f}B" if x >= 1e9 else
                        f"{x/1e6:.1f}M" if x >= 1e6 else
                        f"{x:,.0f}")
    chg_c = lambda v: "#a6e3a1" if v >= 0 else "#f38ba8"

    md  = inv["market"]
    tok = inv["tokenomics"]
    com = inv["community"]
    dev = inv["dev"]
    unl = inv["unlock"]
    hp  = inv["holders_proxy"]
    lnk = inv["links"]

    # ── Рыночные метрики ──
    st.markdown("##### 📊 Рыночные метрики")
    mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
    mc1.metric("Цена",        fmt_p(md["price"]))
    mc2.metric("Мар. кап",    fmt_m(md["market_cap"]))
    mc3.metric("FDV",         fmt_m(md["fdv"]) if md["fdv"] else "–")
    mc4.metric("Объём 24ч",   fmt_m(md["volume_24h"]))
    mc5.metric("Рейтинг MC",  f"#{md['mc_rank']}" if md["mc_rank"] else "–")
    mc6.metric("Vol/MC",      f"{md['vol_mc_ratio']*100:.1f}%" if md["vol_mc_ratio"] else "–",
               help="Отношение объёма к капитализации. >10% = хорошая ликвидность.")

    # Ценовые изменения
    pc1,pc2,pc3,pc4,pc5 = st.columns(5)
    pc1.metric("7д",  f"{md['change_7d']:+.1f}%"  if md['change_7d']  else "–")
    pc2.metric("30д", f"{md['change_30d']:+.1f}%"  if md['change_30d'] else "–")
    pc3.metric("1 год",f"{md['change_1y']:+.1f}%"  if md['change_1y']  else "–")
    pc4.metric("ATH", fmt_p(md["ath"]), delta=f"{md['ath_change']:+.1f}%" if md['ath_change'] else None,
               help=f"Исторический максимум {md['ath_date']}")
    pc5.metric("ATL", fmt_p(md["atl"]), delta=f"{md['atl_change']:+.1f}%" if md['atl_change'] else None,
               help=f"Исторический минимум {md['atl_date']}")

    # ── Токеномика ──
    st.markdown("##### 🏦 Токеномика и риск разлоков")
    tk1,tk2,tk3 = st.columns(3)
    tk1.metric("В обращении", fmt_n(tok["circ_supply"]) if tok["circ_supply"] else "–")
    tk2.metric("Total Supply", fmt_n(tok["total_supply"]) if tok["total_supply"] else "–")
    tk3.metric("Max Supply",  fmt_n(tok["max_supply"])   if tok["max_supply"]  else "∞")

    if tok["circ_ratio"] is not None:
        # Визуальная полоса разлока
        pct = int(tok["circ_ratio"] * 100)
        col_ = "#a6e3a1" if pct >= 70 else ("#f9e2af" if pct >= 40 else "#f38ba8")
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        st.markdown(
            f"<div style='font-family:monospace;color:{col_}'>"
            f"В обращении {pct}% &nbsp;<code>{bar}</code></div>",
            unsafe_allow_html=True,
        )

    if unl["assessment"]:  st.info(unl["assessment"])
    if unl["fdv_note"]:    st.warning(unl["fdv_note"])
    if unl["ath_note"]:    st.caption(unl["ath_note"])

    st.caption(f"📋 Точный вестинг: [TokenUnlocks.io](https://token.unlocks.app) · [Vestlab](https://vestlab.io) · [Cryptorank](https://cryptorank.io)")

    if tok["genesis_date"] and tok["genesis_date"] != "Неизвестно":
        st.caption(f"📅 Genesis date: {tok['genesis_date']}")
    if tok["categories"]:
        st.caption(f"🏷️ Категории: {', '.join(tok['categories'])}")

    # ── Держатели (прокси) ──
    st.markdown("##### 👥 Держатели и ликвидность")
    for note in hp.get("notes", []):
        st.markdown(note)
    st.caption(f"🔍 Точные данные: {hp.get('disclaimer','')}")

    # ── Сообщество ──
    st.markdown("##### 📡 Сообщество")
    cm1,cm2,cm3,cm4 = st.columns(4)
    cm1.metric("Twitter",   f"{com['twitter_followers']:,}"  if com['twitter_followers']  else "–")
    cm2.metric("Reddit",    f"{com['reddit_subscribers']:,}" if com['reddit_subscribers'] else "–")
    cm3.metric("Telegram",  f"{com['telegram_users']:,}"     if com['telegram_users']     else "–")
    cm4.metric("👍 Sentiment", f"{com['sentiment_up'] or 0:.0f}%")

    # ── Dev активность ──
    if dev.get("commits_4w") or dev.get("stars"):
        st.markdown("##### 👨‍💻 Dev активность")
        dv1,dv2,dv3,dv4 = st.columns(4)
        dv1.metric("Коммитов 4н",  dev.get("commits_4w") or 0,
            help="Активность разработчиков. >50/нед = активный проект.")
        dv2.metric("GitHub ★",     dev.get("stars") or 0)
        dv3.metric("Forks",        dev.get("forks") or 0)
        dv4.metric("Contributors", dev.get("contributors") or 0)

    # ── Ссылки ──
    links_md = []
    if lnk.get("homepage"):    links_md.append(f"[🌐 Сайт]({lnk['homepage']})")
    if lnk.get("whitepaper"):  links_md.append(f"[📄 Whitepaper]({lnk['whitepaper']})")
    if dev.get("github"):      links_md.append(f"[💻 GitHub]({dev['github']})")
    if com.get("twitter"):     links_md.append(f"[🐦 Twitter](https://twitter.com/{com['twitter']})")
    if links_md:
        st.markdown("##### 🔗 Ссылки")
        st.markdown("  ·  ".join(links_md))

    # ── Описание ──
    if inv.get("description"):
        with st.expander("📖 О проекте"):
            st.markdown(inv["description"])

    # ── Вывод для инвестора ──
    st.markdown("##### 🧭 Итоговый инвест-вывод")
    score_inv = 0
    reasons   = []
    warns     = []

    mc = md["market_cap"]
    if mc > 1e9:       score_inv += 1; reasons.append("✅ Капитализация >$1B — ликвидный актив")
    elif mc < 50e6:    warns.append("⚠️ Малая капитализация — высокий риск")

    if tok["circ_ratio"] and tok["circ_ratio"] > 0.7:
        score_inv += 1; reasons.append("✅ >70% токенов в обращении — низкий риск разлоков")
    elif tok["circ_ratio"] and tok["circ_ratio"] < 0.4:
        warns.append("⚠️ <40% в обращении — риск инфляции от разлоков")

    if tok["fdv_mc_ratio"] and tok["fdv_mc_ratio"] < 2:
        score_inv += 1; reasons.append("✅ FDV/MC < 2x — умеренное будущее предложение")
    elif tok["fdv_mc_ratio"] and tok["fdv_mc_ratio"] > 5:
        warns.append("⚠️ FDV/MC > 5x — высокое инфляционное давление")

    if md["vol_mc_ratio"] and md["vol_mc_ratio"] > 0.05:
        score_inv += 1; reasons.append("✅ Хорошая ликвидность")

    if dev.get("commits_4w", 0) > 20:
        score_inv += 1; reasons.append("✅ Активная разработка (>20 коммитов/4нед)")
    elif dev.get("commits_4w", 0) == 0:
        warns.append("⚠️ Нет активности разработчиков")

    if com["twitter_followers"] > 100_000:
        score_inv += 1; reasons.append(f"✅ Сильное сообщество ({com['twitter_followers']:,} Twitter)")

    inv_rating = ["⚫ Нет данных","🔴 Высокий риск","🔴 Рискованно",
                   "🟠 Умеренный риск","🟡 Средний","🟢 Хороший","🟢 Отличный"][min(score_inv,6)]

    st.markdown(
        f"""<div style="background:#1e1e2e;border-radius:10px;padding:12px 18px;margin:8px 0">
        <b>Рейтинг инвестора: {inv_rating}</b> ({score_inv}/6 баллов)<br>
        <span style="font-size:.85rem;color:#585b70">Только технический анализ данных — не финансовый совет.</span>
        </div>""", unsafe_allow_html=True,
    )
    for r in reasons: st.markdown(r)
    for w in warns:   st.markdown(w)


def _full_analysis_panel(sym, df_sig, exchange, oi_period, mkt, chart_tf="15m"):
    """Рендерит развёрнутый анализ под таблицей сигналов."""
    row = df_sig[df_sig["Symbol"] == sym].iloc[0]
    anal_tf, higher_tf = TF_HIERARCHY.get(chart_tf, ("15m","1h"))

    st.markdown("---")
    with st.spinner(f"Загружаю данные и анализирую {sym} ({anal_tf}/{higher_tf})..."):
        k_low  = get_klines(sym, anal_tf,  200, exchange=exchange)
        k_high = get_klines(sym, higher_tf, 200, exchange=exchange)
        oi_h   = get_oi_history(sym, oi_period, 60, exchange=exchange)
        report = generate_analysis(sym, k_low, k_high, row, mkt)

    verdict_colors = {
        "✅ ВХОДИМ": "#a6e3a1",
        "⏳ ЖДЁМ":   "#f9e2af",
        "🚫 ПРОПУСКАЕМ": "#f38ba8",
    }
    vc = verdict_colors.get(report.verdict, "#cdd6f4")
    cb = "█"*(report.confidence_pct//10) + "░"*(10-report.confidence_pct//10)
    fmt = lambda x: f"${x:.4f}" if x < 10 else f"${x:.2f}"

    st.markdown(
        f"""<div style="background:#1e1e2e;border:2px solid {vc};
        border-radius:10px;padding:14px 18px;margin:10px 0">
        <span style="font-size:1.35rem;font-weight:700;color:{vc}">{report.verdict}</span>
        &nbsp;&nbsp;&nbsp;
        <span style="color:#cdd6f4">
        {sym} · {anal_tf}/{higher_tf} · Уверенность
        <b>{report.confidence_pct}%</b>
        <code style="color:{vc}">{cb}</code>
        · Направление: <b style="color:{vc}">{report.direction.upper()}</b>
        · R:R ≈ <b>1:{report.rr_ratio}</b>
        </span>
        </div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        if report.factors_ok:
            st.markdown("#### ✅ Подтверждающие")
            for f in report.factors_ok: st.markdown(f"&nbsp;&nbsp;{f}")
        if report.factors_warn:
            st.markdown("#### ⚠️ Требует внимания")
            for f in report.factors_warn: st.markdown(f"&nbsp;&nbsp;{f}")
        if report.factors_no:
            st.markdown("#### ❌ Исключаем")
            for f in report.factors_no: st.markdown(f"&nbsp;&nbsp;{f}")

    with right:
        st.markdown("#### 📍 Торговый план")
        entry_mid = (report.entry_zone[0] + report.entry_zone[1]) / 2
        plan = pd.DataFrame([
            ("Зона входа",  f"{fmt(report.entry_zone[0])} — {fmt(report.entry_zone[1])}"),
            ("Стоп-лосс",   f"{fmt(report.stop_loss)} ({(report.stop_loss/entry_mid-1)*100:+.1f}%)"),
            ("Цель 1",      f"{fmt(report.take_profit_1)} ({(report.take_profit_1/entry_mid-1)*100:+.1f}%)"),
            ("Цель 2",      f"{fmt(report.take_profit_2)} ({(report.take_profit_2/entry_mid-1)*100:+.1f}%)"),
            ("Risk/Reward", f"1:{report.rr_ratio}"),
        ], columns=["Параметр","Значение"])
        st.dataframe(plan, use_container_width=True, hide_index=True, height=200)

        mc1,mc2 = st.columns(2)
        mc1.metric("Цена", fmt(float(row["Price"])))
        mc2.metric("OI Δ", f"{float(row['OI_Change']):+.1f}%")
        mc3,mc4 = st.columns(2)
        mc3.metric("Изм. 24ч", f"{float(row['Change24h']):+.2f}%")
        mc4.metric("CVD", str(row["CVD"]))

    if not k_low.empty:
        st.markdown(f"#### 📈 {sym} · {anal_tf} — SMC + Паттерны + SL/TP")
        render_tv_chart(
            k_low, sym, anal_tf, height=600,
            overlays={
                "obs":    report.order_blocks,
                "fvgs":   report.fvgs,
                "hunts":  report.stop_hunts,
                "tdp":    report.tdp_levels,
                "wave_pivots": report.wave_result.get("pivots", []),
                "fibs":   report.wave_result.get("fib_levels", {}),
                "entry":  {"low": report.entry_zone[0], "high": report.entry_zone[1]},
                "sl":     report.stop_loss,
                "tp1":    report.take_profit_1,
                "tp2":    report.take_profit_2,
                "h_lines": st.session_state.h_lines,
            },
            show_cvd=True, show_oi=oi_h,
        )

    if st.button("✖ Закрыть анализ", key="close_an"):
        st.session_state.analysis_sym = None
        st.rerun()


# ═══════════════════════════════════════════════════════════
#  ВКЛАДКИ
# ═══════════════════════════════════════════════════════════
tab_sig, tab_chart, tab_wave, tab_pred, tab_sent, tab_stat = st.tabs([
    "🔥 Сигналы", "📈 График", "🌊 Волны",
    "🔮 Предсказание", "📡 Сентимент", "📊 Статистика",
])

# ───────────────────────────────────────────────────────────
#  TAB 1 — СИГНАЛЫ + АНАЛИЗ
# ───────────────────────────────────────────────────────────
with tab_sig:
    if df_sig.empty:
        try:
            from data.binance import get_last_error
            _err = get_last_error()
        except Exception:
            _err = ""
        st.error("❌ Нет данных от Binance API.")
        if _err:
            st.code(_err, language="text")
        else:
            st.info("Проверьте соединение или снизьте пороги фильтров в сайдбаре.")
        st.stop()

    # Фильтры
    fc1,fc2,fc3,fc4,fc5 = st.columns(5)
    with fc1:
        min_score_f   = st.selectbox("Мин. Score", [0,1,2,3,4,5], 0,
            help="Фильтр по силе скринерного сигнала.")
    with fc2:
        dir_f = st.selectbox("Направление",
            ["Все","↑ Лонг","↓ Шорт"],
            help="Фильтр по ценовому направлению за 24ч.")
    with fc3:
        sort_f = st.selectbox("Сортировка",
            ["Score","OI_Change","Change24h","Volume24h"],
            help="Поле для сортировки.")
    with fc4:
        search_f = st.text_input("🔍 Поиск", placeholder="SOL...",
            help="Часть тикера.")
    with fc5:
        conf_filter = st.number_input("Уверенность ≥ %", 0, 95, min_confidence, 5,
            help="Только пары где предполагаемый уверенность ≥ X%. "
                 "Значение > 0 значительно замедляет обновление.")

    view = df_sig.copy()
    if min_score_f:     view = view[view["Score"] >= min_score_f]
    if "Лонг" in dir_f: view = view[view["Change24h"] > 0]
    if "Шорт" in dir_f: view = view[view["Change24h"] < 0]
    if search_f:        view = view[view["Symbol"].str.contains(search_f.upper())]
    view = view.sort_values(sort_f, ascending=False)

    # Быстрый скор уверенности (приближение без полного анализа)
    def _quick_conf(row) -> int:
        """Быстрая оценка без полного анализа."""
        s = int(row["Score"])
        if row["Change24h"] > 0 and row["CVD"] == "↑":
            s += 1
        if usdt_d < bull_t:
            s += 1
        return min(int(s / 7 * 100), 95)

    view["Conf%"] = view.apply(_quick_conf, axis=1)
    if conf_filter > 0:
        view = view[view["Conf%"] >= conf_filter]

    # Метрики
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Монет в скане",     len(df_sig))
    s2.metric("После фильтра",     len(view))
    s3.metric("Score 3+",          len(df_sig[df_sig["Score"] >= 3]))
    s4.metric("↑ Лонг",            len(df_sig[df_sig["Change24h"] > 0]))
    s5.metric("Средняя Conf%",     f"{int(view['Conf%'].mean())}%" if not view.empty else "–")

    # Таблица
    disp = view[["Symbol","Price","Change24h","Volume24h",
                 "OI_Change","CVD","Funding","Score","Conf%","Signals"]].copy()
    disp["Price"]     = disp["Price"].apply(lambda x: f"${x:,.4f}" if x<10 else f"${x:,.2f}")
    disp["Change24h"] = disp["Change24h"].apply(lambda x: f"{x:+.2f}%")
    disp["Volume24h"] = disp["Volume24h"].apply(lambda x: f"${x/1e6:.1f}M")
    disp["OI_Change"] = disp["OI_Change"].apply(lambda x: f"{x:+.1f}%")
    disp["Funding"]   = disp["Funding"].apply(lambda x: f"{x:+.4f}%" if pd.notna(x) else "–")
    disp["Score"]     = disp["Score"].apply(lambda x: "⭐"*int(x) if x>0 else "–")
    disp["Conf%"]     = disp["Conf%"].apply(lambda x: f"{x}%")
    disp = disp.rename(columns={
        "Change24h":"Цена 24ч","Volume24h":"Объём 24ч",
        "OI_Change":"ОИ Δ","Funding":"Фандинг","Signals":"Сигналы",
    })

    event = st.dataframe(
        disp, use_container_width=True, height=380, hide_index=True,
        selection_mode="single-row", on_select="rerun", key="sig_df",
    )
    sel = event.selection.rows if hasattr(event, "selection") else []

    col_dl, col_an_select = st.columns([2,3])
    with col_dl:
        st.download_button("⬇️ CSV",
            view.to_csv(index=False).encode(), f"signals_{datetime.now().strftime('%H%M')}.csv", "text/csv")
    with col_an_select:
        an_sym = st.selectbox("🎯 Открыть анализ →",
            ["– выбрать –"] + view["Symbol"].tolist(), key="an_sym_sel",
            help="Выберите монету ИЛИ кликните строку в таблице выше.")

    # Определяем что анализировать
    if sel and not view.empty:
        st.session_state.analysis_sym = view.iloc[sel[0]]["Symbol"]
    elif an_sym != "– выбрать –":
        st.session_state.analysis_sym = an_sym

    # ── ПОЛНЫЙ АНАЛИЗ ─────────────────────────────────────
    if st.session_state.analysis_sym and st.session_state.analysis_sym in df_sig["Symbol"].values:
        _full_analysis_panel(
            st.session_state.analysis_sym,
            df_sig, exchange, oi_period, mkt,
            st.session_state.chart_tf,
        )

# ───────────────────────────────────────────────────────────
#  TAB 2 — ГРАФИК
# ───────────────────────────────────────────────────────────
with tab_chart:
    if df_sig.empty:
        st.info("Нет данных.")
    else:
        gc1, gc2, gc3, gc4 = st.columns([3, 2, 1, 1])
        with gc1:
            sym_opts = df_sig.head(40)["Symbol"].tolist()
            default_sym = st.session_state.chart_sym or sym_opts[0]
            if default_sym not in sym_opts: default_sym = sym_opts[0]
            sel_sym = st.selectbox("Монета", sym_opts,
                index=sym_opts.index(default_sym), key="chart_sym_sel",
                help="Выберите монету для детального графика.")
            st.session_state.chart_sym = sel_sym
        with gc2:
            tf_default = st.session_state.chart_tf
            if tf_default not in ALL_TIMEFRAMES: tf_default = "15m"
            sel_tf = st.selectbox(
                "Таймфрейм",
                ALL_TIMEFRAMES,
                index=ALL_TIMEFRAMES.index(tf_default),
                key="chart_tf_sel",
                help="Таймфрейм свечей. Анализ обновляется автоматически при смене ТФ.",
            )
            st.session_state.chart_tf = sel_tf
        with gc3:
            chart_limit = st.select_slider("Свечей", [50,100,200,300,500,1000], 300,
                help="Количество свечей. 300-500 = хороший баланс.")
        with gc4:
            chart_h = st.select_slider("Высота", [500,600,700,800,900], 700,
                help="Высота графика в пикселях.")

        # Инструменты рисования
        with st.expander("🖊️ Инструменты рисования", expanded=False):
            dr1, dr2, dr3 = st.columns(3)
            with dr1:
                st.markdown("**Горизонтальная линия**")
                hl_price = st.number_input("Цена", 0.0, format="%.6f", key="hl_p",
                    help="Уровень поддержки или сопротивления.")
                hl_color = st.color_picker("Цвет", "#FFD700", key="hl_c")
                hl_lbl   = st.text_input("Подпись", "S/R", key="hl_lbl")
                if st.button("➕ Добавить", key="add_hl"):
                    if hl_price > 0:
                        st.session_state.h_lines.append(
                            {"price": hl_price, "color": hl_color, "label": hl_lbl})
            with dr2:
                st.markdown("**Быстрые уровни**")
                sym_r = df_sig[df_sig["Symbol"] == sel_sym]
                cur_p = float(sym_r["Price"].iloc[0]) if not sym_r.empty else 0.0
                if cur_p > 0:
                    st.caption(f"Текущая цена: ${cur_p:.4f}")
                    qcols = st.columns(3)
                    for off, lbl, col in [
                        (+1,"+1%",qcols[0]),(+2,"+2%",qcols[1]),(+5,"+5%",qcols[2]),
                        (-1,"-1%",qcols[0]),(-2,"-2%",qcols[1]),(-5,"-5%",qcols[2]),
                    ]:
                        with col:
                            if st.button(lbl, key=f"ql_{off}"):
                                lvl = cur_p*(1+off/100)
                                c = "#a6e3a1" if off > 0 else "#f38ba8"
                                st.session_state.h_lines.append(
                                    {"price": lvl, "color": c, "label": f"{lbl} {lvl:.4f}"})
            with dr3:
                st.markdown("**Фибоначчи**")
                fib_lo = st.number_input("Low", 0.0, format="%.6f", key="fib_lo",
                    help="Нижняя точка движения для Фибоначчи.")
                fib_hi = st.number_input("High", 0.0, format="%.6f", key="fib_hi",
                    help="Верхняя точка движения.")
                c1f, c2f = st.columns(2)
                if c1f.button("📐 Фиб (лонг)"):
                    if fib_lo > 0 and fib_hi > fib_lo:
                        st.session_state.fib_range = (fib_lo, fib_hi, "up")
                if c2f.button("📐 Фиб (шорт)"):
                    if fib_lo > 0 and fib_hi > fib_lo:
                        st.session_state.fib_range = (fib_lo, fib_hi, "down")
                if st.button("✖ Убрать фиб"):
                    st.session_state.fib_range = None

            if st.session_state.h_lines:
                st.markdown(f"**Линий на графике: {len(st.session_state.h_lines)}**")
                if st.button("🗑️ Удалить все линии"):
                    st.session_state.h_lines = []

        # Загрузка данных для графика
        @st.cache_data(ttl=20)
        def _chart_data(sym, tf, lim, oi_p, ex):
            return (
                get_klines(sym, tf, lim, exchange=ex),
                get_oi_history(sym, oi_p, 60, exchange=ex),
            )

        k, oi_h = _chart_data(sel_sym, sel_tf, chart_limit, oi_period, exchange)

        # Собираем оверлеи
        anal_tf, _ = TF_HIERARCHY.get(sel_tf, (sel_tf, sel_tf))
        k_anal = get_klines(sel_sym, anal_tf, 200, exchange=exchange) if anal_tf != sel_tf else k

        obs_ov  = find_order_blocks(k_anal) if not k_anal.empty else []
        fvgs_ov = find_fvg(k_anal)          if not k_anal.empty else []
        hunts_ov = find_stop_hunts(k)        if not k.empty else []
        tdp_ov  = get_tdp_levels(k_anal)    if not k_anal.empty else []
        wave_ov = analyze_waves(k_anal, pct=3.0) if not k_anal.empty else {}
        wave_pivots_ov = wave_ov.get("pivots", [])

        # Фибоначчи
        fibs_ov = {}
        if st.session_state.fib_range:
            fl, fh, fd = st.session_state.fib_range
            diff = fh - fl
            for ratio, r_val in FIB_RATIOS.items():
                fibs_ov[ratio] = fh - r_val * diff if fd == "up" else fl + r_val * diff

        # Уровни из анализа
        sl_ov = tp1_ov = tp2_ov = 0.0
        entry_ov = {}
        if st.session_state.analysis_sym == sel_sym and not k.empty:
            try:
                k1h_ov = get_klines(sel_sym, "1h", 200, exchange=exchange)
                r_ov   = df_sig[df_sig["Symbol"] == sel_sym].iloc[0]
                rep    = generate_analysis(sel_sym, k, k1h_ov, r_ov, mkt)
                entry_ov = {"low": rep.entry_zone[0], "high": rep.entry_zone[1]}
                sl_ov    = rep.stop_loss
                tp1_ov   = rep.take_profit_1
                tp2_ov   = rep.take_profit_2
            except Exception:
                pass

        overlays = {
            "obs":         obs_ov,
            "fvgs":        fvgs_ov,
            "hunts":       hunts_ov,
            "h_lines":     st.session_state.h_lines,
            "tdp":         tdp_ov,
            "fibs":        fibs_ov,
            "wave_pivots": wave_pivots_ov,
            "entry":       entry_ov,
            "sl":          sl_ov,
            "tp1":         tp1_ov,
            "tp2":         tp2_ov,
        }

        render_tv_chart(k, sel_sym, sel_tf, height=chart_h,
                        overlays=overlays, show_cvd=True, show_oi=oi_h)

        # Кнопка "Открыть анализ для этой монеты"
        if st.button(f"🎯 Открыть полный анализ {sel_sym}", key="chart_open_analysis"):
            st.session_state.analysis_sym = sel_sym
            st.rerun()

# ───────────────────────────────────────────────────────────
#  TAB 3 — ВОЛНЫ + ПАТТЕРН-СКАНЕР
# ───────────────────────────────────────────────────────────
with tab_wave:
    wave_sub, pattern_scan_sub = st.tabs(["🌊 Волновой анализ", "🔍 Сканер паттернов"])

    # ── ВОЛНОВОЙ АНАЛИЗ ──────────────────────────────────────
    with wave_sub:
        if df_sig.empty:
            st.info("Нет данных.")
        else:
            wc1,wc2,wc3,wc4,wc5 = st.columns(5)
            w_sym = wc1.selectbox("Монета", df_sig.head(40)["Symbol"].tolist(), key="wsym",
                help="Монета для волнового анализа.")
            w_tf  = wc2.selectbox("Таймфрейм", ["5m","15m","30m","1h","2h","4h","12h","1d","3d"], index=3,
                key="wtf", help="1h/4h — оптимально. Крупнее ТФ = значимее волны.")
            w_lim = wc3.select_slider("Свечей", [100,200,300,500], 300, key="wlim")
            w_pct = wc4.slider("ZigZag %", 1.0, 15.0, 3.0, 0.5, key="wpct",
                help="Мин % движение для пивота. 2-3% = средние волны. 5-8% = крупные.")
            w_filter = wc5.selectbox("Искать тип волн",
                ["Все","Импульс (5-волн ↑)","Импульс (5-волн ↓)","Коррекция ABC ↑","Коррекция ABC ↓",
                 "Треугольник","Только пивоты (без разметки)"],
                key="wfilt",
                help="Фильтр по типу волновой структуры. "
                     "Импульс = 5 волн в направлении тренда. "
                     "Коррекция = 3 волны против тренда. "
                     "Треугольник = сужающийся диапазон.")

            @st.cache_data(ttl=60)
            def _wdata(sym, tf, lim, ex):
                return get_klines(sym, tf, lim, exchange=ex)

            wk = _wdata(w_sym, w_tf, w_lim, exchange)
            if wk.empty:
                st.warning(f"Нет данных для {w_sym} {w_tf}")
            else:
                wr     = analyze_waves(wk, pct=w_pct)
                pivots = wr["pivots"]
                wt     = wr["wave_type"]

                # Применяем фильтр
                filter_map = {
                    "Импульс (5-волн ↑)":        ("impulse",   "up"),
                    "Импульс (5-волн ↓)":        ("impulse",   "down"),
                    "Коррекция ABC ↑":           ("corrective","up"),
                    "Коррекция ABC ↓":           ("corrective","down"),
                    "Треугольник":               ("triangle",  None),
                }
                filter_match = True
                if w_filter != "Все" and w_filter != "Только пивоты (без разметки)":
                    fw, fd = filter_map.get(w_filter, (None, None))
                    if fw and wt != fw:
                        filter_match = False

                col_i = {"impulse":"🟢","corrective":"🟡","triangle":"🔵","unknown":"⚪"}

                if not filter_match:
                    st.warning(f"Структура не соответствует фильтру. Найдено: **{wt}** · {wr['description']}")
                else:
                    st.info(f"{col_i.get(wt,'⚪')} **{wr['description']}**")

                ws1,ws2,ws3,ws4,ws5 = st.columns(5)
                ws1.metric("Пивотов",         len(pivots))
                ws2.metric("Тип",             wt.capitalize())
                ws3.metric("Хаев",            sum(1 for p in pivots if p.kind=="H"))
                ws4.metric("Лоев",            sum(1 for p in pivots if p.kind=="L"))
                ws5.metric("Размеченных",     sum(1 for p in pivots if p.label))

                # Детали пивотов
                with st.expander("📋 Все пивоты"):
                    prows = [{"#":i+1, "Тип":p.kind, "Метка":p.label or "–",
                              "Цена":f"${p.price:.4f}" if p.price<10 else f"${p.price:.2f}",
                              "Время":p.timestamp.strftime("%m-%d %H:%M")}
                             for i,p in enumerate(pivots)]
                    st.dataframe(pd.DataFrame(prows), use_container_width=True, hide_index=True, height=200)

                oi_wh = get_oi_history(w_sym, oi_period, 60, exchange=exchange)
                disp_pivots = pivots if w_filter != "Только пивоты (без разметки)" else []
                render_tv_chart(
                    wk, w_sym, w_tf, height=650,
                    overlays={"wave_pivots": disp_pivots,
                              "fibs": wr.get("fib_levels", {})},
                    show_cvd=True, show_oi=oi_wh,
                )

                fib_lvls = wr.get("fib_levels", {})
                if fib_lvls:
                    with st.expander("📊 Уровни Фибоначчи"):
                        cur = float(wk["close"].iloc[-1])
                        frows = [{"Уровень": r,
                                  "Цена": f"${p:.4f}" if p<10 else f"${p:.2f}",
                                  "До цены": f"{(p-cur)/cur*100:+.2f}%",
                                  "Зона": "🎯" if abs((p-cur)/cur*100) < 1 else ""}
                                 for r, p in fib_lvls.items()]
                        st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)

    # ── СКАНЕР ПАТТЕРНОВ ─────────────────────────────────────
    with pattern_scan_sub:
        st.markdown("#### 🔍 Сканер 18 графических паттернов по всем парам")

        if df_sig.empty:
            st.info("Нет данных.")
        else:
            psc1, psc2, psc3, psc4 = st.columns(4)
            with psc1:
                ps_tf = st.selectbox("Таймфрейм", ["5m","15m","1h","4h","1d"], index=2, key="ps_tf",
                    help="1h/4h = надёжнее. 5/15m = быстрее сигналы.")
            with psc2:
                ps_topn = st.slider("Топ N пар", 5, 100, 30, 5, key="ps_topn",
                    help="Больше пар = дольше (~1-2с/пара).")
            with psc3:
                ps_dir = st.selectbox("Направление", ["Все","Лонг (бычьи)","Шорт (медвежьи)"], key="ps_dir")
            with psc4:
                ps_minconf = st.slider("Мин. уверенность %", 0, 90, 55, 5, key="ps_minconf")

            # ── Авто-снятие чекбоксов при смене направления ──
            _BEAR_KEYS = ["ps_dt","ps_tt","ps_hs","ps_rw","ps_rt","ps_ich",
                          "ps_dmt","ps_bbt","ps_tcd","ps_beng","ps_star","ps_dsc"]
            _BULL_KEYS = ["ps_db","ps_tb","ps_ihs","ps_fw","ps_rb","ps_ch",
                          "ps_cnoh","ps_dmb","ps_bbu","ps_tcu","ps_eng","ps_ham","ps_asc"]
            if st.session_state.get("_ps_dir_prev") != ps_dir:
                st.session_state["_ps_dir_prev"] = ps_dir
                if ps_dir == "Лонг (бычьи)":
                    for k in _BEAR_KEYS: st.session_state[k] = False
                    for k in _BULL_KEYS: st.session_state[k] = True
                elif ps_dir == "Шорт (медвежьи)":
                    for k in _BULL_KEYS: st.session_state[k] = False
                    for k in _BEAR_KEYS: st.session_state[k] = True
                else:  # Все
                    for k in _BEAR_KEYS + _BULL_KEYS: st.session_state[k] = True
                st.rerun()

            # ── Все ключи чекбоксов (для снятия/выбора) ──
            _ALL_CB_KEYS = [
                "ps_dt","ps_tt","ps_hs","ps_rw","ps_rt","ps_ich","ps_dmt","ps_bbt","ps_tcd",
                "ps_db","ps_tb","ps_ihs","ps_fw","ps_rb","ps_ch","ps_cnoh","ps_dmb","ps_bbu","ps_tcu",
                "ps_asc","ps_dsc","ps_sym",
                "ps_eng","ps_beng","ps_ham","ps_star",
                "ps_gar","ps_bat","ps_but","ps_crab","ps_shk",
                "ps_elli","ps_ellc","ps_abc",
                "ps_smc","ps_smc_q","ps_smc_bb","ps_smc_td","ps_smc_tt","ps_smc_dv",
            ]
            _GRP_BEAR  = ["ps_dt","ps_tt","ps_hs","ps_rw","ps_rt","ps_ich","ps_dmt","ps_bbt","ps_tcd"]
            _GRP_BULL  = ["ps_db","ps_tb","ps_ihs","ps_fw","ps_rb","ps_ch","ps_cnoh","ps_dmb","ps_bbu","ps_tcu"]
            _GRP_TRI   = ["ps_asc","ps_dsc","ps_sym"]
            _GRP_CNDL  = ["ps_eng","ps_beng","ps_ham","ps_star"]
            _GRP_HARM  = ["ps_gar","ps_bat","ps_but","ps_crab","ps_shk"]
            _GRP_ELL   = ["ps_elli","ps_ellc","ps_abc"]
            _GRP_SMC   = ["ps_smc","ps_smc_q","ps_smc_bb","ps_smc_td","ps_smc_tt","ps_smc_dv"]

            # ── Кнопки управления всеми галочками ──
            st.markdown("""<style>
            div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
                font-size: .75rem; padding: 2px 10px;
            }
            </style>""", unsafe_allow_html=True)

            cb_col1, cb_col2, cb_col3, *_ = st.columns([1,1,1,4])
            if cb_col1.button("☑ Все", key="cb_sel_all", help="Выбрать все паттерны"):
                for k in _ALL_CB_KEYS: st.session_state[k] = True
                st.rerun()
            if cb_col2.button("☐ Снять все", key="cb_clr_all", help="Снять все галочки"):
                for k in _ALL_CB_KEYS: st.session_state[k] = False
                st.rerun()
            if cb_col3.button("↺ По умолчанию", key="cb_reset", help="Восстановить стандартный набор"):
                for k in _ALL_CB_KEYS: st.session_state[k] = True
                st.rerun()

            # ── Сгруппированные блоки паттернов ──
            grp_r1, grp_r2 = st.columns(2)

            # ГРУППА 1: Медвежьи разворот
            with grp_r1:
                with st.expander("📉 Медвежьи разворот (9)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_bear", help="Снять медвежьи"):
                            for k in _GRP_BEAR: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_dt  = st.checkbox("Double Top",        st.session_state.get("ps_dt",True),  key="ps_dt")
                        ps_tt  = st.checkbox("Triple Top",        st.session_state.get("ps_tt",True),  key="ps_tt")
                        ps_hs  = st.checkbox("Head & Shoulders",  st.session_state.get("ps_hs",True),  key="ps_hs")
                        ps_rw  = st.checkbox("Rising Wedge",      st.session_state.get("ps_rw",True),  key="ps_rw")
                        ps_rt  = st.checkbox("Rounded Top",       st.session_state.get("ps_rt",True),  key="ps_rt")
                        ps_ich = st.checkbox("Inv. Cup & Handle", st.session_state.get("ps_ich",True), key="ps_ich")
                        ps_dmt = st.checkbox("Diamond Top",       st.session_state.get("ps_dmt",True), key="ps_dmt")
                        ps_bbt = st.checkbox("Broadening Bear",   st.session_state.get("ps_bbt",True), key="ps_bbt")
                        ps_tcd = st.checkbox("Trend Change ↓",    st.session_state.get("ps_tcd",True), key="ps_tcd")

            # ГРУППА 2: Бычьи разворот
            with grp_r2:
                with st.expander("📈 Бычьи разворот (10)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_bull", help="Снять бычьи"):
                            for k in _GRP_BULL: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_db   = st.checkbox("Double Bottom",   st.session_state.get("ps_db",True),   key="ps_db")
                        ps_tb   = st.checkbox("Triple Bottom",   st.session_state.get("ps_tb",True),   key="ps_tb")
                        ps_ihs  = st.checkbox("Inverse H&S",     st.session_state.get("ps_ihs",True),  key="ps_ihs")
                        ps_fw   = st.checkbox("Falling Wedge",   st.session_state.get("ps_fw",True),   key="ps_fw")
                        ps_rb   = st.checkbox("Rounded Bottom",  st.session_state.get("ps_rb",True),   key="ps_rb")
                        ps_ch   = st.checkbox("Cup & Handle",    st.session_state.get("ps_ch",True),   key="ps_ch")
                        ps_cnoh = st.checkbox("Cup (no handle)", st.session_state.get("ps_cnoh",True), key="ps_cnoh")
                        ps_dmb  = st.checkbox("Diamond Bottom",  st.session_state.get("ps_dmb",True),  key="ps_dmb")
                        ps_bbu  = st.checkbox("Broadening Bull", st.session_state.get("ps_bbu",True),  key="ps_bbu")
                        ps_tcu  = st.checkbox("Trend Change ↑",  st.session_state.get("ps_tcu",True),  key="ps_tcu")

            grp_r3, grp_r4 = st.columns(2)

            # ГРУППА 3: Треугольники + Свечные
            with grp_r3:
                with st.expander("↔️ Треугольники (3)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_tri", help="Снять треугольники"):
                            for k in _GRP_TRI: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_asc = st.checkbox("Ascending Triangle",   st.session_state.get("ps_asc",True), key="ps_asc")
                        ps_dsc = st.checkbox("Descending Triangle",  st.session_state.get("ps_dsc",True), key="ps_dsc")
                        ps_sym = st.checkbox("Symmetrical Triangle", st.session_state.get("ps_sym",True), key="ps_sym")

                with st.expander("🕯️ Свечные паттерны (4)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_cndl", help="Снять свечные"):
                            for k in _GRP_CNDL: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_eng  = st.checkbox("Bullish Engulfing", st.session_state.get("ps_eng",True),  key="ps_eng")
                        ps_beng = st.checkbox("Bearish Engulfing", st.session_state.get("ps_beng",True), key="ps_beng")
                        ps_ham  = st.checkbox("Hammer / Pin Bar",  st.session_state.get("ps_ham",True),  key="ps_ham")
                        ps_star = st.checkbox("Shooting Star",     st.session_state.get("ps_star",True), key="ps_star")

            # ГРУППА 4: Гармоники + Эллиотт
            with grp_r4:
                with st.expander("🎵 Гармонические (5)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_harm", help="Снять гармоники"):
                            for k in _GRP_HARM: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_gar  = st.checkbox("Gartley",   st.session_state.get("ps_gar",True),  key="ps_gar")
                        ps_bat  = st.checkbox("Bat",       st.session_state.get("ps_bat",True),  key="ps_bat")
                        ps_but  = st.checkbox("Butterfly", st.session_state.get("ps_but",True),  key="ps_but")
                        ps_crab = st.checkbox("Crab",      st.session_state.get("ps_crab",True), key="ps_crab")
                        ps_shk  = st.checkbox("Shark",     st.session_state.get("ps_shk",True),  key="ps_shk")

                with st.expander("🌊 Волновой анализ (3)", expanded=False):
                    gc1, gc2 = st.columns([3,1])
                    with gc2:
                        if st.button("☐", key="clr_ell", help="Снять волновые"):
                            for k in _GRP_ELL: st.session_state[k] = False
                            st.rerun()
                    with gc1:
                        ps_ell_i = st.checkbox("Impulse 5-wave", st.session_state.get("ps_elli",True), key="ps_elli")
                        ps_ell_c = st.checkbox("Corrective ABC", st.session_state.get("ps_ellc",True), key="ps_ellc")
                        ps_abc   = st.checkbox("ABC Zigzag",     st.session_state.get("ps_abc",True),  key="ps_abc")

            # ГРУППА 5: SMC — отдельный полноширокий блок
            with st.expander("🔲 Smart Money Concepts — SMC (6)", expanded=True):
                gc1, gc2 = st.columns([5,1])
                with gc2:
                    if st.button("☐ Снять", key="clr_smc", help="Снять SMC"):
                        for k in _GRP_SMC: st.session_state[k] = False
                        st.rerun()
                with gc1:
                    smc_c1, smc_c2, smc_c3 = st.columns(3)
                    with smc_c1:
                        ps_smc   = st.checkbox("Order Block + FVG", st.session_state.get("ps_smc",True),    key="ps_smc",
                            help="OB рядом с текущей ценой + FVG подтверждение.")
                        ps_smc_q = st.checkbox("SMC Quality Setup", st.session_state.get("ps_smc_q",True),  key="ps_smc_q",
                            help="PDF 7: OB+FVG+BOS+структура+ликвидность — полный сетап.")
                    with smc_c2:
                        ps_smc_bb= st.checkbox("Breaker Block",     st.session_state.get("ps_smc_bb",True), key="ps_smc_bb",
                            help="PDF 3/6: Бывший OB пробит, ретест зоны.")
                        ps_smc_td= st.checkbox("Three Drives",      st.session_state.get("ps_smc_td",True), key="ps_smc_td",
                            help="PDF 7: 3 экстремума с похожими ногами — разворот.")
                    with smc_c3:
                        ps_smc_tt= st.checkbox("Three Tap Setup",   st.session_state.get("ps_smc_tt",True), key="ps_smc_tt",
                            help="PDF 7: 3 теста одного уровня. Локальный + глобальный поиск.")
                        ps_smc_dv= st.checkbox("Range + Deviation", st.session_state.get("ps_smc_dv",True), key="ps_smc_dv",
                            help="PDF 6: Консолидация → ложный пробой → возврат.")

            # ── Показываем сколько паттернов выбрано ──
            _active_count = sum(1 for k in _ALL_CB_KEYS if st.session_state.get(k, True))
            st.caption(f"✅ Выбрано типов для поиска: **{_active_count}** из {len(_ALL_CB_KEYS)}")

            if "ps_scan_results" not in st.session_state:
                st.session_state.ps_scan_results = None

            _COIN_SPINNER_HTML = """
<style>
@keyframes coinSpin { 0%{transform:rotateY(0deg)}100%{transform:rotateY(360deg)} }
@keyframes coinFall {
  0%   { transform: translateY(-10px) rotate(0deg);   opacity: 1; }
  80%  { opacity: 1; }
  100% { transform: translateY(110px) rotate(720deg); opacity: 0; }
}
@keyframes glowPulse { 0%,100%{box-shadow:0 0 18px #FFD70099}50%{box-shadow:0 0 36px #FFD700cc,0 0 60px #FFD70055} }
.cs-wrap {
  display:flex; flex-direction:column; align-items:center; gap:10px;
  padding:24px 0; user-select:none;
}
.cs-coins-row { position:relative; height:90px; width:260px; }
.cs-coin-fall {
  position:absolute; font-size:22px;
  animation: coinFall 1.8s ease-in infinite;
}
.cs-coin-fall:nth-child(1){left:5%;  animation-delay:0.0s;}
.cs-coin-fall:nth-child(2){left:20%; animation-delay:0.3s;}
.cs-coin-fall:nth-child(3){left:38%; animation-delay:0.6s;}
.cs-coin-fall:nth-child(4){left:55%; animation-delay:0.9s;}
.cs-coin-fall:nth-child(5){left:72%; animation-delay:1.2s;}
.cs-coin-fall:nth-child(6){left:88%; animation-delay:1.5s;}
.cs-logo {
  width:68px; height:68px; border-radius:50%;
  background:radial-gradient(circle at 38% 38%, #FFE066, #B8860B 70%, #7a5700);
  display:flex; align-items:center; justify-content:center;
  font-size:32px;
  animation: glowPulse 1.4s ease-in-out infinite, coinSpin 2.5s linear infinite;
}
.cs-label { color:#FFD700; font-size:1.05rem; font-weight:700; letter-spacing:.04em; }
.cs-sub   { color:#a89060; font-size:.82rem; }
</style>
<div class="cs-wrap">
  <div class="cs-coins-row">
    <span class="cs-coin-fall">🪙</span>
    <span class="cs-coin-fall">💰</span>
    <span class="cs-coin-fall">🪙</span>
    <span class="cs-coin-fall">💰</span>
    <span class="cs-coin-fall">🪙</span>
    <span class="cs-coin-fall">💰</span>
  </div>
  <div class="cs-logo">📊</div>
  <div class="cs-label">Сканирование рынка...</div>
  <div class="cs-sub">Анализируем паттерны по всем парам</div>
</div>
"""

            if st.button("🚀 Запустить сканирование", type="primary", use_container_width=True):
                # Собираем ключевые слова для фильтра
                sel_keys = []
                if ps_dt:  sel_keys.append("double_top")
                if ps_tt:  sel_keys.append("triple_top")
                if ps_hs:  sel_keys.append("head_&_shoulders")
                if ps_rw:  sel_keys.append("rising_wedge")
                if ps_rt:  sel_keys.append("rounded_top")
                if ps_ich: sel_keys.append("inverted_cup")
                if ps_dmt: sel_keys.append("diamond_top")
                if ps_bbt: sel_keys.append("bearish_broadening")
                if ps_tcd: sel_keys.append("trend_change_↓")
                if ps_db:  sel_keys.append("double_bottom")
                if ps_tb:  sel_keys.append("triple_bottom")
                if ps_ihs: sel_keys.append("inverse_h")
                if ps_fw:  sel_keys.append("falling_wedge")
                if ps_rb:  sel_keys.append("rounded_bottom")
                if ps_ch:  sel_keys.append("cup_&_handle")
                if ps_cnoh:sel_keys.append("cup_(no_handle)")
                if ps_dmb: sel_keys.append("diamond_bottom")
                if ps_bbu: sel_keys.append("bullish_broadening")
                if ps_tcu: sel_keys.append("trend_change_↑")
                if ps_asc: sel_keys.append("ascending_triangle")
                if ps_dsc: sel_keys.append("descending_triangle")
                if ps_sym: sel_keys.append("symmetrical_triangle")
                if ps_eng: sel_keys.append("bullish_engulfing")
                if ps_beng:sel_keys.append("bearish_engulfing")
                if ps_ham: sel_keys.append("hammer")
                if ps_star:sel_keys.append("shooting_star")
                if ps_gar: sel_keys.append("harmonic_gartley")
                if ps_bat: sel_keys.append("harmonic_bat")
                if ps_but: sel_keys.append("harmonic_butterfly")
                if ps_crab:sel_keys.append("harmonic_crab")
                if ps_shk: sel_keys.append("harmonic_shark")
                if ps_abc: sel_keys.append("abc_correction")

                syms    = df_sig.head(ps_topn)["Symbol"].tolist()
                results = []
                _spin   = st.empty()
                _spin.html(_COIN_SPINNER_HTML)
                prog    = st.progress(0, text="")

                for i, sym in enumerate(syms):
                    prog.progress((i + 1) / len(syms), text=f"[{i+1}/{len(syms)}] {sym}...")
                    try:
                        kl = get_klines(sym, ps_tf, 250, exchange=exchange)
                        if kl.empty: continue
                        cur_price = float(kl["close"].iloc[-1])

                        # ── Все паттерны из patterns.py ──
                        pats = scan_patterns(kl)
                        for p in pats:
                            if not p.found: continue
                            name_lc = p.name.lower().replace(" ","_").replace("&","&")
                            matched = not sel_keys or any(k in name_lc for k in sel_keys)
                            if not matched: continue
                            if p.confidence * 100 < ps_minconf: continue
                            if ps_dir == "Лонг (бычьи)"    and p.direction != "bullish": continue
                            if ps_dir == "Шорт (медвежьи)" and p.direction != "bearish": continue
                            dir_ico = "↑ Лонг" if p.direction=="bullish" else ("↓ Шорт" if p.direction=="bearish" else "↔")
                            spark = _sparkline_svg(
                                kl["close"].tolist()[-80:],
                                key_levels=p.key_levels,
                                direction=p.direction,
                            )
                            results.append({
                                "График":     spark,
                                "Монета":     sym,
                                "Паттерн":    p.name,
                                "Направление":dir_ico,
                                "Уверенность":f"{p.confidence*100:.0f}%",
                                "Подтверждён":"✅" if p.confirmed else "⏳",
                                "Цена":       f"${cur_price:,.4f}" if cur_price<10 else f"${cur_price:,.2f}",
                                "ТФ":         ps_tf,
                                "_desc":      p.description[:120],
                            })

                        # ── Elliott Wave ──
                        if ps_ell_i or ps_ell_c:
                            wr2 = analyze_waves(kl, pct=3.0)
                            wt2 = wr2["wave_type"]
                            if (wt2=="impulse" and ps_ell_i) or (wt2=="corrective" and ps_ell_c):
                                if ps_dir not in ["Лонг (бычьи)","Шорт (медвежьи)"]:
                                    results.append({
                                        "График":     _sparkline_svg(kl["close"].tolist()[-80:]),
                                        "Монета":     sym,
                                        "Паттерн":    f"Elliott {wt2.capitalize()}",
                                        "Направление":"↑/↓",
                                        "Уверенность":"70%",
                                        "Подтверждён":"✅",
                                        "Цена":       f"${cur_price:,.4f}" if cur_price<10 else f"${cur_price:,.2f}",
                                        "ТФ":         ps_tf,
                                        "_desc":      wr2["description"][:120],
                                    })

                        # ── SMC OB+FVG ──
                        if ps_smc:
                            for ob in find_order_blocks(kl)[:4]:
                                dist = abs(cur_price - (ob.high+ob.low)/2) / cur_price * 100
                                if dist > 2.5: continue
                                dir_s = "bullish" if ob.kind=="bullish" else "bearish"
                                if ps_dir=="Лонг (бычьи)"    and dir_s!="bullish": continue
                                if ps_dir=="Шорт (медвежьи)" and dir_s!="bearish": continue
                                results.append({
                                    "График":     _sparkline_svg(kl["close"].tolist()[-80:], [ob.high, ob.low], dir_s),
                                    "Монета":     sym,
                                    "Паттерн":    "SMC Order Block",
                                    "Направление":"↑ Лонг" if dir_s=="bullish" else "↓ Шорт",
                                    "Уверенность":f"{max(55, int(100-dist*20))}%",
                                    "Подтверждён":"✅",
                                    "Цена":       f"${cur_price:,.4f}" if cur_price<10 else f"${cur_price:,.2f}",
                                    "ТФ":         ps_tf,
                                    "_desc":      f"{ob.kind} OB @ {(ob.high+ob.low)/2:.4f} (dist {dist:.1f}%)",
                                })

                        # ── SMC Расширенные сетапы ──
                        _smc_types = []
                        if ps_smc_q:  _smc_types.append("quality_ob")
                        if ps_smc_bb: _smc_types.append("breaker")
                        if ps_smc_td: _smc_types.append("three_drives")
                        if ps_smc_tt: _smc_types.append("three_tap")
                        if ps_smc_dv: _smc_types.append("deviation")
                        if _smc_types:
                            smc_setups = scan_smc_setups(kl, _smc_types, tf=ps_tf)
                            for ss in smc_setups:
                                if ss.confidence * 100 < ps_minconf: continue
                                dir_s = ss.kind
                                if ps_dir=="Лонг (бычьи)"    and dir_s!="bullish": continue
                                if ps_dir=="Шорт (медвежьи)" and dir_s!="bearish": continue
                                kl_c = kl["close"].tolist()
                                spark = _sparkline_svg(
                                    kl_c[-80:],
                                    key_levels=ss.key_levels[:2] if ss.key_levels else None,
                                    direction=dir_s,
                                )
                                dir_ico = "↑ Лонг" if dir_s=="bullish" else "↓ Шорт"
                                results.append({
                                    "График":     spark,
                                    "Монета":     sym,
                                    "Паттерн":    ss.name,
                                    "Направление":dir_ico,
                                    "Уверенность":f"{ss.confidence*100:.0f}%",
                                    "Подтверждён":"✅" if ss.confirmed else "⏳",
                                    "Цена":       f"${cur_price:,.4f}" if cur_price<10 else f"${cur_price:,.2f}",
                                    "ТФ":         ps_tf,
                                    "_desc":      ss.description[:120],
                                    "_kp":        ss.key_points,
                                    "_kl":        ss.key_levels,
                                })

                    except Exception:
                        pass

                prog.empty()
                _spin.empty()
                st.session_state.ps_scan_results = results

            # ── Результаты сканирования ──
            scan_res = st.session_state.get("ps_scan_results")
            if scan_res is not None:
                if not scan_res:
                    st.warning("Паттернов не найдено. Снизьте мин. уверенность или добавьте больше типов.")
                else:
                    r_df = pd.DataFrame(scan_res)
                    r_df["_c"] = r_df["Уверенность"].str.replace("%","").astype(int)
                    r_df = r_df.sort_values("_c", ascending=False).drop("_c", axis=1)

                    sm1,sm2,sm3,sm4,sm5 = st.columns(5)
                    sm1.metric("Паттернов",    len(scan_res))
                    sm2.metric("Монет",        r_df["Монета"].nunique())
                    sm3.metric("↑ Бычьих",     len(r_df[r_df["Направление"]=="↑ Лонг"]))
                    sm4.metric("↓ Медвежьих",  len(r_df[r_df["Направление"]=="↓ Шорт"]))
                    sm5.metric("✅ Подтверждено",len(r_df[r_df["Подтверждён"]=="✅"]))

                    # Колонки для отображения (без служебных _*)
                    disp_cols = [c for c in r_df.columns if not c.startswith("_")]
                    disp_df   = r_df[disp_cols]

                    # Диалог с полным графиком паттерна
                    @st.dialog("📈 График паттерна", width="large")
                    def _show_pattern_dialog(ph_dict: dict):
                        _render_pattern_detail(ph_dict, exchange, oi_period, mkt, df_sig)

                    st.caption("👆 Кликни строку → откроется график с паттерном")
                    ev2 = st.dataframe(
                        disp_df,
                        use_container_width=True,
                        height=min(40 + len(disp_df) * 58, 520),
                        hide_index=True,
                        selection_mode="single-row",
                        on_select="rerun",
                        key="scan_df",
                        column_config={
                            "График": st.column_config.ImageColumn(
                                "График", width="medium", help="Превью паттерна (последние 80 свечей)"),
                            "Монета":      st.column_config.TextColumn("Монета",      width="small"),
                            "Паттерн":     st.column_config.TextColumn("Паттерн",     width="medium"),
                            "Направление": st.column_config.TextColumn("↕",           width="small"),
                            "Уверенность": st.column_config.TextColumn("Увер.",       width="small"),
                            "Подтверждён": st.column_config.TextColumn("✓",           width="small"),
                            "Цена":        st.column_config.TextColumn("Цена",        width="small"),
                            "ТФ":          st.column_config.TextColumn("ТФ",          width="small"),
                        },
                    )
                    sel2 = ev2.selection.rows if hasattr(ev2, "selection") else []

                    if sel2:
                        row_s  = r_df.iloc[sel2[0]]
                        ph_now = row_s.to_dict()
                        st.session_state.pat_highlight    = ph_now
                        st.session_state.inv_analysis_sym = row_s["Монета"]
                        _show_pattern_dialog(ph_now)

                    col_dl, col_clr = st.columns([3, 1])
                    col_dl.download_button(
                        "⬇️ CSV",
                        disp_df.to_csv(index=False).encode(),
                        "pattern_scan.csv", "text/csv",
                    )
                    if col_clr.button("✖ Очистить", key="clear_scan"):
                        st.session_state.ps_scan_results  = None
                        st.session_state.pat_highlight    = None
                        st.session_state.inv_analysis_sym = None
                        st.rerun()

            # Показываем детейл если был выбран паттерн ранее (до нового сканирования)
            ph = st.session_state.get("pat_highlight")
            if ph and not st.session_state.get("ps_scan_results"):
                _render_pattern_detail(ph, exchange, oi_period, mkt, df_sig)

# ───────────────────────────────────────────────────────────
#  TAB 4 — ПРЕДСКАЗАНИЕ
# ───────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("#### 🔮 Предсказание на основе истории")
    st.caption("Ищем похожие паттерны в истории и показываем что было дальше.")

    if df_sig.empty:
        st.info("Нет данных.")
    else:
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        p_sym    = pc1.selectbox("Монета", df_sig.head(30)["Symbol"].tolist(), key="psym")
        p_tf     = pc2.selectbox("ТФ",     ALL_TIMEFRAMES, index=ALL_TIMEFRAMES.index("1h"), key="ptf",
            help="Чем крупнее ТФ — тем надёжнее паттерны.")
        p_tlen   = pc3.select_slider("Длина шаблона (свечи)",  [10,15,20,30,40,50], 20, key="ptlen",
            help="Сколько последних свечей взять как 'образец'. "
                 "20-30 = оптимально для большинства ТФ.")
        p_prlen  = pc4.select_slider("Прогноз (свечи)",        [5,10,15,20,30],     15, key="pprlen",
            help="На сколько свечей вперёд строить прогноз.")
        p_mincor = pc5.slider("Мин. корреляция", 0.60, 0.95, 0.75, 0.05, key="pmincor",
            help="Насколько похожим должен быть исторический паттерн (0 = любой, 1 = идентичный).")

        @st.cache_data(ttl=60)
        def _pdata(sym, tf, ex):
            return get_klines(sym, tf, 1000, exchange=ex)

        pk = _pdata(p_sym, p_tf, exchange)
        if pk.empty:
            st.warning("Нет данных.")
        else:
            with st.spinner("Анализирую историю..."):
                sim = find_similar_patterns(pk, p_tlen, p_prlen, top_n=10, min_corr=p_mincor)

            # Вердикт
            if sim.up_probability >= 0.65:
                v_col, v_txt = "#a6e3a1", "📈 ВЕРОЯТЕН РОСТ"
            elif sim.up_probability <= 0.35:
                v_col, v_txt = "#f38ba8", "📉 ВЕРОЯТНО ПАДЕНИЕ"
            else:
                v_col, v_txt = "#f9e2af", "↔️ НЕЙТРАЛЬНО"

            st.markdown(
                f"""<div style="background:#1e1e2e;border:2px solid {v_col};
                border-radius:10px;padding:14px;margin-bottom:10px">
                <b style="color:{v_col};font-size:1.2rem">{v_txt}</b>
                &nbsp;&nbsp;<span style="color:#cdd6f4">
                Вероятность роста: <b>{sim.up_probability*100:.0f}%</b> ·
                Найдено паттернов: <b>{len(sim.matches)}</b> ·
                Уверенность: <b>{sim.confidence*100:.0f}%</b></span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(sim.description)

            if sim.matches:
                pm1, pm2 = st.columns(2)

                with pm1:
                    # Распределение исходов
                    changes = [m.next_change_pct for m in sim.matches]
                    fig_hist = px.histogram(
                        x=changes, nbins=10,
                        title="Распределение исходов (%)",
                        template="plotly_dark",
                        color_discrete_sequence=["#89b4fa"],
                        labels={"x": "Движение %"},
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="#f38ba8")
                    fig_hist.add_vline(x=sim.avg_move_pct, line_dash="dot", line_color="#f9e2af",
                                       annotation_text=f"Среднее {sim.avg_move_pct:+.1f}%",
                                       annotation=dict(font_color="#f9e2af"))
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,17,27,0.9)",
                        height=280, margin=dict(t=35,b=20,l=10,r=10))
                    st.plotly_chart(fig_hist, use_container_width=True)

                with pm2:
                    # Средняя траектория
                    if sim.predicted_path:
                        fig_path = go.Figure()
                        x_path = list(range(len(sim.predicted_path)))
                        fig_path.add_trace(go.Scatter(
                            x=x_path, y=sim.predicted_path,
                            mode="lines", name="Средний путь",
                            line=dict(color="#f9e2af", width=2),
                        ))
                        # Зона уверенности
                        all_paths = [m.candles_after for m in sim.matches if len(m.candles_after) == p_prlen]
                        if len(all_paths) > 1:
                            import numpy as np
                            arr = np.array(all_paths)
                            upper = (np.percentile(arr, 75, axis=0)).tolist()
                            lower = (np.percentile(arr, 25, axis=0)).tolist()
                            fig_path.add_trace(go.Scatter(
                                x=x_path + x_path[::-1],
                                y=upper + lower[::-1],
                                fill="toself",
                                fillcolor="rgba(249,226,175,0.12)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="IQR зона",
                            ))
                        fig_path.add_hline(y=0, line_dash="dash", line_color="#585b70")
                        fig_path.update_layout(
                            title=f"Прогноз следующих {p_prlen} свечей",
                            template="plotly_dark", height=280,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,17,27,0.9)",
                            margin=dict(t=35,b=20,l=10,r=10), showlegend=False,
                            yaxis_title="% от входа",
                            xaxis_title=f"Свечи вперёд ({p_tf})",
                        )
                        st.plotly_chart(fig_path, use_container_width=True)

                # Таблица совпадений
                with st.expander(f"🔍 Детали похожих паттернов ({len(sim.matches)})"):
                    match_rows = [{
                        "Дата":         m.timestamp.strftime("%Y-%m-%d %H:%M"),
                        "Корреляция":   f"{m.correlation:.3f}",
                        "Движение после": f"{m.next_change_pct:+.2f}%",
                        "Направление":  "↑" if m.direction=="up" else "↓",
                    } for m in sim.matches]
                    st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────
#  TAB 5 — СЕНТИМЕНТ
# ───────────────────────────────────────────────────────────
with tab_sent:
    st.markdown("#### 📡 Рыночный сентимент и социальные сигналы")

    @st.cache_data(ttl=300)
    def _movers(): return get_top_movers()

    movers = _movers()

    # Fear & Greed
    fg_col = ("#f38ba8" if fg_val < 30 else
              "#f38ba8" if fg_val < 45 else
              "#f9e2af" if fg_val < 56 else
              "#a6e3a1" if fg_val < 75 else "#cba6f7")

    fg_bar_filled = int(fg_val / 10)
    fg_bar = "█" * fg_bar_filled + "░" * (10 - fg_bar_filled)

    st.markdown(
        f"""<div style="background:#1e1e2e;border:1px solid #313244;
        border-radius:10px;padding:16px 20px;margin-bottom:16px">
        <b style="font-size:1.1rem">Fear & Greed Index</b><br>
        <span style="font-size:2rem;font-weight:700;color:{fg_col}">{fg_val}</span>
        &nbsp;&nbsp;
        <span style="color:{fg_col};font-size:1rem">{fg.get('label','–')}</span><br>
        <code style="color:{fg_col};font-size:0.9rem">{fg_bar}</code>
        &nbsp;&nbsp;
        <span style="color:#585b70;font-size:0.85rem">
        Вчера: {fg.get('yesterday','–')} · Ср. за 7д: {fg.get('week_avg','–')}
        </span><br>
        <span style="color:#585b70;font-size:0.82rem">
        {'💡 Экстремальный страх = исторически хорошее время для покупки' if fg_val < 30 else
         '⚠️ Экстремальная жадность = будьте осторожны с лонгами' if fg_val > 75 else
         'Рынок в нейтральной зоне'}
        </span>
        </div>""",
        unsafe_allow_html=True,
    )

    st_c1, st_c2 = st.columns(2)

    with st_c1:
        st.markdown("#### 🔥 Trending coins (CoinGecko)")
        if trend:
            for t in trend:
                st.markdown(f"**{t['symbol']}** · {t['name']} · MarCap #{t['rank']}")
        else:
            st.caption("Нет данных (CoinGecko rate limit)")

    with st_c2:
        st.markdown("#### 🏆 Топ гейнеры 24ч")
        if movers.get("gainers"):
            for g in movers["gainers"][:5]:
                col = "#a6e3a1" if g["change"] > 0 else "#f38ba8"
                st.markdown(
                    f"**{g['symbol']}** — "
                    f"<span style='color:{col}'>{g['change']:+.1f}%</span>  "
                    f"<span style='color:#585b70'>vol ${g['volume']/1e6:.0f}M</span>",
                    unsafe_allow_html=True,
                )

    st.divider()

    # Сентимент по конкретной монете
    st.markdown("#### 🔍 Сентимент по монете")
    sent_sym = st.selectbox("Монета для анализа сентимента",
        df_sig.head(20)["Symbol"].tolist() if not df_sig.empty else [],
        key="sent_sym_sel",
        help="Данные от CoinGecko: голоса сообщества, followers, trending.")

    if sent_sym:
        with st.spinner("Загружаю данные сентимента..."):
            @st.cache_data(ttl=600)
            def _sent(sym): return get_social_score(sym)
            soc = _sent(sent_sym)

        coin_d = soc.get("coin", {})
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Социальный Score", soc.get("score", 0),
            help="Агрегированный балл: Fear&Greed + trending + сентимент сообщества")
        sc2.metric("👍 Голоса за",     f"{coin_d.get('sentiment_up') or 0:.0f}%")
        sc3.metric("Twitter",          f"{coin_d.get('twitter_followers') or 0:,}")
        sc4.metric("Telegram",         f"{coin_d.get('telegram_users') or 0:,}")

        if soc.get("in_trend"):
            st.success(f"🔥 **{sent_sym}** сейчас в топ-7 трендов CoinGecko!")

        for note in soc.get("notes", []):
            st.markdown(f"- {note}")

# ───────────────────────────────────────────────────────────
#  TAB 6 — СТАТИСТИКА
# ───────────────────────────────────────────────────────────
with tab_stat:
    if df_sig.empty:
        st.info("Нет данных.")
    else:
        sc1, sc2 = st.columns(2)
        with sc1:
            sco = df_sig["Score"].value_counts().sort_index()
            fig_s = px.bar(x=sco.index, y=sco.values,
                           title="Score распределение", template="plotly_dark",
                           color=sco.values, color_continuous_scale="RdYlGn",
                           labels={"x":"Score","y":"Монет"})
            fig_s.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(17,17,27,0.9)", showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)
        with sc2:
            fig_oi = px.histogram(df_sig, x="OI_Change", nbins=30,
                                  title="Изменение ОИ %",
                                  template="plotly_dark",
                                  color_discrete_sequence=["#89b4fa"])
            fig_oi.add_vline(x=0, line_dash="dash", line_color="#f38ba8")
            fig_oi.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(17,17,27,0.9)")
            st.plotly_chart(fig_oi, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🚀 Топ-10 по росту цены")
            tp = df_sig.nlargest(10,"Change24h")[["Symbol","Change24h","Volume24h","Score"]].copy()
            tp["Change24h"] = tp["Change24h"].apply(lambda x:f"{x:+.2f}%")
            tp["Volume24h"] = tp["Volume24h"].apply(lambda x:f"${x/1e6:.1f}M")
            st.dataframe(tp, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("#### 📈 Топ-10 по росту ОИ")
            to = df_sig.nlargest(10,"OI_Change")[["Symbol","OI_Change","OI_USD","CVD","Score"]].copy()
            to["OI_Change"] = to["OI_Change"].apply(lambda x:f"{x:+.1f}%")
            to["OI_USD"]    = to["OI_USD"].apply(lambda x:f"${x/1e6:.1f}M" if x else "–")
            st.dataframe(to, use_container_width=True, hide_index=True)

        if st.session_state.signal_counter:
            st.markdown("#### 🔁 Сигналы за сессию")
            cnt = pd.DataFrame(list(st.session_state.signal_counter.items()),
                               columns=["Symbol","Сигналов"]).sort_values("Сигналов",ascending=False)
            st.dataframe(cnt, use_container_width=True, hide_index=True)
