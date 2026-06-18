"""
TradingView-style chart component with full drawing tools + indicator toggles.
Drawing tools: Line, Ray, HLine, VLine, Channel, Rectangle, Triangle,
               Fibonacci Retracement/Extension, Text, Arrow, Pitchfork.
Indicators: MA20/50/200, EMA, BB, VWAP, RSI, MACD, Stoch.
"""
import json
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import streamlit as st

ALL_TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","3d","1w"]

TF_HIERARCHY = {
    "1m": ("1m","15m"),  "3m": ("3m","15m"),  "5m": ("5m","1h"),
    "15m":("15m","1h"),  "30m":("30m","4h"),  "1h": ("1h","4h"),
    "2h": ("2h","1d"),   "4h": ("4h","1d"),   "6h": ("6h","1d"),
    "12h":("12h","1d"),  "1d": ("1d","1d"),   "3d": ("3d","1d"),
    "1w": ("1w","1d"),
}


def _to_tv(klines: pd.DataFrame) -> tuple:
    candles, volumes, cvds = [], [], []
    cum = 0.0
    for _, r in klines.iterrows():
        t = int(r["timestamp"].timestamp())
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        candles.append({"time":t,"open":o,"high":h,"low":l,"close":c})
        is_bull = c >= o
        vol = float(r["quote_volume"])
        volumes.append({"time":t,"value":vol,"color":"#26a69a" if is_bull else "#ef5350"})
        buy  = float(r.get("taker_buy_quote", vol*0.5))
        sell = vol - buy
        cum += buy - sell
        cvds.append({"time":t,"value":cum,"color":"#89b4fa" if cum>=0 else "#f38ba8"})
    return candles, volumes, cvds


def _calc_indicators(klines: pd.DataFrame) -> dict:
    c = klines["close"].astype(float)
    h = klines["high"].astype(float)
    l = klines["low"].astype(float)
    v = klines["quote_volume"].astype(float)
    ts = [int(t.timestamp()) for t in klines["timestamp"]]

    def series(vals):
        return [{"time": t, "value": round(float(v),8)} for t,v in zip(ts,vals) if not np.isnan(v)]

    result = {}

    # MAs
    for p in [20, 50, 200]:
        ma = c.rolling(p).mean()
        result[f"ma{p}"] = series(ma)

    # EMA 21
    ema21 = c.ewm(span=21, adjust=False).mean()
    result["ema21"] = series(ema21)

    # Bollinger Bands (20,2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    result["bb_mid"]  = series(sma20)
    result["bb_up"]   = series(sma20 + 2*std20)
    result["bb_dn"]   = series(sma20 - 2*std20)

    # VWAP (rolling session-like: from start of data)
    cum_pv = (((h+l+c)/3) * v).cumsum()
    cum_v  = v.cumsum()
    vwap   = cum_pv / cum_v
    result["vwap"] = series(vwap)

    # RSI 14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    result["rsi"] = series(rsi)

    # MACD (12,26,9)
    ema12 = c.ewm(span=12,adjust=False).mean()
    ema26 = c.ewm(span=26,adjust=False).mean()
    macd  = ema12 - ema26
    signal= macd.ewm(span=9,adjust=False).mean()
    hist  = macd - signal
    result["macd_line"]   = series(macd)
    result["macd_signal"] = series(signal)
    result["macd_hist"]   = [{"time":t,"value":round(float(v),8),
                               "color":"#a6e3a1" if v>=0 else "#f38ba8"}
                              for t,v in zip(ts,hist) if not np.isnan(v)]

    # Stochastic (14,3,3)
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    k_raw  = 100*(c - low14)/(high14-low14+1e-10)
    k_smo  = k_raw.rolling(3).mean()
    d_smo  = k_smo.rolling(3).mean()
    result["stoch_k"] = series(k_smo)
    result["stoch_d"] = series(d_smo)

    return result


def _serialize_overlays(klines: pd.DataFrame, overlays: dict) -> dict:
    if klines.empty:
        return {}
    ts0 = int(klines["timestamp"].iloc[0].timestamp())
    ts1 = int(klines["timestamp"].iloc[-1].timestamp())

    obs  = [{"kind":o.kind,"high":o.high,"low":o.low}   for o in overlays.get("obs",[])]
    fvgs = [{"kind":f.kind,"high":f.high,"low":f.low}    for f in overlays.get("fvgs",[]) if not f.filled]
    h_lines = overlays.get("h_lines", [])

    entry = overlays.get("entry", {})
    ent = {"low": entry.get("low",0),"high":entry.get("high",0)} if entry else {"low":0,"high":0}
    sl   = overlays.get("sl",  0)
    tp1  = overlays.get("tp1", 0)
    tp2  = overlays.get("tp2", 0)

    hunts = []
    for h in overlays.get("hunts", []):
        idx = h.candle_idx
        if 0 <= idx < len(klines):
            hunts.append({"time":int(klines["timestamp"].iloc[idx].timestamp()),
                          "price":float(klines["low"].iloc[idx]) if h.kind=="bullish"
                                  else float(klines["high"].iloc[idx]),
                          "kind":h.kind})

    wave_marks = []
    for p in overlays.get("wave_pivots", []):
        if p.label:
            wave_marks.append({"time":int(p.timestamp.timestamp()),
                               "price":float(p.price),"label":p.label,"kind":p.kind})

    tdp  = [{"price":lvl.price,"desc":lvl.kind} for lvl in overlays.get("tdp",[])]
    fibs = overlays.get("fibs", {})

    pattern_shapes = overlays.get("pattern_shapes", [])

    return {"ts0":ts0,"ts1":ts1,"obs":obs,"fvgs":fvgs,"h_lines":h_lines,
            "entry":ent,"sl":sl,"tp1":tp1,"tp2":tp2,
            "hunts":hunts,"wave_marks":wave_marks,"tdp":tdp,"fibs":fibs,
            "pattern_shapes": pattern_shapes}


def render_tv_chart(
    klines:    pd.DataFrame,
    symbol:    str,
    timeframe: str,
    height:    int  = 680,
    overlays:  dict = None,
    show_cvd:  bool = True,
    show_oi:   pd.DataFrame = None,
) -> None:
    if klines is None or klines.empty:
        st.warning("Нет данных для графика")
        return

    candles, volumes, cvds = _to_tv(klines)
    ov    = _serialize_overlays(klines, overlays or {})
    inds  = _calc_indicators(klines)

    oi_data = []
    if show_oi is not None and not show_oi.empty:
        for _, r in show_oi.iterrows():
            oi_data.append({"time":int(r["timestamp"].timestamp()),
                            "value":float(r["sumOpenInterestValue"])})

    candles_js = json.dumps(candles)
    volumes_js = json.dumps(volumes)
    cvds_js    = json.dumps(cvds)
    oi_js      = json.dumps(oi_data)
    ov_js      = json.dumps(ov)
    inds_js    = json.dumps(inds)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0f0f1a;overflow:hidden;font-family:'Segoe UI',monospace;color:#cdd6f4;}}
#root{{width:100%;height:{height}px;display:flex;flex-direction:column;}}

/* ── TOOLBARS ── */
#topbar{{
  display:flex;align-items:center;gap:3px;padding:4px 6px;
  background:#0f0f1a;border-bottom:1px solid #1e1e2e;flex-shrink:0;flex-wrap:wrap;
}}
.tbtn{{
  background:#1e1e2e;border:1px solid #313244;color:#cdd6f4;
  font-size:11px;padding:3px 7px;border-radius:4px;cursor:pointer;
  white-space:nowrap;user-select:none;transition:all .15s;
}}
.tbtn:hover{{background:#313244;}}
.tbtn.active{{background:#313244;border-color:#89b4fa;color:#89b4fa;}}
.tbtn.on{{border-color:#a6e3a1;color:#a6e3a1;background:#1e2a1e;}}
.sep{{width:1px;height:18px;background:#313244;margin:0 2px;}}
.tool-group{{display:flex;gap:2px;align-items:center;}}
.group-label{{font-size:9px;color:#585b70;padding:0 3px;}}

/* ── LEFT DRAWING TOOLBAR ── */
#drawtbar{{
  position:absolute;left:0;top:0;bottom:0;width:34px;
  background:#0d0d1a;border-right:1px solid #1e1e2e;
  display:flex;flex-direction:column;align-items:center;
  padding:4px 2px;gap:2px;z-index:30;overflow-y:auto;
}}
#drawtbar::-webkit-scrollbar{{width:3px;}}
#drawtbar::-webkit-scrollbar-thumb{{background:#313244;border-radius:2px;}}
.dtool{{
  width:28px;height:28px;border-radius:4px;border:1px solid transparent;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  font-size:13px;background:transparent;color:#cdd6f4;
  transition:all .15s;user-select:none;flex-shrink:0;
}}
.dtool:hover{{background:#1e1e2e;border-color:#313244;}}
.dtool.active{{background:#313244;border-color:#89b4fa;color:#89b4fa;}}
.dtool.sep{{width:26px;height:1px;background:#1e1e2e;cursor:default;}}
.dtool.sep:hover{{background:#1e1e2e;border:none;}}

/* ── CHART AREA ── */
#chartarea{{position:relative;flex:1;display:flex;flex-direction:column;}}
#chartinner{{position:relative;flex:1;display:flex;flex-direction:column;margin-left:34px;}}

/* ── PANELS ── */
.panel{{width:100%;position:relative;overflow:hidden;}}
.panel+.panel{{border-top:1px solid #1e1e2e;}}
#p_main{{flex:10;}}
#p_vol {{flex:3;}}
#p_cvd {{flex:3;}}
#p_oi  {{flex:3;}}
#p_rsi {{flex:3;}}
#p_macd{{flex:3;}}
#p_stoch{{flex:3;}}
.panel-label{{
  position:absolute;top:3px;left:6px;z-index:9;
  color:#585b70;font-size:9px;pointer-events:none;
  background:rgba(15,15,26,.7);padding:1px 4px;border-radius:2px;
}}

/* ── DRAWING CANVAS ── */
#draw-canvas{{
  position:absolute;top:0;left:0;width:100%;height:100%;
  z-index:20;pointer-events:none;
}}
#draw-canvas.active{{pointer-events:all;cursor:crosshair;}}
#draw-canvas.cursor-text{{cursor:text;}}
#draw-canvas.cursor-default{{cursor:default;}}

/* ── FULLSCREEN ── */
#root:fullscreen, #root:-webkit-full-screen {{
  width:100vw !important; height:100vh !important;
  background:#0f0f1a;
}}

/* ── TOOLTIP ── */
#tooltip{{
  position:absolute;background:rgba(15,15,26,.92);
  border:1px solid #313244;border-radius:5px;
  padding:5px 8px;font-size:10px;pointer-events:none;
  z-index:50;white-space:nowrap;display:none;
}}

/* ── CONTEXT MENU ── */
#ctx-menu{{
  position:fixed;background:#1e1e2e;border:1px solid #313244;
  border-radius:6px;padding:4px 0;z-index:100;display:none;min-width:130px;
}}
.ctx-item{{
  padding:5px 14px;font-size:11px;cursor:pointer;color:#cdd6f4;
}}
.ctx-item:hover{{background:#313244;}}
.ctx-item.danger{{color:#f38ba8;}}
.ctx-sep{{height:1px;background:#313244;margin:3px 0;}}

/* ── COLOR PICKER POPUP ── */
#color-popup{{
  position:fixed;background:#1e1e2e;border:1px solid #313244;
  border-radius:8px;padding:10px;z-index:110;display:none;
}}
.color-grid{{display:grid;grid-template-columns:repeat(7,22px);gap:4px;margin-bottom:8px;}}
.color-swatch{{width:22px;height:22px;border-radius:4px;cursor:pointer;border:2px solid transparent;}}
.color-swatch:hover{{border-color:#cdd6f4;}}
.color-swatch.selected{{border-color:#fff;}}
</style>
</head>
<body>

<div id="root">
  <!-- TOP TOOLBAR -->
  <div id="topbar">
    <!-- Cursor / Nav -->
    <div class="tool-group">
      <span class="group-label">Режим</span>
      <button class="tbtn active" id="btn_cursor" onclick="setMode('cursor')" title="Курсор (Esc)">↖</button>
      <button class="tbtn" id="btn_crosshair" onclick="setMode('crosshair')" title="Перекрестие">✛</button>
    </div>
    <div class="sep"></div>

    <!-- Panels -->
    <div class="tool-group">
      <span class="group-label">Панели</span>
      <button class="tbtn" id="btn_cvd"   onclick="togglePanel('cvd')"  title="CVD — Кумулятивная дельта">CVD</button>
      <button class="tbtn" id="btn_oi"    onclick="togglePanel('oi')"   title="OI — Открытый интерес">OI</button>
      <button class="tbtn" id="btn_rsi"   onclick="togglePanel('rsi')"  title="RSI 14">RSI</button>
      <button class="tbtn" id="btn_macd"  onclick="togglePanel('macd')" title="MACD 12/26/9">MACD</button>
      <button class="tbtn" id="btn_stoch" onclick="togglePanel('stoch')" title="Стохастик 14/3/3">Stoch</button>
    </div>
    <div class="sep"></div>

    <!-- Overlays on main chart -->
    <div class="tool-group">
      <span class="group-label">Индикаторы</span>
      <button class="tbtn" id="btn_ma20"  onclick="toggleInd('ma20')"  title="MA 20">MA20</button>
      <button class="tbtn" id="btn_ma50"  onclick="toggleInd('ma50')"  title="MA 50">MA50</button>
      <button class="tbtn" id="btn_ma200" onclick="toggleInd('ma200')" title="MA 200">MA200</button>
      <button class="tbtn" id="btn_ema21" onclick="toggleInd('ema21')" title="EMA 21">EMA21</button>
      <button class="tbtn" id="btn_bb"    onclick="toggleInd('bb')"    title="Bollinger Bands">BB</button>
      <button class="tbtn" id="btn_vwap"  onclick="toggleInd('vwap')"  title="VWAP">VWAP</button>
    </div>
    <div class="sep"></div>

    <!-- Overlays -->
    <div class="tool-group">
      <span class="group-label">SMC</span>
      <button class="tbtn on" id="btn_obs"   onclick="toggleOv('obs')"   title="Order Blocks">OB</button>
      <button class="tbtn on" id="btn_fvgs"  onclick="toggleOv('fvgs')"  title="Fair Value Gaps">FVG</button>
      <button class="tbtn on" id="btn_waves" onclick="toggleOv('waves')" title="Волновая разметка">Waves</button>
      <button class="tbtn on" id="btn_tdp"   onclick="toggleOv('tdp')"   title="TDP уровни">TDP</button>
      <button class="tbtn on" id="btn_sltp"  onclick="toggleOv('sltp')"  title="SL/TP уровни">SL/TP</button>
    </div>
    <div class="sep"></div>

    <!-- Actions -->
    <div class="tool-group">
      <button class="tbtn" onclick="fitContent()" title="Уместить всё">⊞ Fit</button>
      <button class="tbtn" id="btn_log" onclick="toggleLog()" title="Логарифмическая шкала">Log</button>
      <button class="tbtn" onclick="takeSnapshot()" title="Скриншот графика">📸</button>
      <button class="tbtn" onclick="toggleFullscreen()" id="btn_fs" title="Полноэкранный режим">⛶</button>
      <button class="tbtn" onclick="clearAllDrawings()" title="Удалить все рисунки">🗑 Очистить</button>
      <button class="tbtn" onclick="undoDrawing()" title="Отменить (Ctrl+Z)">↩ Undo</button>
    </div>
  </div>

  <!-- CHART AREA -->
  <div id="chartarea">
    <!-- LEFT DRAWING TOOLBAR -->
    <div id="drawtbar">
      <button class="dtool active" id="dt_cursor" onclick="selectTool('cursor')" title="Курсор (Esc)">↖</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_trendline" onclick="selectTool('trendline')" title="Линия тренда (T)">╱</button>
      <button class="dtool" id="dt_ray" onclick="selectTool('ray')" title="Луч (R)">→</button>
      <button class="dtool" id="dt_extline" onclick="selectTool('extline')" title="Расш. линия">↔</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_hline" onclick="selectTool('hline')" title="Горизонтальная линия (H)">─</button>
      <button class="dtool" id="dt_vline" onclick="selectTool('vline')" title="Вертикальная линия (V)">│</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_channel" onclick="selectTool('channel')" title="Параллельный канал">⫠</button>
      <button class="dtool" id="dt_rect" onclick="selectTool('rect')" title="Прямоугольник (B)">▭</button>
      <button class="dtool" id="dt_triangle" onclick="selectTool('triangle')" title="Треугольник">△</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_fib" onclick="selectTool('fib')" title="Фибоначчи ретрейсмент (F)">𝑓</button>
      <button class="dtool" id="dt_fibext" onclick="selectTool('fibext')" title="Фибоначчи экстеншн">𝑓⁺</button>
      <button class="dtool" id="dt_fibcircle" onclick="selectTool('fibcircle')" title="Фибо-дуга">◌</button>
      <button class="dtool" id="dt_pitchfork" onclick="selectTool('pitchfork')" title="Вилы Эндрюса">𝔽</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_arrow" onclick="selectTool('arrow')" title="Стрелка">↑</button>
      <button class="dtool" id="dt_text" onclick="selectTool('text')" title="Текст (L)">𝐓</button>
      <div class="dtool sep"></div>

      <button class="dtool" id="dt_measure" onclick="selectTool('measure')" title="Измерение">📏</button>
      <button class="dtool" id="dt_eraser" onclick="selectTool('eraser')" title="Ластик (E)">⌫</button>
    </div>

    <!-- CHART PANELS -->
    <div id="chartinner">
      <canvas id="draw-canvas"></canvas>
      <div class="panel" id="p_main">
        <div class="panel-label" id="lbl_main">{symbol} · {timeframe}</div>
      </div>
      <div class="panel" id="p_vol">
        <div class="panel-label">Volume</div>
      </div>
      <div class="panel" id="p_cvd" style="display:none">
        <div class="panel-label">CVD (Cumulative Volume Delta)</div>
      </div>
      <div class="panel" id="p_oi" style="display:none">
        <div class="panel-label">Open Interest</div>
      </div>
      <div class="panel" id="p_rsi" style="display:none">
        <div class="panel-label">RSI (14)</div>
      </div>
      <div class="panel" id="p_macd" style="display:none">
        <div class="panel-label">MACD (12,26,9)</div>
      </div>
      <div class="panel" id="p_stoch" style="display:none">
        <div class="panel-label">Stochastic (14,3,3)</div>
      </div>
    </div>
  </div>
</div>

<!-- TOOLTIP -->
<div id="tooltip"></div>

<!-- CONTEXT MENU -->
<div id="ctx-menu">
  <div class="ctx-item" onclick="ctxAction('color')">🎨 Изменить цвет</div>
  <div class="ctx-item" onclick="ctxAction('style')">─ Стиль линии</div>
  <div class="ctx-item" onclick="ctxAction('width')">✦ Толщина</div>
  <div class="ctx-sep"></div>
  <div class="ctx-item" onclick="ctxAction('duplicate')">⧉ Дублировать</div>
  <div class="ctx-sep"></div>
  <div class="ctx-item danger" onclick="ctxAction('delete')">🗑 Удалить</div>
</div>

<!-- COLOR POPUP -->
<div id="color-popup">
  <div class="color-grid" id="color-grid"></div>
  <input type="color" id="color-custom" style="width:100%;margin-top:4px;border:none;background:none;cursor:pointer;height:24px;">
</div>

<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<script>
// ═══════════════════════════════════════════════════════════
//  DATA
// ═══════════════════════════════════════════════════════════
const CANDLES = {candles_js};
const VOLUMES = {volumes_js};
const CVDS    = {cvds_js};
const OI_DATA = {oi_js};
const OV      = {ov_js};
const INDS    = {inds_js};

const LC = LightweightCharts;

// ═══════════════════════════════════════════════════════════
//  CHART OPTIONS
// ═══════════════════════════════════════════════════════════
const CHART_OPTS = {{
  layout: {{ background:{{type:'solid',color:'#0f0f1a'}}, textColor:'#cdd6f4',
             fontSize:11, fontFamily:"'Segoe UI',monospace" }},
  grid:   {{ vertLines:{{color:'rgba(49,50,68,.4)'}}, horzLines:{{color:'rgba(49,50,68,.4)'}} }},
  crosshair: {{ mode:1 }},
  timeScale: {{ borderColor:'#313244', timeVisible:true, secondsVisible:false }},
  rightPriceScale: {{ borderColor:'#313244' }},
  handleScrolls: {{ mouseWheel:true, pressedMouseMove:true }},
  handleScale:   {{ mouseWheel:true, axisPressedMouseMove:true }},
}};

// ═══════════════════════════════════════════════════════════
//  CREATE CHARTS
// ═══════════════════════════════════════════════════════════
const charts = {{}};
const series = {{}};

function makeChart(panelId, opts = {{}}) {{
  const el = document.getElementById(panelId);
  const c  = LC.createChart(el, {{ ...CHART_OPTS, ...opts,
    width: el.clientWidth, height: el.clientHeight }});
  charts[panelId] = c;
  return c;
}}

// Main
const cMain  = makeChart('p_main');
const sCand  = cMain.addCandlestickSeries({{
  upColor:'#a6e3a1', downColor:'#f38ba8',
  borderUpColor:'#a6e3a1', borderDownColor:'#f38ba8',
  wickUpColor:'#a6e3a1', wickDownColor:'#f38ba8',
}});
sCand.setData(CANDLES);
series['candle'] = sCand;

// Volume
const cVol   = makeChart('p_vol', {{ rightPriceScale:{{ visible:true, scaleMargins:{{top:.1,bottom:.0}} }} }});
const sVol   = cVol.addHistogramSeries({{ priceFormat:{{type:'volume'}}, priceScaleId:'right' }});
sVol.setData(VOLUMES);

// CVD
let sCVD;
const cCVD   = makeChart('p_cvd');
sCVD = cCVD.addHistogramSeries({{ priceScaleId:'right' }});
sCVD.setData(CVDS);

// OI
let sOI;
const cOI    = makeChart('p_oi');
sOI  = cOI.addAreaSeries({{ lineColor:'#cba6f7', topColor:'rgba(203,166,247,.3)', bottomColor:'rgba(203,166,247,.0)' }});
if (OI_DATA.length) sOI.setData(OI_DATA);

// RSI
const cRSI   = makeChart('p_rsi', {{ rightPriceScale:{{ visible:true }} }});
const sRSI   = cRSI.addLineSeries({{ color:'#f9e2af', lineWidth:1 }});
sRSI.setData(INDS.rsi || []);
cRSI.addLineSeries({{ color:'#f38ba8', lineWidth:1, lineStyle:2 }}).setData((INDS.rsi||[]).map(p=>{{return{{time:p.time,value:70}}}}));
cRSI.addLineSeries({{ color:'#a6e3a1', lineWidth:1, lineStyle:2 }}).setData((INDS.rsi||[]).map(p=>{{return{{time:p.time,value:30}}}}));

// MACD
const cMACD  = makeChart('p_macd');
const sMACD_hist   = cMACD.addHistogramSeries({{ priceScaleId:'right' }});
const sMACD_line   = cMACD.addLineSeries({{ color:'#89b4fa', lineWidth:1 }});
const sMACD_signal = cMACD.addLineSeries({{ color:'#f38ba8', lineWidth:1 }});
sMACD_hist.setData(INDS.macd_hist || []);
sMACD_line.setData(INDS.macd_line || []);
sMACD_signal.setData(INDS.macd_signal || []);

// Stoch
const cStoch = makeChart('p_stoch');
const sStK   = cStoch.addLineSeries({{ color:'#89b4fa', lineWidth:1 }});
const sStD   = cStoch.addLineSeries({{ color:'#f38ba8', lineWidth:1 }});
sStK.setData(INDS.stoch_k || []);
sStD.setData(INDS.stoch_d || []);
cStoch.addLineSeries({{ color:'#f38ba8', lineWidth:1, lineStyle:2 }}).setData((INDS.stoch_k||[]).map(p=>{{return{{time:p.time,value:80}}}}));
cStoch.addLineSeries({{ color:'#a6e3a1', lineWidth:1, lineStyle:2 }}).setData((INDS.stoch_k||[]).map(p=>{{return{{time:p.time,value:20}}}}));

// ═══════════════════════════════════════════════════════════
//  INDICATORS ON MAIN CHART
// ═══════════════════════════════════════════════════════════
const indSeries = {{}};
const indActive = {{ ma20:false, ma50:false, ma200:false, ema21:false, bb:false, vwap:false }};
const ovActive  = {{ obs:true, fvgs:true, waves:true, tdp:true, sltp:true }};

function _addLine(data, color, lw=1, style=0) {{
  const s = cMain.addLineSeries({{color, lineWidth:lw, lineStyle:style, priceLineVisible:false, lastValueVisible:false}});
  s.setData(data || []);
  return s;
}}

indSeries.ma20  = _addLine(INDS.ma20,  '#f9e2af', 1);
indSeries.ma50  = _addLine(INDS.ma50,  '#89b4fa', 1);
indSeries.ma200 = _addLine(INDS.ma200, '#cba6f7', 1);
indSeries.ema21 = _addLine(INDS.ema21, '#a6e3a1', 1);
indSeries.bb_up = _addLine(INDS.bb_up, '#94e2d5', 1, 2);
indSeries.bb_dn = _addLine(INDS.bb_dn, '#94e2d5', 1, 2);
indSeries.bb_mid= _addLine(INDS.bb_mid,'#94e2d5', 1, 2);
indSeries.vwap  = _addLine(INDS.vwap,  '#fab387', 1, 0);

// All indicators hidden initially
['ma20','ma50','ma200','ema21','bb_up','bb_dn','bb_mid','vwap'].forEach(k => {{
  if (indSeries[k]) indSeries[k].applyOptions({{ visible:false }});
}});

function toggleInd(name) {{
  const btn = document.getElementById('btn_' + name);
  const on  = !indActive[name];
  indActive[name] = on;
  btn.classList.toggle('on', on);
  if (name === 'bb') {{
    ['bb_up','bb_dn','bb_mid'].forEach(k => indSeries[k]?.applyOptions({{visible:on}}));
  }} else {{
    indSeries[name]?.applyOptions({{visible:on}});
  }}
}}

function toggleOv(name) {{
  const btn = document.getElementById('btn_' + name);
  ovActive[name] = !ovActive[name];
  btn.classList.toggle('on', ovActive[name]);
  redrawOverlays();
}}

// ═══════════════════════════════════════════════════════════
//  SYNC TIMESCALES
// ═══════════════════════════════════════════════════════════
const allCharts = [cMain, cVol, cCVD, cOI, cRSI, cMACD, cStoch];
let syncing = false;
allCharts.forEach(src => {{
  src.timeScale().subscribeVisibleTimeRangeChange(range => {{
    if (syncing) return; syncing = true;
    allCharts.forEach(dst => {{ if (dst !== src) dst.timeScale().setVisibleRange(range); }});
    syncing = false;
    renderDrawings();
  }});
}});

// ═══════════════════════════════════════════════════════════
//  OVERLAY DRAWINGS (OB, FVG, etc.)
// ═══════════════════════════════════════════════════════════
// We draw overlays as price lines and bands using Lightweight Charts primitives
const priceLines = [];

function clearPriceLines() {{
  priceLines.forEach(pl => {{ try {{ sCand.removePriceLine(pl); }} catch(e) {{}} }});
  priceLines.length = 0;
}}

function addBand(low, high, color, alpha=0.12) {{
  const up = sCand.createPriceLine({{ price:high, color:color+'88', lineWidth:1, lineStyle:2, axisLabelVisible:false, title:'' }});
  const dn = sCand.createPriceLine({{ price:low,  color:color+'88', lineWidth:1, lineStyle:2, axisLabelVisible:false, title:'' }});
  priceLines.push(up, dn);
}}

function addPriceLine(price, color, title='', lw=1, style=0) {{
  if (!price || price === 0) return;
  const pl = sCand.createPriceLine({{ price, color, lineWidth:lw, lineStyle:style, axisLabelVisible:true, title }});
  priceLines.push(pl);
}}

function redrawOverlays() {{
  clearPriceLines();
  // OB
  if (ovActive.obs) OV.obs?.forEach(ob => {{
    addBand(ob.low, ob.high, ob.kind==='bullish' ? '#a6e3a1' : '#f38ba8');
  }});
  // FVG
  if (ovActive.fvgs) OV.fvgs?.forEach(f => {{
    addBand(f.low, f.high, f.kind==='bullish' ? '#89b4fa' : '#cba6f7');
  }});
  // TDP
  if (ovActive.tdp) OV.tdp?.forEach(t => {{
    addPriceLine(t.price, '#f9e2af', t.desc, 1, 1);
  }});
  // H-lines
  OV.h_lines?.forEach(hl => {{
    addPriceLine(hl.price, hl.color||'#f9e2af', hl.label||'', 1, 0);
  }});
  // SL/TP
  if (ovActive.sltp) {{
    if (OV.entry?.low && OV.entry?.high) {{ addBand(OV.entry.low, OV.entry.high, '#f9e2af'); }}
    addPriceLine(OV.sl,  '#f38ba8', 'SL', 1, 2);
    addPriceLine(OV.tp1, '#a6e3a1', 'TP1',1, 2);
    addPriceLine(OV.tp2, '#a6e3a1', 'TP2',1, 2);
  }}
  // Fib
  const fibColors = ['#f5c2e7','#cba6f7','#89b4fa','#94e2d5','#a6e3a1','#f9e2af','#fab387'];
  let fi = 0;
  for (const [ratio, price] of Object.entries(OV.fibs||{{}})) {{
    addPriceLine(+price, fibColors[fi++ % fibColors.length], ratio, 1, 1);
  }}
}}

redrawOverlays();

// Wave markers as series markers
function drawWaveMarkers() {{
  if (!ovActive.waves || !OV.wave_marks?.length) {{ sCand.setMarkers([]); return; }}
  const markers = OV.wave_marks.map(m => ({{
    time: m.time,
    position: m.kind==='L' ? 'belowBar' : 'aboveBar',
    color: m.kind==='L' ? '#a6e3a1' : '#f38ba8',
    shape: m.kind==='L' ? 'arrowUp' : 'arrowDown',
    text: m.label, size: 1,
  }}));
  try {{ sCand.setMarkers(markers); }} catch(e) {{}}
}}
drawWaveMarkers();

// ═══════════════════════════════════════════════════════════
//  PANEL TOGGLE
// ═══════════════════════════════════════════════════════════
const panels = {{ cvd:'p_cvd', oi:'p_oi', rsi:'p_rsi', macd:'p_macd', stoch:'p_stoch' }};
const panelVis = {{ cvd:false, oi:false, rsi:false, macd:false, stoch:false }};

function togglePanel(name) {{
  panelVis[name] = !panelVis[name];
  const el = document.getElementById(panels[name]);
  el.style.display = panelVis[name] ? 'block' : 'none';
  document.getElementById('btn_' + name).classList.toggle('active', panelVis[name]);
  resizeAll();
}}

function fitContent() {{
  allCharts.forEach(c => c.timeScale().fitContent());
}}

function focusOnLast(n) {{
  if (!CANDLES.length) return;
  const last = CANDLES[CANDLES.length - 1];
  const from = CANDLES[Math.max(0, CANDLES.length - n)];
  allCharts.forEach(c => {{
    try {{ c.timeScale().setVisibleRange({{ from: from.time, to: last.time }}); }} catch(e) {{}}
  }});
}}

let logMode = false;
function toggleLog() {{
  logMode = !logMode;
  document.getElementById('btn_log').classList.toggle('on', logMode);
  cMain.applyOptions({{ rightPriceScale: {{ mode: logMode ? 1 : 0 }} }});
}}

function takeSnapshot() {{
  // Merge chart canvas + draw canvas into one image
  const chartEl = document.getElementById('p_main');
  const chartCanvas = chartEl.querySelector('canvas');
  if (!chartCanvas) return;
  const drawCanvas = document.getElementById('draw-canvas');
  const merged = document.createElement('canvas');
  merged.width  = chartCanvas.width;
  merged.height = chartCanvas.height;
  const mctx = merged.getContext('2d');
  mctx.drawImage(chartCanvas, 0, 0);
  mctx.drawImage(drawCanvas,  0, 0, drawCanvas.width, drawCanvas.height, 0, 0, merged.width, merged.height);
  const link = document.createElement('a');
  link.download = 'chart_snapshot.png';
  link.href = merged.toDataURL('image/png');
  link.click();
}}

function toggleFullscreen() {{
  const root = document.getElementById('root');
  if (!document.fullscreenElement) {{
    root.requestFullscreen().catch(e => {{}});
    document.getElementById('btn_fs').textContent = '✕';
  }} else {{
    document.exitFullscreen();
    document.getElementById('btn_fs').textContent = '⛶';
  }}
  setTimeout(() => {{ resizeAll(); renderDrawings(); }}, 300);
}}

document.addEventListener('fullscreenchange', () => {{
  if (!document.fullscreenElement) {{
    document.getElementById('btn_fs').textContent = '⛶';
    setTimeout(() => {{ resizeAll(); renderDrawings(); }}, 300);
  }}
}});

function setMode(m) {{
  document.getElementById('btn_cursor').classList.toggle('active', m==='cursor');
  document.getElementById('btn_crosshair').classList.toggle('active', m==='crosshair');
  allCharts.forEach(c => c.applyOptions({{
    crosshair: {{ mode: m==='crosshair' ? 0 : 1 }}
  }}));
}}

// ═══════════════════════════════════════════════════════════
//  RESIZE
// ═══════════════════════════════════════════════════════════
function resizeAll() {{
  allCharts.forEach(c => {{
    const el = c.chartElement().parentElement;
    if (!el || !el.offsetWidth) return;
    c.resize(el.clientWidth, el.clientHeight);
  }});
  resizeDrawCanvas();
  renderDrawings();
}}

const ro = new ResizeObserver(resizeAll);
ro.observe(document.getElementById('root'));
setTimeout(resizeAll, 100);
setTimeout(() => {{
  // Show last 150 candles by default instead of all history
  if (CANDLES.length > 150) {{
    focusOnLast(150);
  }} else {{
    fitContent();
  }}
  renderDrawings();
}}, 300);

// ═══════════════════════════════════════════════════════════
//  ══════════════ DRAWING ENGINE ══════════════
// ═══════════════════════════════════════════════════════════
const canvas   = document.getElementById('draw-canvas');
const ctx2d    = canvas.getContext('2d');
let currentTool = 'cursor';
let drawing    = false;
let startPt    = null;    // {{x,y}} in canvas pixels
let tempPt     = null;
let tempPts    = [];      // for multi-point tools (channel, pitchfork, triangle)
const drawings = [];      // all finished drawings
let selectedIdx= -1;      // index of selected drawing
let isDragging = false;
let dragOffset = null;
let history    = [];      // undo stack

// Current style
let drawColor  = '#f9e2af';
let lineWidth  = 1.5;
let lineStyle  = 'solid'; // solid | dashed | dotted

// ─── CANVAS SIZE ───
function resizeDrawCanvas() {{
  const inner = document.getElementById('chartinner');
  const bb    = inner.getBoundingClientRect();
  canvas.width  = bb.width;
  canvas.height = bb.height;
  canvas.style.width  = bb.width  + 'px';
  canvas.style.height = bb.height + 'px';
}}

// ─── COORD CONVERTERS ───
function getMainRect() {{
  const main = document.getElementById('p_main');
  return main.getBoundingClientRect();
}}
function getInnerRect() {{
  const inner = document.getElementById('chartinner');
  return inner.getBoundingClientRect();
}}

function canvasToChart(cx, cy) {{
  // Returns {{price, time}} from canvas coordinates
  const mainRect = getMainRect();
  const innerRect = getInnerRect();
  const relX = cx + innerRect.left - mainRect.left;
  const relY = cy + innerRect.top  - mainRect.top;
  if (relY < 0 || relY > mainRect.height) return null;
  const price = sCand.coordinateToPrice(relY);
  const time  = cMain.timeScale().coordinateToTime(relX);
  return (price !== null && time !== null) ? {{price, time, x:cx, y:cy}} : null;
}}

function chartToCanvas(price, time) {{
  // Returns {{x,y}} canvas coordinates
  const mainRect  = getMainRect();
  const innerRect = getInnerRect();
  const x = cMain.timeScale().timeToCoordinate(time);
  const y = sCand.priceToCoordinate(price);
  if (x === null || y === null) return null;
  const cx = x + mainRect.left - innerRect.left;
  const cy = y + mainRect.top  - innerRect.top;
  return {{x:cx, y:cy}};
}}

function canvasXToTime(cx) {{
  const mainRect  = getMainRect();
  const innerRect = getInnerRect();
  const relX = cx + innerRect.left - mainRect.left;
  return cMain.timeScale().coordinateToTime(relX);
}}
function canvasYToPrice(cy) {{
  const mainRect  = getMainRect();
  const innerRect = getInnerRect();
  const relY = cy + innerRect.top - mainRect.top;
  return sCand.coordinateToPrice(relY);
}}
function timeToCx(time) {{
  const mainRect  = getMainRect();
  const innerRect = getInnerRect();
  const x = cMain.timeScale().timeToCoordinate(time);
  return x === null ? null : x + mainRect.left - innerRect.left;
}}
function priceToCy(price) {{
  const mainRect  = getMainRect();
  const innerRect = getInnerRect();
  const y = sCand.priceToCoordinate(price);
  return y === null ? null : y + mainRect.top - innerRect.top;
}}

// ─── TOOL SELECTION ───
function selectTool(tool) {{
  currentTool = tool;
  drawing = false; startPt = null; tempPts = [];
  // Update button states
  document.querySelectorAll('.dtool').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('dt_' + tool);
  if (btn) btn.classList.add('active');
  // Canvas pointer events
  canvas.className = 'active';
  if (tool === 'cursor') {{ canvas.className = 'cursor-default'; }}
  else if (tool === 'text') {{ canvas.className += ' cursor-text'; }}
  selectedIdx = -1;
  renderDrawings();
}}

// ─── CANVAS EVENTS ───
canvas.addEventListener('mousedown', onMouseDown);
canvas.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('mouseup',   onMouseUp);
canvas.addEventListener('dblclick',  onDblClick);
canvas.addEventListener('contextmenu', onContextMenu);
document.addEventListener('keydown', onKeyDown);

function getCanvasPos(e) {{
  const rect = canvas.getBoundingClientRect();
  return {{ x: e.clientX - rect.left, y: e.clientY - rect.top }};
}}

function onMouseDown(e) {{
  if (e.button === 2) return;
  const pos = getCanvasPos(e);

  if (currentTool === 'cursor') {{
    // Selection / drag
    selectedIdx = hitTest(pos);
    if (selectedIdx >= 0) {{
      isDragging  = true;
      const d     = drawings[selectedIdx];
      dragOffset  = {{ dx: pos.x - (d.p1?.x||0), dy: pos.y - (d.p1?.y||0) }};
    }}
    renderDrawings();
    return;
  }}

  if (currentTool === 'eraser') {{
    const idx = hitTest(pos);
    if (idx >= 0) {{ saveHistory(); drawings.splice(idx,1); renderDrawings(); }}
    return;
  }}

  if (currentTool === 'text') {{
    const lbl = prompt('Текст метки:', '');
    if (!lbl) return;
    saveHistory();
    drawings.push({{ tool:'text', p1:pos, text:lbl, color:drawColor, fontSize:13 }});
    renderDrawings();
    return;
  }}

  // Multi-point tools
  if (['channel','triangle','pitchfork'].includes(currentTool)) {{
    tempPts.push(pos);
    if ((currentTool === 'channel'   && tempPts.length === 3) ||
        (currentTool === 'triangle'  && tempPts.length === 3) ||
        (currentTool === 'pitchfork' && tempPts.length === 3)) {{
      saveHistory();
      drawings.push(buildDrawing(currentTool, [...tempPts]));
      tempPts = [];
    }}
    return;
  }}

  // Two-point tools
  drawing  = true;
  startPt  = pos;
  tempPt   = pos;
}}

function onMouseMove(e) {{
  const pos = getCanvasPos(e);
  updateTooltip(e, pos);

  if (currentTool === 'cursor' && isDragging && selectedIdx >= 0) {{
    const d   = drawings[selectedIdx];
    const dx  = pos.x - (dragOffset.dx + (d.p1?.x||0));
    const dy  = pos.y - (dragOffset.dy + (d.p1?.y||0));
    moveDrawing(d, dx, dy);
    dragOffset = {{ dx: pos.x - (d.p1?.x||0), dy: pos.y - (d.p1?.y||0) }};
    renderDrawings();
    return;
  }}

  if (drawing) {{
    tempPt = pos;
    renderDrawings();
    drawTemp(startPt, tempPt);
  }}
  if (tempPts.length > 0) {{
    renderDrawings();
    drawTempMulti([...tempPts, pos]);
  }}
}}

function onMouseUp(e) {{
  if (currentTool === 'cursor') {{ isDragging = false; return; }}
  if (!drawing) return;
  drawing = false;
  const pos = getCanvasPos(e);
  if (Math.abs(pos.x - startPt.x) < 3 && Math.abs(pos.y - startPt.y) < 3 &&
      !['hline','vline'].includes(currentTool)) return;

  saveHistory();
  const d = buildDrawing(currentTool, [startPt, pos]);
  if (d) drawings.push(d);
  startPt = null; tempPt = null;
  renderDrawings();
}}

function onDblClick(e) {{
  const pos = getCanvasPos(e);
  if (currentTool === 'cursor') {{
    const idx = hitTest(pos);
    if (idx >= 0) showContextMenuForDrawing(idx, e.clientX, e.clientY);
  }}
}}

function onKeyDown(e) {{
  if (e.key === 'Escape') {{ selectTool('cursor'); }}
  if ((e.ctrlKey||e.metaKey) && e.key === 'z') undoDrawing();
  if (e.key === 'Delete' && selectedIdx >= 0) {{
    saveHistory(); drawings.splice(selectedIdx,1); selectedIdx=-1; renderDrawings();
  }}
  // Hotkeys
  const hot = {{t:'trendline',r:'ray',h:'hline',v:'vline',f:'fib',b:'rect',e:'eraser',l:'text'}};
  if (!e.ctrlKey && !e.metaKey && hot[e.key.toLowerCase()]) selectTool(hot[e.key.toLowerCase()]);
}}

// ─── BUILD DRAWING ───
function buildDrawing(tool, pts) {{
  const p1 = pts[0], p2 = pts[1] || pts[0];
  const base = {{ tool, p1, p2, color:drawColor, width:lineWidth, style:lineStyle }};
  if (tool === 'channel' || tool === 'triangle' || tool === 'pitchfork') {{
    const ptsPT = pts.map(p => canvasToChart(p.x, p.y));
    return {{ ...base, pts:[...pts], ptsPT }};
  }}
  // Store price/time for persistence across zoom/pan
  const pt1 = canvasToChart(p1.x, p1.y);
  const pt2 = canvasToChart(p2.x, p2.y);
  if (!pt1) return null;
  return {{ ...base, pt1, pt2: pt2||pt1 }};
}}

function moveDrawing(d, dx, dy) {{
  if (d.pts) {{
    d.pts = d.pts.map(p => p ? ({{x:p.x+dx, y:p.y+dy}}) : p);
    if (d.ptsPT) d.ptsPT = d.pts.map(p => p ? canvasToChart(p.x, p.y) : null);
    if (d.pts[0]) {{ d.p1 = d.pts[0]; d.p2 = d.pts[1]||d.pts[0]; }}
    return;
  }}
  d.p1 = {{x:(d.p1?.x||0)+dx, y:(d.p1?.y||0)+dy}};
  d.p2 = {{x:(d.p2?.x||0)+dx, y:(d.p2?.y||0)+dy}};
  if (d.pt1) {{ const c = canvasToChart(d.p1.x, d.p1.y); if (c) d.pt1 = c; }}
  if (d.pt2) {{ const c = canvasToChart(d.p2.x, d.p2.y); if (c) d.pt2 = c; }}
}}

// ─── PATTERN SHAPES (server-side, price+time anchored) ───
function drawPatternShapes() {{
  if (!OV.pattern_shapes?.length) return;
  OV.pattern_shapes.forEach(shape => {{
    const pts = (shape.points||[]).map(pt => chartToCanvas(pt.price, pt.time)).filter(Boolean);
    if (pts.length < 2) return;
    ctx2d.save();
    ctx2d.strokeStyle = shape.color || '#f9e2af';
    ctx2d.lineWidth = 2.5;
    ctx2d.setLineDash([]);
    ctx2d.shadowColor = shape.color || '#f9e2af';
    ctx2d.shadowBlur = 6;
    ctx2d.beginPath();
    ctx2d.moveTo(pts[0].x, pts[0].y);
    pts.slice(1).forEach(p => ctx2d.lineTo(p.x, p.y));
    ctx2d.stroke();
    pts.forEach(p => {{
      ctx2d.beginPath();
      ctx2d.arc(p.x, p.y, 4, 0, Math.PI*2);
      ctx2d.fillStyle = shape.color || '#f9e2af';
      ctx2d.shadowBlur = 0;
      ctx2d.fill();
    }});
    if (shape.label && pts[0]) {{
      ctx2d.font = 'bold 11px monospace';
      ctx2d.fillStyle = shape.color || '#f9e2af';
      ctx2d.shadowBlur = 0;
      ctx2d.fillText(shape.label, pts[0].x + 4, pts[0].y - 8);
    }}
    ctx2d.restore();
  }});
}}

// ─── RENDER ALL DRAWINGS ───
function renderDrawings() {{
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);
  // Pattern shapes stay anchored to price/time
  drawPatternShapes();
  drawings.forEach((d, i) => {{
    // Multi-point tools: resolve canvas coords from stored price/time
    if (d.ptsPT?.length) {{
      d.pts = d.ptsPT.map(pt => pt ? chartToCanvas(pt.price, pt.time) : null);
      const valid = d.pts.filter(Boolean);
      if (valid.length) {{ d.p1 = valid[0]; d.p2 = valid[1]||valid[0]; }}
    }}
    // Two-point tools: resolve from stored price/time
    let p1 = d.p1, p2 = d.p2;
    if (d.pt1) {{ const r = chartToCanvas(d.pt1.price, d.pt1.time); if (r) {{ d.p1 = p1 = r; }} }}
    if (d.pt2) {{ const r = chartToCanvas(d.pt2.price, d.pt2.time); if (r) {{ d.p2 = p2 = r; }} }}
    drawShape(d, i === selectedIdx);
  }});
}}

function drawTemp(p1, p2) {{
  drawShape({{ tool:currentTool, p1, p2, color:drawColor, width:lineWidth, style:lineStyle }}, false, true);
}}
function drawTempMulti(pts) {{
  if (pts.length >= 2) drawShape({{ tool:currentTool, pts, color:drawColor, width:lineWidth, style:lineStyle }}, false, true);
}}

// ─── SHAPE RENDERER ───
const FIB_RATIOS_DRAW = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
const FIB_EXT_RATIOS  = [0, 0.618, 1.0, 1.272, 1.618, 2.0, 2.618];
const FIB_COLORS = ['#a6e3a1','#89b4fa','#cba6f7','#f9e2af','#fab387','#f5c2e7','#94e2d5'];

function setCtxStyle(d, selected) {{
  ctx2d.strokeStyle = selected ? '#fff' : (d.color || '#f9e2af');
  ctx2d.fillStyle   = selected ? 'rgba(255,255,255,.08)' : (d.color || '#f9e2af') + '20';
  ctx2d.lineWidth   = (d.width || 1.5) * (selected ? 1.3 : 1);
  ctx2d.setLineDash(d.style==='dashed' ? [6,3] : d.style==='dotted' ? [2,3] : []);
}}

function drawShape(d, selected=false, temp=false) {{
  if (!d.p1) return;
  const p1 = d.p1, p2 = d.p2 || p1;
  setCtxStyle(d, selected);
  ctx2d.beginPath();

  switch (d.tool) {{
    case 'trendline':
      ctx2d.moveTo(p1.x, p1.y); ctx2d.lineTo(p2.x, p2.y);
      ctx2d.stroke();
      break;

    case 'ray': {{
      // Extend from p1 through p2 to edge
      const [ex,ey] = extendToEdge(p1, p2);
      ctx2d.moveTo(p1.x, p1.y); ctx2d.lineTo(ex, ey);
      ctx2d.stroke();
      break;
    }}

    case 'extline': {{
      const [ex1,ey1] = extendToEdge(p2, p1);
      const [ex2,ey2] = extendToEdge(p1, p2);
      ctx2d.moveTo(ex1, ey1); ctx2d.lineTo(ex2, ey2);
      ctx2d.stroke();
      break;
    }}

    case 'hline':
      ctx2d.moveTo(0, p1.y); ctx2d.lineTo(canvas.width, p1.y);
      ctx2d.stroke();
      if (!temp) {{ ctx2d.font = '10px monospace'; ctx2d.fillStyle = d.color||'#f9e2af';
        const price = d.pt1 ? d.pt1.price.toPrecision(6) : '';
        ctx2d.fillText(price, canvas.width - 80, p1.y - 3); }}
      break;

    case 'vline':
      ctx2d.moveTo(p1.x, 0); ctx2d.lineTo(p1.x, canvas.height);
      ctx2d.stroke();
      break;

    case 'rect': {{
      const w = p2.x-p1.x, h = p2.y-p1.y;
      ctx2d.fillRect(p1.x, p1.y, w, h);
      ctx2d.strokeRect(p1.x, p1.y, w, h);
      break;
    }}

    case 'triangle': {{
      const pts = d.pts || [p1, p2, {{x:p1.x+(p2.x-p1.x)/2, y:p1.y-80}}];
      ctx2d.moveTo(pts[0].x, pts[0].y);
      pts.slice(1).forEach(p => ctx2d.lineTo(p.x, p.y));
      ctx2d.closePath();
      ctx2d.fill(); ctx2d.stroke();
      break;
    }}

    case 'channel': {{
      const pts = d.pts || [p1, p2, {{x:p2.x, y:p2.y+50}}];
      // Line 1: pts[0]→pts[1], Line 2: parallel offset by pts[2]
      const dx = pts[1].x-pts[0].x, dy = pts[1].y-pts[0].y;
      const ox = pts[2] ? pts[2].x-pts[0].x : 0;
      const oy = pts[2] ? pts[2].y-pts[0].y : 50;
      ctx2d.moveTo(pts[0].x, pts[0].y); ctx2d.lineTo(pts[1].x, pts[1].y);
      ctx2d.moveTo(pts[0].x+ox, pts[0].y+oy); ctx2d.lineTo(pts[1].x+ox, pts[1].y+oy);
      ctx2d.stroke();
      // Fill between
      ctx2d.beginPath();
      ctx2d.moveTo(pts[0].x,pts[0].y);ctx2d.lineTo(pts[1].x,pts[1].y);
      ctx2d.lineTo(pts[1].x+ox,pts[1].y+oy);ctx2d.lineTo(pts[0].x+ox,pts[0].y+oy);
      ctx2d.closePath();ctx2d.fill();
      break;
    }}

    case 'pitchfork': {{
      const pts = d.pts || [p1, p2, {{x:p2.x+20,y:p2.y+20}}];
      if (pts.length < 3) {{ ctx2d.moveTo(pts[0].x,pts[0].y); if(pts[1]) ctx2d.lineTo(pts[1].x,pts[1].y); ctx2d.stroke(); break; }}
      const mid = {{ x:(pts[1].x+pts[2].x)/2, y:(pts[1].y+pts[2].y)/2 }};
      // Handle
      ctx2d.moveTo(pts[0].x,pts[0].y); ctx2d.lineTo(mid.x, mid.y); ctx2d.stroke();
      // Prongs
      const [e1x,e1y] = extendToEdge(mid, pts[1]);
      const [e2x,e2y] = extendToEdge(mid, pts[2]);
      ctx2d.beginPath(); setCtxStyle(d,selected);
      ctx2d.moveTo(pts[1].x,pts[1].y); ctx2d.lineTo(e1x,e1y); ctx2d.stroke();
      ctx2d.beginPath();
      ctx2d.moveTo(pts[2].x,pts[2].y); ctx2d.lineTo(e2x,e2y); ctx2d.stroke();
      break;
    }}

    case 'arrow': {{
      ctx2d.moveTo(p1.x, p1.y); ctx2d.lineTo(p2.x, p2.y); ctx2d.stroke();
      drawArrowHead(p1, p2, d.color||'#f9e2af');
      break;
    }}

    case 'text':
      ctx2d.font = `${{d.fontSize||13}}px 'Segoe UI',monospace`;
      ctx2d.fillStyle = d.color || '#f9e2af';
      ctx2d.setLineDash([]);
      ctx2d.fillText(d.text||'', p1.x, p1.y);
      if (selected) {{
        const m = ctx2d.measureText(d.text||'');
        ctx2d.strokeStyle = '#89b4fa'; ctx2d.lineWidth = 1;
        ctx2d.strokeRect(p1.x-2, p1.y-(d.fontSize||13)-2, m.width+4, (d.fontSize||13)+4);
      }}
      break;

    case 'fib':
    case 'fibext': {{
      const ratios = d.tool==='fib' ? FIB_RATIOS_DRAW : FIB_EXT_RATIOS;
      const dy = p2.y - p1.y;
      ctx2d.setLineDash([]);
      ratios.forEach((r, i) => {{
        const y = d.tool==='fib' ? p1.y + dy*r : p2.y - dy*r;
        ctx2d.strokeStyle = FIB_COLORS[i % FIB_COLORS.length];
        ctx2d.lineWidth   = r === 0 || r === 1 ? 1.5 : 1;
        ctx2d.setLineDash(r===0||r===1?[]:[4,2]);
        ctx2d.beginPath();
        ctx2d.moveTo(p1.x, y); ctx2d.lineTo(p2.x, y);
        ctx2d.stroke();
        // Price label
        const price = d.pt1 ? (d.pt1.price - (d.pt1.price - (d.pt2?.price||d.pt1.price))*r) : '';
        ctx2d.font = '9px monospace';
        ctx2d.fillStyle = FIB_COLORS[i % FIB_COLORS.length];
        ctx2d.fillText(`${{r}} — ${{typeof price==='number'?price.toPrecision(5):''}}`, p2.x + 4, y + 3);
      }});
      // Vertical lines
      ctx2d.strokeStyle = (d.color||'#f9e2af')+'60';
      ctx2d.lineWidth = 1; ctx2d.setLineDash([3,3]);
      ctx2d.beginPath(); ctx2d.moveTo(p1.x, p1.y); ctx2d.lineTo(p1.x, p2.y); ctx2d.stroke();
      ctx2d.beginPath(); ctx2d.moveTo(p2.x, p1.y); ctx2d.lineTo(p2.x, p2.y); ctx2d.stroke();
      break;
    }}

    case 'fibcircle': {{
      const radius = Math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2);
      FIB_RATIOS_DRAW.forEach((r,i) => {{
        ctx2d.strokeStyle = FIB_COLORS[i%FIB_COLORS.length];
        ctx2d.lineWidth = 1; ctx2d.setLineDash([]);
        ctx2d.beginPath();
        ctx2d.arc(p1.x, p1.y, radius*r, 0, Math.PI*2);
        ctx2d.stroke();
      }});
      break;
    }}

    case 'measure': {{
      const w = p2.x-p1.x, h = p2.y-p1.y;
      ctx2d.strokeStyle = '#f9e2af'; ctx2d.lineWidth=1; ctx2d.setLineDash([4,2]);
      ctx2d.strokeRect(p1.x,p1.y,w,h);
      ctx2d.setLineDash([]);
      const price1 = d.pt1?.price||canvasYToPrice(p1.y)||0;
      const price2 = d.pt2?.price||canvasYToPrice(p2.y)||0;
      const pct = price1 ? ((price2-price1)/price1*100).toFixed(2) : '?';
      const bars = d.pt1&&d.pt2 ? Math.abs(d.pt2.time - d.pt1.time) : '?';
      const label = `${{pct}}%  ${{bars}}s`;
      ctx2d.font = 'bold 11px monospace';
      ctx2d.fillStyle = '#f9e2af';
      ctx2d.fillText(label, p1.x + w/2 - ctx2d.measureText(label).width/2, Math.min(p1.y,p2.y) - 4);
      break;
    }}
  }}
}}

function extendToEdge(from, to) {{
  const dx = to.x - from.x, dy = to.y - from.y;
  if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) return [to.x, to.y];
  let t = 1e6;
  if (dx > 0) t = Math.min(t, (canvas.width  - from.x) / dx);
  if (dx < 0) t = Math.min(t, (0 - from.x) / dx);
  if (dy > 0) t = Math.min(t, (canvas.height - from.y) / dy);
  if (dy < 0) t = Math.min(t, (0 - from.y) / dy);
  return [from.x + dx*t, from.y + dy*t];
}}

function drawArrowHead(from, to, color) {{
  const angle = Math.atan2(to.y-from.y, to.x-from.x);
  const size  = 9;
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = color;
  ctx2d.beginPath();
  ctx2d.moveTo(to.x, to.y);
  ctx2d.lineTo(to.x - size*Math.cos(angle-Math.PI/7), to.y - size*Math.sin(angle-Math.PI/7));
  ctx2d.lineTo(to.x - size*Math.cos(angle+Math.PI/7), to.y - size*Math.sin(angle+Math.PI/7));
  ctx2d.closePath(); ctx2d.fill();
}}

// ─── HIT TEST ───
function hitTest(pos) {{
  for (let i = drawings.length-1; i >= 0; i--) {{
    const d = drawings[i];
    if (!d.p1) continue;
    if (d.tool === 'text') {{
      ctx2d.font = `${{d.fontSize||13}}px monospace`;
      const w = ctx2d.measureText(d.text||'').width;
      if (pos.x >= d.p1.x-2 && pos.x <= d.p1.x+w+2 &&
          pos.y >= d.p1.y-(d.fontSize||13)-2 && pos.y <= d.p1.y+2) return i;
      continue;
    }}
    if (d.tool === 'hline') {{
      if (Math.abs(pos.y - d.p1.y) < 5) return i;
      continue;
    }}
    if (d.tool === 'vline') {{
      if (Math.abs(pos.x - d.p1.x) < 5) return i;
      continue;
    }}
    if (d.tool === 'rect' || d.tool === 'measure') {{
      const x0=Math.min(d.p1.x,d.p2.x), x1=Math.max(d.p1.x,d.p2.x);
      const y0=Math.min(d.p1.y,d.p2.y), y1=Math.max(d.p1.y,d.p2.y);
      if (pos.x>=x0-3&&pos.x<=x1+3&&pos.y>=y0-3&&pos.y<=y1+3) return i;
      continue;
    }}
    if (distToSegment(pos, d.p1, d.p2||d.p1) < 6) return i;
  }}
  return -1;
}}

function distToSegment(p, a, b) {{
  const dx=b.x-a.x, dy=b.y-a.y;
  if (dx===0&&dy===0) return Math.hypot(p.x-a.x, p.y-a.y);
  const t = Math.max(0, Math.min(1, ((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy)));
  return Math.hypot(p.x-(a.x+t*dx), p.y-(a.y+t*dy));
}}

// ─── HISTORY ───
function saveHistory() {{ history.push(JSON.stringify(drawings)); if (history.length>50) history.shift(); }}
function undoDrawing()  {{ if (history.length) {{ drawings.length=0; JSON.parse(history.pop()).forEach(d=>drawings.push(d)); renderDrawings(); }} }}
function clearAllDrawings() {{ saveHistory(); drawings.length=0; selectedIdx=-1; renderDrawings(); }}

// ─── CONTEXT MENU ───
let ctxTargetIdx = -1;
function onContextMenu(e) {{
  e.preventDefault();
  const pos = getCanvasPos(e);
  const idx = hitTest(pos);
  if (idx >= 0) showContextMenuForDrawing(idx, e.clientX, e.clientY);
  else hideContextMenu();
}}
function showContextMenuForDrawing(idx, cx, cy) {{
  ctxTargetIdx = idx;
  const menu = document.getElementById('ctx-menu');
  menu.style.display = 'block';
  menu.style.left = cx + 'px';
  menu.style.top  = cy + 'px';
}}
function hideContextMenu() {{
  document.getElementById('ctx-menu').style.display = 'none';
  ctxTargetIdx = -1;
}}
document.addEventListener('click', () => hideContextMenu());

function ctxAction(action) {{
  hideContextMenu();
  if (ctxTargetIdx < 0) return;
  const d = drawings[ctxTargetIdx];
  if (action === 'delete') {{ saveHistory(); drawings.splice(ctxTargetIdx,1); selectedIdx=-1; renderDrawings(); }}
  if (action === 'duplicate') {{ saveHistory(); drawings.push(JSON.parse(JSON.stringify(d)));  renderDrawings(); }}
  if (action === 'color')  showColorPopup(ctxTargetIdx);
  if (action === 'width')  {{
    const w = prompt('Толщина линии (1-5):', d.width||1.5);
    if (w) {{ d.width = +w; renderDrawings(); }}
  }}
  if (action === 'style') {{
    const s = prompt('Стиль (solid / dashed / dotted):', d.style||'solid');
    if (s) {{ d.style = s; renderDrawings(); }}
  }}
}}

// ─── COLOR PICKER ───
const PALETTE = [
  '#f9e2af','#f38ba8','#a6e3a1','#89b4fa','#cba6f7','#fab387','#94e2d5',
  '#f5c2e7','#eba0ac','#a6adc8','#585b70','#313244','#ffffff','#ff0000',
  '#00ff00','#0080ff','#ff8800','#ff00ff','#00ffff','#ffff00','#c8a2c8',
];
const cgrid = document.getElementById('color-grid');
let colorTargetIdx = -1;
PALETTE.forEach(c => {{
  const sw = document.createElement('div');
  sw.className = 'color-swatch'; sw.style.background = c;
  sw.onclick = () => {{
    applyColor(c);
    document.getElementById('color-popup').style.display = 'none';
  }};
  cgrid.appendChild(sw);
}});
document.getElementById('color-custom').oninput = (e) => applyColor(e.target.value);

function applyColor(color) {{
  drawColor = color;
  if (colorTargetIdx >= 0 && drawings[colorTargetIdx]) {{
    drawings[colorTargetIdx].color = color;
    renderDrawings();
  }}
}}
function showColorPopup(idx) {{
  colorTargetIdx = idx;
  const popup = document.getElementById('color-popup');
  popup.style.display = 'block';
  popup.style.left = '60px'; popup.style.top = '100px';
}}
document.addEventListener('click', (e) => {{
  const popup = document.getElementById('color-popup');
  if (!popup.contains(e.target) && e.target.id !== 'color-custom') {{
    popup.style.display = 'none';
    colorTargetIdx = -1;
  }}
}});

// ─── TOOLTIP ───
function updateTooltip(e, canvasPos) {{
  if (currentTool !== 'cursor') {{ document.getElementById('tooltip').style.display='none'; return; }}
  const price = canvasYToPrice(canvasPos.y);
  const time  = canvasXToTime(canvasPos.x);
  if (!price || !time) {{ document.getElementById('tooltip').style.display='none'; return; }}
  // Find nearest candle
  const ts = time;
  const candle = CANDLES.find(c => c.time === ts) ||
                 CANDLES.reduce((best,c) => Math.abs(c.time-ts)<Math.abs(best.time-ts)?c:best, CANDLES[0]);
  if (!candle) return;
  const chg = ((candle.close-candle.open)/candle.open*100).toFixed(2);
  const tip = document.getElementById('tooltip');
  tip.innerHTML = `O:<b>${{candle.open.toPrecision(6)}}</b> H:<b>${{candle.high.toPrecision(6)}}</b> L:<b>${{candle.low.toPrecision(6)}}</b> C:<b>${{candle.close.toPrecision(6)}}</b> <span style="color:${{chg>=0?'#a6e3a1':'#f38ba8'}}">${{chg>0?'+':''}}${{chg}}%</span>`;
  tip.style.display = 'block';
  tip.style.left    = (e.clientX + 10) + 'px';
  tip.style.top     = (e.clientY - 30) + 'px';
}}

// ─── SUBSCRIBE TO CHART UPDATES ───
cMain.timeScale().subscribeVisibleTimeRangeChange(() => renderDrawings());
cMain.subscribeCrosshairMove(() => renderDrawings());

setTimeout(() => {{ resizeAll(); fitContent(); renderDrawings(); }}, 200);
setTimeout(() => renderDrawings(), 700);
</script>
</body></html>"""

    components.html(html, height=height + 50, scrolling=False)
