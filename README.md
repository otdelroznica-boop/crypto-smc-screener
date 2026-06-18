# 📊 Crypto Trading Screener

Персональный скринер криптовалютных фьючерсов на основе Smart Money Concepts (SMC).

## Возможности

- **Рыночный обзор** — топ монет по объёму, OI, CVD, фандинг
- **Паттерн-сканер** — 30+ паттернов: Double Bottom/Top, H&S, гармоники, Эллиотт, ABC Zigzag
- **SMC сканер** — Order Block, FVG, Breaker Block, Three Drives, Three Tap, Range Deviation
- **Волновой анализ** — Elliott Wave с Фибоначчи уровнями
- **VA+SMC панель** — Swing H/L, RSI дивергенция, киллзоны, named liquidity (PDH/PDL/EQH/EQL)
- **Интерактивные графики** — TradingView Lightweight Charts, Log шкала, скриншот, полный экран

## Данные

Binance Futures публичный API (без ключей).

## Запуск

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
