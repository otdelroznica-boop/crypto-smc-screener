"""
Telegram-уведомления.
Формат сообщения по вашей спецификации из документа.
"""
import requests


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str):
        self.token   = bot_token
        self.chat_id = chat_id
        self._url    = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._sent   = set()  # дедупликация — не слать одно и то же дважды за сессию

    def send_raw(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            return False
        try:
            r = requests.post(self._url, json={
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": "HTML",
            }, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def send_signal(
        self,
        row,
        exchange:   str = "Binance",
        timeframe:  str = "5m",
        signal_count: int = 1,
    ) -> bool:
        """
        Формат по вашему ТЗ:

        [Биржа] – ТФ – [Инструмент]
        ОИ вырос на X% ($YM)
        Изменение цены: Z%
        CVD: ↑
        Сигналы за сутки: N
        """
        dedup_key = f"{row['Symbol']}_{row['OI_Change']}_{row['Change24h']}"
        if dedup_key in self._sent:
            return False

        oi_str  = f"{row['OI_Change']:+.1f}%  (${row['OI_USD']/1e6:.1f}M)" if row["OI_USD"] else "–"
        stars   = "⭐" * min(int(row["Score"]), 5)

        text = (
            f"🔥 <b>{exchange}</b> – {timeframe} – <b>{row['Symbol']}</b>\n"
            f"ОИ: <b>{oi_str}</b>\n"
            f"Изменение цены: <b>{row['Change24h']:+.2f}%</b>\n"
            f"CVD: {row['CVD']}\n"
            f"Фандинг: {row['Funding']:+.4f}%\n"
            f"Сигналы: {row['Signals']}\n"
            f"Score: {stars} ({row['Score']})\n"
            f"Сигналов за сутки по монете: <b>{signal_count}</b>"
        )

        ok = self.send_raw(text)
        if ok:
            self._sent.add(dedup_key)
        return ok

    def test_connection(self) -> bool:
        return self.send_raw("✅ Trading Screener подключён")
