@echo off
echo ============================================
echo   Trading Screener — запуск
echo ============================================

cd /d "%~dp0"

:: Проверить Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Python не найден. Установите Python 3.10+
    pause
    exit /b 1
)

:: Установить зависимости если нужно
echo Проверяем зависимости...
pip install -q -r requirements.txt

:: Запустить дашборд
echo.
echo Открываю браузер на http://localhost:8501
echo Для остановки нажмите Ctrl+C
echo.
python -m streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false --server.headless true

pause
