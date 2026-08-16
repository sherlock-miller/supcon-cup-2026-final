@echo off
chcp 65001 >nul
title 汪汪队中控杯决赛算法服务

echo ==========================================
echo   汪汪队 — 中控杯决赛算法服务 v2.0
echo ==========================================
echo.
echo 接口列表:
echo   GET  http://127.0.0.1:5000/api/health
echo   POST http://127.0.0.1:5000/api/task1/execute
echo   POST http://127.0.0.1:5000/api/task2/execute
echo   POST http://127.0.0.1:5000/api/task3/execute
echo.
echo API 文档: http://127.0.0.1:5000/docs
echo.
echo 启动中...

cd /d "%~dp0"
python app.py

pause
