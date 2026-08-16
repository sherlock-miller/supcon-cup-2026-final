@echo off
chcp 65001 >nul
title 汪汪队相机服务

echo ==========================================
echo   汪汪队相机服务 (宿主机, 端口 5002)
echo ==========================================
echo.
echo 前置: pip install fastapi uvicorn opencv-python pyorbbecsdk
echo.

cd /d "%~dp0"
python camera_server.py

pause
