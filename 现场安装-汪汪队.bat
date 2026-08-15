@echo off
chcp 65001 >nul
title 现场环境安装 - 汪汪队

echo ==========================================
echo   现场环境安装脚本（离线，Win11 工控机）
echo ==========================================
echo.
echo 前置条件:
echo   1. 已安装 Python 3.10 (U盘附带 python-3.10.x-amd64.exe)
echo   2. U盘中有 wheels 文件夹
echo   3. U盘中有 models 文件夹（模型缓存）
echo.
echo 按任意键开始安装...
pause >nul

cd /d "%~dp0"

echo.
echo [1/3] 检查 Python...
python --version 2>nul
if errorlevel 1 (
    echo ❌ 未检测到 Python，请先运行U盘中的 Python 安装包
    pause
    exit /b 1
)

echo.
echo [2/3] 安装依赖（约 5-10 分钟）...
python -m pip install --no-index --find-links=wheels -r requirements.txt
if errorlevel 1 (
    echo ⚠️ 离线安装失败，尝试在线安装...
    python -m pip install -r requirements.txt
)

echo.
echo [3/3] 拷贝模型缓存...
if exist "..\models" (
    xcopy /E /I /Y "..\models" "%USERPROFILE%\.cache\huggingface" >nul
    xcopy /E /I /Y "..\models\.EasyOCR" "%USERPROFILE%\.EasyOCR" >nul
    echo ✅ 模型缓存已拷贝
) else (
    echo ⚠️ 未找到 models 文件夹，模型将在首次运行时下载（需联网）
)

echo.
echo ==========================================
echo   ✅ 环境安装完成！
echo.
echo   下一步:
echo     1. python scripts\hardware_check.py  (硬件自检)
echo     2. python scripts\preheat.py        (模型预热)
echo     3. python app.py                     (启动服务)
echo ==========================================
pause
