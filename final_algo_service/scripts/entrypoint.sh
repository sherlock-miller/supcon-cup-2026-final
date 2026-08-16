#!/bin/bash
# 容器入口 — 启动预热后运行服务
set -e

echo "=========================================="
echo "  汪汪队 — 中控杯决赛算法服务 (Docker)"
echo "=========================================="
echo "  机械臂:   ${ARM_BASE_URL:-http://127.0.0.1:8087}"
echo "  灵巧手:   ${HAND_BASE_URL:-http://127.0.0.1:8088}"
echo "  服务端口: ${ALGO_PORT:-5000}"
echo "=========================================="

cd /app

# 预热模型（后台），同时启动服务
if [ -f preheat.py ]; then
    echo "开始模型预热（后台运行）..."
    python preheat.py &
    PREHEAT_PID=$!
fi

# 启动 FastAPI 服务
exec python app.py
