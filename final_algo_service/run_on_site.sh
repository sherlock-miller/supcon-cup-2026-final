#!/bin/bash
# 现场部署脚本 — Win11 工控机版
# ===============================================
# 前提: 工控机已安装 Docker Desktop (WSL2 backend)
# 用法: bash run_on_site.sh
#
# ⚠️ Windows Docker Desktop 注意事项:
#   - 不支持 --network host → 用 host.docker.internal 访问宿主机服务
#   - USB 直通受限 → 相机服务跑在宿主机 (camera_server.py)

set -e

echo "=========================================="
echo "  汪汪队决赛算法服务 — 现场部署 (Win11)"
echo "=========================================="

cd "$(dirname "$0")"

# 1. 检查 Docker
if ! docker --version &> /dev/null; then
    echo "❌ 未检测到 Docker。"
    echo "   方案1: U盘携带 Docker Desktop 安装包，现场安装"
    echo "   方案2: 放弃 Docker，用 现场安装-汪汪队.bat (wheels 离线方案)"
    exit 1
fi
echo "✅ Docker: $(docker --version)"

# 2. 加载镜像
if ! docker image inspect wangwang-final:latest &> /dev/null; then
    echo "加载镜像..."
    docker load -i "汪汪队决赛镜像.tar"
fi
echo "✅ 镜像就绪"

# 3. 检查宿主机服务（机械臂 HTTP API 是官方预装的）
echo ""
echo "检查宿主机硬件服务..."
if curl -s --max-time 3 http://127.0.0.1:8087/api/status > /dev/null 2>&1; then
    echo "✅ 机械臂服务在线 (127.0.0.1:8087)"
else
    echo "⚠️  机械臂服务未响应 — 请确认官方 HTTP API 已启动"
fi

# 4. 启动相机服务（宿主机 Windows 原生，Orbbec SDK）
echo ""
echo "启动相机服务..."
if [ -f "../相机服务/start_camera_server.bat" ]; then
    echo "  请手动运行: 相机服务/start_camera_server.bat"
    echo "  （Windows 原生进程，Docker 内无法直接访问 USB）"
fi

# 5. 启动容器
echo ""
echo "启动算法服务容器..."
docker rm -f wangwang_final 2>/dev/null || true
docker run -d \
    --name wangwang_final \
    -p 5000:5000 \
    --add-host=host.docker.internal:host-gateway \
    -e ALGO_PORT=5000 \
    -e ARM_BASE_URL=http://host.docker.internal:8087 \
    -e HAND_BASE_URL=http://host.docker.internal:8088 \
    -e CAMERA_SERVER_URL=http://host.docker.internal:5002 \
    --restart unless-stopped \
    wangwang-final:latest

sleep 5

# 6. 验证
if curl -s --max-time 5 http://127.0.0.1:5000/api/health | grep -q success; then
    echo ""
    echo "=========================================="
    echo "  ✅ 部署完成！"
    echo "  竞赛操作软件 Base URL: http://127.0.0.1:5000"
    echo "  日志: docker logs -f wangwang_final"
    echo "=========================================="
else
    echo "⚠️  服务可能仍在初始化，查看日志: docker logs -f wangwang_final"
fi
