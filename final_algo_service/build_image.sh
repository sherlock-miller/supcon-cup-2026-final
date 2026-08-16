#!/bin/bash
# 构建 + 导出 Docker 镜像（在家/有网环境执行一次）
# ===============================================
# 用法: bash build_image.sh
# 产物: 汪汪队决赛镜像.tar (U盘拷贝到现场)

set -e

echo "=========================================="
echo "  构建汪汪队决赛 Docker 镜像"
echo "=========================================="

cd "$(dirname "$0")"

echo "[1/3] 构建镜像（首次约 20-30 分钟，含模型下载）..."
docker build -t wangwang-final:latest .

echo "[2/3] 验证镜像..."
docker run --rm wangwang-final:latest python -c "
from vision.classifier import CLIPClassifier
from vision.detector import GroundingDinoDetector
from vision.ocr_engine import OCREngine
import easyocr
print('✅ 所有模块导入成功')
"

echo "[3/3] 导出镜像为 tar 文件..."
docker save -o "汪汪队决赛镜像.tar" wangwang-final:latest

echo ""
echo "=========================================="
echo "  ✅ 构建完成！"
echo "  镜像文件: 汪汪队决赛镜像.tar ($(du -h 汪汪队决赛镜像.tar | cut -f1))"
echo ""
echo "  现场部署步骤:"
echo "    1. U盘拷贝 汪汪队决赛镜像.tar + 整个项目文件夹"
echo "    2. 现场工控机: bash run_on_site.sh"
echo "=========================================="
