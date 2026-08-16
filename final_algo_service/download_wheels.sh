#!/bin/bash
# 在家执行：下载全部依赖 wheel 包（Windows 版本）
# ===============================================
# 现场工控机 Win11 无外网/网络波动时的最稳方案。
# 产物: wheels/ 文件夹（约 2-3GB），随 U盘携带。
#
# 用法: bash download_wheels.sh

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  下载 Windows 依赖 wheel 包"
echo "=========================================="

mkdir -p wheels

# 用国内源下载（清华源 wheel 最全）
python -m pip download \
    --platform win_amd64 \
    --python-version 310 \
    --only-binary=:all: \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -d wheels \
    fastapi uvicorn pydantic requests \
    transformers ultralytics easyocr \
    opencv-python Pillow numpy python-multipart \
    pyorbbecsdk 2>/dev/null || true

# torch 单独下载（CPU 版，需要指定 PyTorch 官方源）
python -m pip download \
    --platform win_amd64 \
    --python-version 310 \
    --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cpu \
    -d wheels \
    torch torchvision 2>/dev/null || true

# 统计
echo ""
echo "=========================================="
echo "  ✅ wheel 下载完成"
echo "  文件数: $(ls wheels | wc -l)"
echo "  总大小: $(du -sh wheels | cut -f1)"
echo ""
echo "  现场安装命令:"
echo "    pip install --no-index --find-links=wheels -r requirements.txt"
echo "=========================================="
