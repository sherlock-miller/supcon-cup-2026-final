# 中控杯决赛 — 汪汪队算法服务 Docker 镜像
# ==========================================
# 构建: bash build_image.sh
# 导出: docker save -o 汪汪队决赛镜像.tar wangwang-final:latest
# 现场: bash run_on_site.sh
#
# 参考: 实习项目 boxing_robot_camera Dockerfile 经验
#   - 清华 pip 源加速
#   - CPU torch 走官方 CDN extra-index
#   - 模型预下载进镜像（现场无需联网）

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# 国内 HF 镜像（下载模型用）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_DISABLE_XET=1

# ─── 系统依赖 ───
# libgl1 + libglib2.0-0: OpenCV 运行依赖
# libgomp1: torch 运行依赖
# libusb-1.0-0 + udev: Gemini335 USB 设备访问
# wget: 模型预下载
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    udev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── pip 依赖（清华源） ───
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir \
    torch torchvision \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    transformers \
    ultralytics \
    easyocr \
    opencv-python-headless \
    Pillow \
    numpy \
    python-multipart

# ─── 预下载模型（构建时下载，现场零联网） ───
# 1. CLIP ViT-B/32 (~600MB)
RUN python -c "from transformers import CLIPProcessor, CLIPModel; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    print('CLIP OK')"

# 2. Grounding DINO tiny (~700MB)
RUN python -c "from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection; \
    GroundingDinoForObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
    GroundingDinoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
    print('GroundingDINO OK')"

# 3. EasyOCR 中英文模型 (~100MB) — 下载后放入 /root/.EasyOCR
RUN python -c "import easyocr; \
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, download_enabled=True); \
    print('EasyOCR OK')"

# ─── 项目代码 ───
WORKDIR /app
COPY app.py config.py requirements.txt ./
COPY hardware/ ./hardware/
COPY vision/ ./vision/
COPY tasks/ ./tasks/
COPY utils/ ./utils/

# ─── 预热脚本 ───
COPY scripts/preheat.py /app/preheat.py

# ─── 入口 ───
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["/entrypoint.sh"]
