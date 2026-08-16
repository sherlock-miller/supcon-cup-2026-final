"""
图片加载与预处理工具
- URL/Base64 加载
- 低光照增强（classify L3 夜间街景）
- 抗反光/去污渍（OCR L3 工业铭牌）
"""
import io
import base64
import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def load_image(image_data: dict) -> Image.Image:
    """加载图片，支持 url / base64 格式"""
    fmt = image_data.get("format", "url")
    data = image_data.get("data", "")

    if fmt == "url":
        return _load_from_url(data)
    elif fmt == "base64":
        return _load_from_base64(data)
    else:
        raise ValueError(f"不支持的图片格式: {fmt}")


def _load_from_url(url: str) -> Image.Image:
    from urllib.request import urlopen, Request
    from urllib.error import URLError
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            img_bytes = resp.read()
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except URLError as e:
        logger.error(f"下载图片失败: {url} — {e}")
        raise


def _load_from_base64(b64_str: str) -> Image.Image:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ================================================================
# 预处理管线
# ================================================================

def preprocess_for_classify(image: Image.Image) -> Image.Image:
    """
    分类预处理: 增强低光照场景（如夜间街景 L3）
    - 自动亮度检测 + 自适应增强
    """
    gray = image.convert("L")
    avg_brightness = np.array(gray).mean()
    if avg_brightness < 80:  # 低光照
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        logger.info(f"低光照增强: 亮度 {avg_brightness:.0f}→{np.array(image.convert('L')).mean():.0f}")
    return image


def preprocess_for_detect(image: Image.Image) -> Image.Image:
    """检测预处理: 保持原始信息，仅做轻量对比度增强"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.1)


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    OCR 预处理: 轻量增强对比度，不过度处理
    EasyOCR 自带预处理，我们只需做轻量增强
    """
    try:
        import cv2
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)
    except ImportError:
        return image


def detect_lamp_regions(image: Image.Image, min_brightness: int = 190) -> list:
    """
    台灯辅助检测: 检测图片中发光的灯区域
    只在室内暗光场景启用（室外不需要台灯检测）
    返回 [(cx, cy, score), ...]
    """
    try:
        import cv2
        arr = np.array(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        avg_brightness = gray.mean()
        # 只在室内暗光场景启用（avg<100），户外白天不检测台灯
        if avg_brightness > 120:
            return []

        # 自适应阈值找高亮区域
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, -10
        )

        # 形态学闭运算连接断裂区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lamps = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 过滤太小区域
                continue
            # 灯通常是圆形或椭圆形
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 1
            if 0.3 < aspect_ratio < 3.0:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    # 检查该区域的平均亮度
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]
                    if mean_brightness > min_brightness:
                        score = min(0.9, mean_brightness / 255 * 0.95)
                        lamps.append({"label": "台灯", "cx": float(cx), "cy": float(cy), "score": round(float(score), 3)})

        # 去重：合并距离过近的检测
        lamps = _deduplicate_lamp_detections(lamps)
        return lamps[:5]  # 最多5个台灯
    except ImportError:
        return []
    except Exception as e:
        logger.warning(f"台灯检测异常: {e}")
        return []


def _deduplicate_lamp_detections(lamps: list, min_dist: float = 50) -> list:
    """合并距离过近的台灯检测"""
    if len(lamps) <= 1:
        return lamps
    kept = []
    used = set()
    for i, a in enumerate(lamps):
        if i in used:
            continue
        for j, b in enumerate(lamps):
            if j <= i or j in used:
                continue
            dist = ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5
            if dist < min_dist:
                used.add(j)
        kept.append(a)
    return kept
