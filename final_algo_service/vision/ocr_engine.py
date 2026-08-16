"""
OCR 文字识别 — EasyOCR (PyTorch)（决赛优化版）
==============================================
保留工业铭牌模式（predict），新增决赛任务2的单数字识别模式：

  predict_single_digit — 只认 1-4：
    1. EasyOCR 识别 + 数字混淆修复（I/l→1, O/o→0）
    2. 模板匹配兜底（PIL 渲染数字模板 + 归一化互相关，纯 numpy 无 ML 依赖）
    3. 极简几何规则（数字 1 天然最窄）作为最低优先级兜底

数字区域预处理：灰度化 + CLAHE 对比度增强 + 大津二值化前景提取。

注意：numpy 延迟导入（同 classifier.py），模板匹配函数不依赖 EasyOCR，
EasyOCR 缺失/模型下载失败时单数字识别仍可用。
"""
import re
import logging
from functools import lru_cache
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 单数字识别配置
# ============================================================
VALID_DIGITS = (1, 2, 3, 4)
# 模板匹配画布尺寸
_TEMPLATE_SIZE = 96
# 模板匹配接受阈值（|NCC| 归一化互相关，极性鲁棒）
_TEMPLATE_MIN_SCORE = 0.45
# 数字渲染字体（测试图/现场印刷字体为微软雅黑，优先匹配）
_DIGIT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
# 前景像素占比范围（crop 内数字占比过小/过大视为无效区域）
_FG_MIN_RATIO = 0.002
_FG_MAX_RATIO = 0.80
# 数字 1 几何先验：前景外接框宽高比阈值
# 实测测试集：1 的 w/h≈0.35，2/3/4 的 w/h≥0.56，间隙明显
_DIGIT1_ASPECT_MAX = 0.50
# 数字 1 前景占比上限（笔画少，实测≈0.08；2/3/4 明显更高）
_DIGIT1_MAX_FG_RATIO = 0.25
# 数字 1 几何兜底：前景外接框高度占裁剪高度的最小比例
# （低于此值视为噪点/笔画断裂，不判为数字 1）
_DIGIT1_MIN_HEIGHT_RATIO = 0.30


def _clean_ocr_text(text: str) -> str:
    """
    OCR后处理：修复工业铭牌常见识别错误
    - 清理前导/尾随垃圾字符
    - 字母全转大写（工业铭牌规范）
    - 修复数字/字母混淆（8↔o, 0↔O等）
    """
    # 1. 去掉前导垃圾符号
    text = re.sub(r'^[;:\'\"`\s\-\.\,]+', '', text)
    # 2. 去掉尾随垃圾
    text = re.sub(r'[;:\'\"`\s\-\.\,]+$', '', text)
    # 3. 去掉多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return text

    # 4. 分隔单词和数字区域，分别处理
    # 将字符串分成token：中文、英文单词、数字串
    tokens = re.split(r'(\s+)', text)
    fixed = []
    for token in tokens:
        if not token.strip():
            fixed.append(token)
            continue
        # 移除token前后的垃圾但保留中间
        token = token.strip(';:\'\"`')

        # 检测是否为数字/单位混合（如 "8oow", "1.6MPa"）
        has_digit = bool(re.search(r'\d', token))
        has_letter = bool(re.search(r'[a-zA-Z]', token))

        if has_digit and has_letter:
            # 数字+字母混合：大写字母，修复数字区的o→0
            result = []
            # 找到数字区域和字母区域
            for i, ch in enumerate(token):
                if ch == 'o' or ch == 'O':
                    # o在数字旁边或单独在数字区→0
                    prev_digit = i > 0 and token[i-1].isdigit()
                    next_digit = i < len(token)-1 and token[i+1].isdigit()
                    if prev_digit or next_digit:
                        result.append('0')
                    else:
                        result.append('O')
                elif ch == 'l' or ch == 'I':
                    prev_digit = i > 0 and token[i-1].isdigit()
                    next_digit = i < len(token)-1 and token[i+1].isdigit()
                    if prev_digit or next_digit:
                        result.append('1')
                    else:
                        result.append('I')
                elif ch.isalpha():
                    result.append(ch.upper())
                else:
                    result.append(ch)
            fixed.append(''.join(result))
        elif has_letter:
            # 纯字母：全大写
            fixed.append(token.upper())
        else:
            # 纯数字/中文/符号：不变
            fixed.append(token)

    result = ''.join(fixed)
    return result.strip()


def _otsu_threshold(gray: "np.ndarray") -> int:
    """纯 numpy 大津二值化阈值（256 级直方图）"""
    import numpy as np
    hist, _ = np.histogram(gray.ravel(), 256, [0, 256])
    total = gray.size
    sum_all = float((np.arange(256) * hist).sum())
    sum_b = 0.0
    w_b = 0
    best = 0.0
    best_t = 0
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * float(hist[t])
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best:
            best = var_between
            best_t = t
    return best_t


def _load_digit_font(size: int):
    """加载数字模板字体（微软雅黑优先）"""
    from PIL import ImageFont
    for path in _DIGIT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _render_digit_templates():
    """PIL 渲染数字 1-4 模板（多字号，黑字白底，画布 96x96）

    返回 {digit: [(画布数组, 字宽, 字高), ...]}——记录每个模板中字的
    实际尺寸，匹配时把候选区域缩放到相同字尺寸，保证尺度对齐。
    """
    import numpy as np
    from PIL import Image, ImageDraw
    templates: Dict[int, List[tuple]] = {}
    for digit in VALID_DIGITS:
        imgs = []
        for size in (36, 48, 60, 72, 84):
            canvas = Image.new("L", (_TEMPLATE_SIZE, _TEMPLATE_SIZE), 255)
            draw = ImageDraw.Draw(canvas)
            font = _load_digit_font(size)
            text = str(digit)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text(
                ((_TEMPLATE_SIZE - tw) / 2 - bbox[0],
                 (_TEMPLATE_SIZE - th) / 2 - bbox[1]),
                text, fill=0, font=font,
            )
            imgs.append((np.asarray(canvas), tw, th))
        templates[digit] = imgs
    return templates


@lru_cache(maxsize=1)
def _get_digit_templates_cached():
    """模板渲染缓存：返回 ((digit, ((bytes, tw, th), ...)), ...)，
    bytes 可用 np.frombuffer 还原为 96x96 数组（ndarray 不可哈希，需转 bytes）"""
    templates = _render_digit_templates()
    return tuple(
        (digit, tuple((img.tobytes(), tw, th) for img, tw, th in imgs))
        for digit, imgs in sorted(templates.items())
    )


def _ncc(a: "np.ndarray", b: "np.ndarray") -> float:
    """归一化互相关（减均值除标准差）"""
    import numpy as np
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return 0.0
    return float((a * b).sum() / denom)


def template_match_single_digit(
    image,
    valid_digits: Tuple[int, ...] = VALID_DIGITS,
    min_score: float = _TEMPLATE_MIN_SCORE,
) -> Optional[Dict[str, Any]]:
    """
    单数字模板匹配（纯 numpy/PIL，无 ML 依赖）

    流程：
      灰度 → 大津二值化 → 提取前景（少数像素类=数字笔画）
      → 前景外接框 resize 到模板画布 → 与多字号模板算 |NCC|
      （取绝对值 → 对"深字浅底/浅字深底"两种印刷极性都鲁棒）

    数字 1 强化（决赛实测漏检根因）：
      - 几何先验：前景外接框 w/h < 0.5 时只与数字 1 的模板比较
        （数字 1 天然最窄，避免与 2/3/4 模板误匹配）
      - 几何兜底：NCC 匹配失败时，窄长前景 + 笔画占比合理 → 数字 1

    Args:
        image: PIL Image，裁剪后的数字区域
        valid_digits: 允许的数字集合
        min_score: 最低接受分数（|NCC|）

    Returns:
        {"digit": int, "confidence": float, "method": "template"/"geometry",
         "aspect": float} 或 None
        aspect 为前景外接框宽高比（供上层交叉验证仲裁用）
    """
    import numpy as np
    from PIL import Image

    if image is None:
        return None

    gray = np.asarray(image.convert("L"))
    crop_h, crop_w = gray.shape[:2]

    # 大津二值化 → 少数像素类 = 数字笔画（前景）
    thr = _otsu_threshold(gray)
    bright = gray > thr
    total = gray.size
    if bright.sum() > total - bright.sum():
        fg = ~bright  # 亮类为背景 → 字是暗像素
        fg_is_bright = False
    else:
        fg = bright   # 亮类为字 → 字是亮像素
        fg_is_bright = True

    ratio = float(fg.sum()) / total
    if ratio < _FG_MIN_RATIO or ratio > _FG_MAX_RATIO:
        return None  # 区域内没有合理大小的数字

    # 前景外接框 → 按模板字尺寸等比缩放 → 居中放入画布
    ys, xs = np.nonzero(fg)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    crop = gray[y0:y1 + 1, x0:x1 + 1]
    # 极性归一化：浅字深底印刷（前景为亮像素）→ 反相为黑字白底，
    # 与模板极性一致（|NCC| 对整体反相鲁棒，但对"仅字反相"不鲁棒）
    if fg_is_bright:
        crop = 255 - crop
    h, w = crop.shape
    aspect = float(w) / float(h) if h > 0 else 1.0
    if h < 4 or w < 4:
        return None

    # 数字 1 几何先验：窄字形只与数字 1 的模板比较（减少误匹配）
    candidate_digits = valid_digits
    if 1 in valid_digits and aspect < _DIGIT1_ASPECT_MAX:
        candidate_digits = (1,)

    # 与模板逐字号匹配：候选缩放到与模板字相同尺寸，|NCC| 最大者胜出
    best_digit: Optional[int] = None
    best_score = min_score
    for digit, tmpl_infos in _get_digit_templates_cached():
        if digit not in candidate_digits:
            continue
        for img_bytes, tw, th in tmpl_infos:
            # 候选区域等比缩放到模板字尺寸（保持长宽比，向小取整）
            scale = min(tw / max(w, 1), th / max(h, 1))
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            resized = np.asarray(
                Image.fromarray(crop).resize((nw, nh), Image.BILINEAR)
            )
            canvas = np.full(
                (_TEMPLATE_SIZE, _TEMPLATE_SIZE), 255, dtype=np.uint8
            )
            oy, ox = (_TEMPLATE_SIZE - nh) // 2, (_TEMPLATE_SIZE - nw) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized

            tmpl = np.frombuffer(img_bytes, dtype=np.uint8).reshape(
                _TEMPLATE_SIZE, _TEMPLATE_SIZE
            )
            s = abs(_ncc(canvas, tmpl))
            if s > best_score:
                best_digit, best_score = digit, s

    if best_digit is not None:
        return {
            "digit": best_digit,
            "confidence": round(float(best_score), 3),
            "method": "template",
            "aspect": round(aspect, 3),
        }

    # 几何规则兜底：窄长前景 = 数字 1（需笔画完整，排除噪点竖线）
    if (
        1 in valid_digits
        and aspect < _DIGIT1_ASPECT_MAX
        and ratio <= _DIGIT1_MAX_FG_RATIO
        and h >= _DIGIT1_MIN_HEIGHT_RATIO * crop_h
    ):
        return {
            "digit": 1,
            "confidence": 0.60,
            "method": "geometry",
            "aspect": round(aspect, 3),
        }
    return None


def preprocess_digit(image):
    """
    数字区域预处理：灰度化 + CLAHE 对比度增强（EasyOCR 输入用）
    cv2 缺失时返回原始灰度图
    """
    import numpy as np
    from PIL import Image

    gray = image.convert("L")
    try:
        import cv2
        arr = np.asarray(gray)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(arr)
        return Image.fromarray(enhanced).convert("RGB")
    except ImportError:
        return gray.convert("RGB")


class OCREngine:
    """EasyOCR 识别引擎（铭牌模式 + 单数字模式）"""

    def __init__(self):
        self.available = False
        self.reader = None
        try:
            # 设置短超时：无网络时模型下载快速失败，避免挂起整个服务
            import socket
            socket.setdefaulttimeout(15)
            import easyocr
            self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
            self.available = True
            logger.info("EasyOCR 引擎就绪")
        except ImportError:
            logger.warning("EasyOCR 未安装（单数字识别将走模板匹配兜底）")
        except Exception as e:
            logger.error(f"EasyOCR 初始化异常: {e}")

    # ============================================================
    # 铭牌模式（工业场景，保留原有行为）
    # ============================================================
    def predict(self, image) -> Dict[str, Any]:
        if not self.available or self.reader is None:
            return {"text": ""}

        import numpy as np
        arr = np.array(image.convert("RGB"))

        try:
            results = self.reader.readtext(arr)
        except Exception as e:
            logger.error(f"OCR 推理异常: {e}")
            return {"text": ""}

        # 收集每行文本 + 内联清理
        lines = []
        for _, text, _ in results:
            if not text or not text.strip():
                continue
            t = text.strip()
            # 只去前导垃圾（保留冒号，它是标签的一部分）
            t = t.lstrip(';\"\'` ')
            # 去尾随垃圾（保留冒号和字母数字）
            t = t.rstrip(';\"\'` ')
            # 去掉冒号后的空格（EasyOCR: "型号: KF"→"型号:KF"）
            t = t.replace(': ', ':')
            # 全转大写（工业铭牌规范）
            t = t.upper()
            # 修复数字/字母混淆：O在数字旁→0
            chars = list(t)
            for i, ch in enumerate(chars):
                if ch == 'O':
                    prev_digit = i > 0 and chars[i-1].isdigit()
                    next_digit = i < len(chars)-1 and chars[i+1].isdigit()
                    if prev_digit or next_digit:
                        chars[i] = '0'
            t = ''.join(chars)
            if t:
                lines.append(t)

        full_text = " ".join(lines)
        # 合并后再清理一次冒号空格（跨行合并引入的）
        full_text = full_text.replace(': ', ':')
        return {"text": full_text}

    # ============================================================
    # 单数字模式（任务2：只认 1-4）
    # ============================================================
    # 孤立的非数字单字符 → 数字 1 映射
    # （EasyOCR 对细长的"1"常见误识别：I/l/L/|/!）
    _DIGIT1_OCR_ALIASES = ("I", "L", "|", "!")

    def predict_single_digit(
        self,
        image,
        valid_digits: Tuple[int, ...] = VALID_DIGITS,
    ) -> Optional[Dict[str, Any]]:
        """
        识别裁剪区域内的单个数字（1-4）。

        策略（双引擎交叉验证，修复数字 1 漏检）：
          1. EasyOCR 识别 → 混淆修复 → 提取有效数字（暂存不立即返回）
          2. 模板匹配（纯 numpy，总是执行，含数字 1 几何先验）
          3. 融合：
             - 两引擎一致 → 高置信返回
             - 不一致 → 数字 1 几何仲裁（窄前景必为 1）
               → 模板匹配高置信优先 → 否则信任 EasyOCR

        Returns:
            {"digit": int, "confidence": float,
             "method": "easyocr"/"template"/"easyocr+template"/"geometry"}
            或 None（区域中不存在有效数字）
        """
        ocr_result: Optional[Dict[str, Any]] = None

        # 路径1：EasyOCR
        if self.available and self.reader is not None:
            import numpy as np
            try:
                processed = preprocess_digit(image)
                arr = np.asarray(processed.convert("RGB"))
                results = self.reader.readtext(arr)
                # 按置信度降序，逐条提取数字
                results = sorted(results, key=lambda r: r[2] if len(r) > 2 else 0.0, reverse=True)
                for row in results:
                    text = row[1]
                    conf = float(row[2]) if len(row) > 2 else 0.0
                    if not text or not text.strip():
                        continue
                    # 混淆修复：数字旁的 I/l → 1
                    cleaned = _clean_ocr_text(text.strip())
                    cleaned = cleaned.upper()
                    # 只保留修复后的数字串
                    digit_str = re.sub(r'[^0-9]', '', cleaned)
                    if len(digit_str) == 1:
                        digit = int(digit_str)
                        if digit in valid_digits:
                            ocr_result = {
                                "digit": digit,
                                "confidence": round(max(0.0, min(1.0, conf)), 3),
                                "method": "easyocr",
                            }
                            break
                    # 孤立单字符 I/l/L/|/! → 数字 1（EasyOCR 对细长"1"的常见误识别）
                    if (
                        len(cleaned) == 1
                        and cleaned in self._DIGIT1_OCR_ALIASES
                        and 1 in valid_digits
                    ):
                        ocr_result = {
                            "digit": 1,
                            "confidence": round(min(0.6, max(0.0, conf)), 3),
                            "method": "easyocr",
                        }
                        break
            except Exception as e:
                logger.warning(f"EasyOCR 单数字识别异常: {e}")

        # 路径2：模板匹配（交叉验证，数字1几何先验在内）
        tmpl_result = template_match_single_digit(
            image, valid_digits=valid_digits
        )

        # ---- 融合 ----
        if ocr_result is None:
            return tmpl_result
        if tmpl_result is None:
            return ocr_result

        if ocr_result["digit"] == tmpl_result["digit"]:
            # 双引擎一致 → 高置信
            return {
                "digit": ocr_result["digit"],
                "confidence": round(
                    max(ocr_result["confidence"], tmpl_result["confidence"]), 3
                ),
                "method": "easyocr+template",
            }

        # 不一致仲裁
        # 1) 前景窄字形（w/h<0.5）→ 数字必为 1（2/3/4 实测 w/h≥0.56）
        if (
            tmpl_result.get("aspect", 1.0) < _DIGIT1_ASPECT_MAX
            and 1 in valid_digits
        ):
            return {
                "digit": 1,
                "confidence": 0.85,
                "method": "geometry",
            }
        # 2) 模板匹配高置信（实测测试集零误判）→ 采用模板匹配
        if tmpl_result["confidence"] >= 0.75:
            return tmpl_result
        # 3) 模板匹配置信度不足 → 信任 EasyOCR
        return ocr_result

    def normalize_text(self, text: str, rules: Optional[Dict[str, Any]]) -> str:
        """平台侧规范化：trim_space + case_insensitive 转小写比对"""
        if not rules:
            return text
        result = text
        if rules.get("trim_space", False):
            result = result.strip()
        if rules.get("case_insensitive", False):
            result = result.lower()
        result = re.sub(r"\s+", " ", result)
        return result


_ocr_engine: Optional[OCREngine] = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine
