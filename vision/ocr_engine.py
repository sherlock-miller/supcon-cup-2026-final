"""
OCR 文字识别 — EasyOCR (PyTorch)
工业铭牌/仪表中文文字识别 + 后处理纠错
"""
import re
import logging
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)


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


class OCREngine:
    """EasyOCR 工业文字识别引擎"""

    def __init__(self):
        self.available = False
        self.reader = None
        try:
            import easyocr
            self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
            self.available = True
            logger.info("EasyOCR 引擎就绪")
        except ImportError:
            logger.warning("EasyOCR 未安装")
        except Exception as e:
            logger.error(f"EasyOCR 初始化异常: {e}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if not self.available or self.reader is None:
            return {"text": ""}

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
