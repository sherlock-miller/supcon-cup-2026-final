"""任务1 策略二: 三灯分类模型推理（红/白/绿灯亮）
================================================
训练: scripts/train_light_classifier.py → weights/light_classifier.pth
推理: MobileNetV3-Small, CPU 约 20ms/张, 无需 GPU。

用法:
  from vision.light_classifier import LightClassifier
  clf = LightClassifier()            # 延迟加载权重
  label, conf = clf.predict(image)   # label ∈ green/red/white
"""
import logging
import os
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "light_classifier.pth")

# 类别 → 灯号（与 config SWITCH_PANEL lights 一致）
CLASS_TO_LIGHT = {"green": "light_3", "red": "light_1", "white": "light_2"}

_instance = None
_lock = threading.Lock()


class LightClassifier:
    """三灯分类器（懒加载, 线程安全单例）"""

    def __init__(self):
        self._model = None
        self._class_names = None
        self._load_lock = threading.Lock()
        self._load_error = None
        self._load_error_ts = 0.0  # B7 修复: 失败可重试

    def _load(self):
        # B7 修复: 加载失败不永久缓存——60 秒后允许重试
        if self._model is not None:
            return
        if self._load_error and time.time() - self._load_error_ts < 60:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error and time.time() - self._load_error_ts < 60:
                return
            try:
                import torch
                from torchvision import transforms
                from torchvision.models import mobilenet_v3_small

                if not os.path.exists(WEIGHTS_PATH):
                    self._load_error = f"模型权重不存在: {WEIGHTS_PATH}"
                    self._load_error_ts = time.time()
                    return
                ckpt = torch.load(WEIGHTS_PATH, map_location="cpu",
                                  weights_only=False)
                model = mobilenet_v3_small()
                n_cls = len(ckpt["class_names"])
                import torch.nn as nn
                model.classifier[3] = nn.Linear(
                    model.classifier[3].in_features, n_cls)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._model = model
                self._class_names = ckpt["class_names"]
                self._load_error = None
                self._transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
                logger.info(
                    f"灯分类模型已加载 (val_acc={ckpt.get('val_acc', '?')})")
            except Exception as e:
                self._load_error = f"模型加载失败: {e}"
                self._load_error_ts = time.time()
                logger.error(self._load_error)

    def predict(self, image) -> Optional[Tuple[str, float]]:
        """返回 (label, confidence)，label ∈ green/red/white；失败 None"""
        self._load()
        if self._model is None:
            return None
        try:
            import torch
            x = self._transform(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(x)
                prob = torch.softmax(logits, dim=1)[0]
            idx = int(prob.argmax())
            return self._class_names[idx], round(float(prob[idx]), 3)
        except Exception as e:
            logger.warning(f"灯分类推理失败: {e}")
            return None

    def predict_light_id(self, image) -> Optional[Tuple[str, str, float]]:
        """返回 (light_id, color_label, confidence)，失败 None"""
        r = self.predict(image)
        if r is None:
            return None
        label, conf = r
        return CLASS_TO_LIGHT.get(label), label, conf


def get_light_classifier() -> LightClassifier:
    """线程安全单例"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LightClassifier()
    return _instance
