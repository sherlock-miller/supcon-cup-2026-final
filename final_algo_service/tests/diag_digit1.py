#!/usr/bin/env python3
"""
诊断脚本：验证数字1漏检修复（纯 numpy/PIL，无需 ML 依赖）
===============================================================
1. 重放生成脚本 random 消耗得到 数字→槽位 真值映射
2. 纯模板匹配：32 个数字应全对
3. 模拟 EasyOCR 误识别场景（mock reader），验证
   predict_single_digit 双引擎交叉验证融合逻辑
"""
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")


def replay_cube_maps(n=8):
    """精确重放生成脚本 random 消耗 → 每张图的 数字→(sx,sy) 映射"""
    random.seed(42)
    for _ in range(6):          # gen_switch_panel(None, n=6)
        random.randint(0, 2)
        random.uniform(-20, 20)
    for _ in range(6):          # gen_switch_panel(idx, n=2) × 3
        random.uniform(-20, 20)
    slots_base = [(120, 200), (260, 200), (400, 200), (520, 200)]
    maps = []
    for _ in range(n):          # gen_number_cubes(n=8)
        slots = list(slots_base)
        random.shuffle(slots)
        random.uniform(-20, 20)
        maps.append({1: slots[0], 2: slots[1], 3: slots[2], 4: slots[3]})
    return maps


def crop_slot(arr, sx, sy, inset=4, half=50):
    x1, y1 = sx - half + inset, sy - half + inset
    x2, y2 = sx + half - inset, sy + half - inset
    return Image.fromarray(arr[y1:y2, x1:x2])


class MockReader:
    """模拟 EasyOCR reader：readtext 返回 [(bbox, text, conf), ...]"""

    def __init__(self, behavior):
        self.behavior = behavior  # callable(crop_arr) -> list of rows

    def readtext(self, arr):
        return self.behavior(arr)


def make_mock_engine(behavior):
    """构造 OCREngine 实例并注入 mock reader（不触发真实 EasyOCR 初始化）"""
    import vision.ocr_engine as oe
    eng = oe.OCREngine.__new__(oe.OCREngine)
    eng.available = True
    eng.reader = MockReader(behavior)
    return eng


def test_pure_template():
    """纯模板匹配（含数字1几何先验）应 32/32 全对"""
    from vision.ocr_engine import template_match_single_digit
    maps = replay_cube_maps()
    files = sorted(f for f in os.listdir(IMG_DIR) if f.startswith("cubes_v"))
    total = fail = 0
    for v, fname in enumerate(files):
        arr = np.asarray(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))
        for digit in (1, 2, 3, 4):
            r = template_match_single_digit(crop_slot(arr, *maps[v][digit]))
            pred = r["digit"] if r else None
            total += 1
            if pred != digit:
                fail += 1
                print(f"  ❌ {fname} {digit}→{pred}")
    print(f"[1] 纯模板匹配: {total - fail}/{total} 正确")
    return fail == 0


def test_fusion_scenarios():
    """模拟 EasyOCR 误识别场景，验证交叉验证融合"""
    from vision.ocr_engine import VALID_DIGITS
    maps = replay_cube_maps()
    files = sorted(f for f in os.listdir(IMG_DIR) if f.startswith("cubes_v"))

    def run(engine, crop):
        return engine.predict_single_digit(crop, valid_digits=VALID_DIGITS)

    ok = True

    # 场景A：EasyOCR 把数字1误识别为 4（conf 0.80）→ 应修正为 1
    def bad_ocr_a(arr):
        return [([0, 0, 10, 10], "4", 0.80)]
    eng = make_mock_engine(bad_ocr_a)
    arr = np.asarray(Image.open(os.path.join(IMG_DIR, files[0])).convert("RGB"))
    r = run(eng, crop_slot(arr, *maps[0][1]))
    print(f"[A] EasyOCR 误识别 1→4: 融合结果 {r['digit']} ({r['method']},{r['confidence']})")
    if r["digit"] != 1:
        ok = False

    # 场景B：EasyOCR 把数字1识别为孤立 "I" → 别名映射 + 模板一致 → 1
    def bad_ocr_b(arr):
        return [([0, 0, 10, 10], "I", 0.90)]
    eng = make_mock_engine(bad_ocr_b)
    r = run(eng, crop_slot(arr, *maps[0][1]))
    print(f"[B] EasyOCR 识别为 'I':  融合结果 {r['digit']} ({r['method']},{r['confidence']})")
    if r["digit"] != 1:
        ok = False

    # 场景C：EasyOCR 正确识别 2 → 与模板一致 → 2
    def good_ocr(arr):
        return [([0, 0, 10, 10], "2", 0.95)]
    eng = make_mock_engine(good_ocr)
    r = run(eng, crop_slot(arr, *maps[0][2]))
    print(f"[C] EasyOCR 正确识别 2:  融合结果 {r['digit']} ({r['method']},{r['confidence']})")
    if r["digit"] != 2:
        ok = False

    # 场景D：EasyOCR 正确识别 1 → 与模板一致 → 1
    def good_ocr1(arr):
        return [([0, 0, 10, 10], "1", 0.95)]
    eng = make_mock_engine(good_ocr1)
    r = run(eng, crop_slot(arr, *maps[0][1]))
    print(f"[D] EasyOCR 正确识别 1:  融合结果 {r['digit']} ({r['method']},{r['confidence']})")
    if r["digit"] != 1:
        ok = False

    # 场景E：EasyOCR 误识别 1→3 且模板匹配强 → 修正为 1（全 8 图×数字1）
    def bad_ocr_e(arr):
        return [([0, 0, 10, 10], "3", 0.85)]
    eng = make_mock_engine(bad_ocr_e)
    n_fixed = 0
    for v, fname in enumerate(files):
        a = np.asarray(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))
        r = run(eng, crop_slot(a, *maps[v][1]))
        if r["digit"] == 1:
            n_fixed += 1
    print(f"[E] EasyOCR 全错 1→3 时: 修正 {n_fixed}/8 张数字1")
    if n_fixed != 8:
        ok = False

    # 场景F：EasyOCR 不可用 → 纯模板匹配兜底
    eng = make_mock_engine(None)
    eng.available = False
    r = run(eng, crop_slot(arr, *maps[0][1]))
    print(f"[F] EasyOCR 不可用:       结果 {r['digit']} ({r['method']},{r['confidence']})")
    if r["digit"] != 1:
        ok = False

    print(f"\n融合场景验证: {'✅ 全部通过' if ok else '❌ 存在失败'}")
    return ok


if __name__ == "__main__":
    t1 = test_pure_template()
    t2 = test_fusion_scenarios()
    print(f"\n总结: 模板匹配 {'✅' if t1 else '❌'} | 融合逻辑 {'✅' if t2 else '❌'}")
    sys.exit(0 if (t1 and t2) else 1)
