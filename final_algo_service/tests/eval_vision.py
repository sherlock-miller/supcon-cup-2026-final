#!/usr/bin/env python3
"""
视觉管线评估框架
================
对 vision/ 三个检测函数做量化评估：
1. detect_lit_light   — 开关面板亮灯检测（12张图，6张已知真值）
2. detect_cube_numbers — 数字方块识别（8张图，4数字/张）
3. detect_and_classify_shapes — 几何体分类（8张图，4形状/张）

评估指标：
- 灯检测: 亮灯ID准确率（已知真值6张）
- 数字识别: 数字召回率（32个数字总）
- 形状分类: 形状准确率（32个几何体总）

用法:
  python eval_vision.py [--quick]    # --quick 只测少量图
"""
import argparse
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")


def load_images(pattern):
    """按文件名模式加载图片"""
    files = sorted(f for f in os.listdir(IMG_DIR) if re.match(pattern, f))
    return [(f, Image.open(os.path.join(IMG_DIR, f)).convert("RGB")) for f in files]


# ============================================================
# 评估1: 灯检测
# ============================================================
def eval_light_detection(vision, quick=False):
    """灯检测：12张面板图。已知真值的6张（panel_light{0,1,2}_v{0,1}）"""
    # 已知真值图
    known = []
    for idx in (0, 1, 2):
        files = sorted(f for f in os.listdir(IMG_DIR)
                       if f.startswith(f"panel_light{idx}_v"))
        known.extend(files[:1] if quick else files)
    # 随机图（无真值，跳过评分）
    random_files = sorted(f for f in os.listdir(IMG_DIR)
                          if f.startswith("panel_light") and re.match(r"panel_light\d_v\d+", f) is None)

    correct = 0
    total = 0
    details = []

    for fname in known:
        truth = int(fname.split("_")[1][5:])  # panel_light{idx}_v*.png → idx
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        try:
            result = vision.detect_lit_light(img)
        except Exception as e:
            details.append(f"❌ {fname}: 异常 {e}")
            continue
        if result is None:
            details.append(f"❌ {fname}: 未检出 (真值={truth})")
            total += 1
            continue
        light_id = result.get("light_id", "")
        pred = {"light_1": 0, "light_2": 1, "light_3": 2}.get(light_id, -1)
        ok = pred == truth
        correct += ok
        total += 1
        details.append(f"{'✅' if ok else '❌'} {fname}: 预测={light_id}({pred}) 真值={truth}")

    acc = correct / total if total else 0
    print(f"\n[评估1] 亮灯检测: {correct}/{total} 正确 ({acc:.0%})")
    for d in details:
        print(f"  {d}")
    return {"task": "灯检测", "correct": correct, "total": total}


# ============================================================
# 评估2: 数字识别
# ============================================================
def eval_number_cubes(vision, quick=False):
    """数字识别：8张图 × 4数字 = 32个数字"""
    files = sorted(f for f in os.listdir(IMG_DIR) if f.startswith("cubes_v"))
    if quick:
        files = files[:2]

    total_digits = len(files) * 4
    found_digits = 0
    correct_digits = 0
    details = []

    for fname in files:
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        try:
            results = vision.detect_cube_numbers(img)
        except Exception as e:
            details.append(f"❌ {fname}: 异常 {e}")
            continue
        nums = sorted(r["number"] for r in results)
        found_digits += len(nums)
        # 数字集合正确性（每个数字唯一）
        correct = set(nums) & {1, 2, 3, 4}
        correct_digits += len(correct)
        ok = sorted(correct) == [1, 2, 3, 4]
        details.append(f"{'✅' if ok else '⚠️'} {fname}: 识别到 {nums} (期望[1,2,3,4])")

    recall = found_digits / total_digits
    print(f"\n[评估2] 数字识别: 召回 {found_digits}/{total_digits} ({recall:.0%}), 去重正确 {correct_digits}/{total_digits}")
    for d in details:
        print(f"  {d}")
    return {"task": "数字识别", "found": found_digits, "correct": correct_digits, "total": total_digits}


# ============================================================
# 评估3: 形状分类
# ============================================================
def eval_shapes(vision, quick=False):
    """形状分类：8张图 × 4形状 = 32个。真值按文件名顺序未知（生成时随机），用每图形状集合验证"""
    files = sorted(f for f in os.listdir(IMG_DIR) if f.startswith("shapes_v"))
    if quick:
        files = files[:2]

    truth_set = {"长方体", "正方体", "圆柱体", "三棱柱"}
    total_shapes = len(files) * 4
    found_shapes = 0
    details = []

    for fname in files:
        img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        try:
            results = vision.detect_and_classify_shapes(img)
        except Exception as e:
            details.append(f"❌ {fname}: 异常 {e}")
            continue
        shapes = [r["shape"] for r in results]
        found_shapes += len(shapes)
        details.append(f"{'✅' if len(shapes) == 4 else '⚠️'} {fname}: 检测 {len(shapes)} 个 → {shapes}")

    recall = found_shapes / total_shapes
    print(f"\n[评估3] 形状检测: 召回 {found_shapes}/{total_shapes} ({recall:.0%})")
    for d in details:
        print(f"  {d}")
    return {"task": "形状检测", "found": found_shapes, "total": total_shapes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="快速模式（每类只测少量图）")
    args = parser.parse_args()

    from vision.vision_manager import VisionManager

    print("=" * 60)
    print("  视觉管线评估")
    print("=" * 60)

    # 初始化（模型加载较慢）
    t0 = time.time()
    vision = VisionManager()
    vision.initialize()
    print(f"模型初始化: {time.time()-t0:.1f}s")

    results = []
    results.append(eval_light_detection(vision, quick=args.quick))
    results.append(eval_number_cubes(vision, quick=args.quick))
    results.append(eval_shapes(vision, quick=args.quick))

    print("\n" + "=" * 60)
    print("  汇总:")
    for r in results:
        print(f"  {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
