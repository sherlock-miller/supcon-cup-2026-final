#!/usr/bin/env python3
"""任务1 三灯位置交互标定工具
================================
拍照位姿固定时，三个灯在相机画面中的像素位置固定。
此工具: 相机实时预览 → 鼠标依次点击三个灯中心 → 保存 ROI。

用法:
  python scripts/calibrate_lights.py

操作:
  1. 机械臂移到任务1拍照位姿（相机正对开关面板）
  2. 预览窗口中依次【左键点击】:
       灯1（红按钮）→ 灯2（拨动开关）→ 灯3（绿按钮）
      （灯没亮也没关系，点灯罩中心即可）
  3. 按 s 保存 → 现场配置/lights_roi.json
  4. 按 r 重置重来，按 q 退出

提示: 灯的位置应点在灯罩发光区域的中心。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from interactive_calib import Gemini335  # 复用标定脚本的相机封装

LIGHT_ORDER = [
    ("light_1", "灯1 - 红按钮"),
    ("light_2", "灯2 - 拨动开关"),
    ("light_3", "灯3 - 绿按钮"),
]
DEFAULT_RADIUS = 30
SAVE_PATH = Path(__file__).parent.parent / "现场配置" / "lights_roi.json"


def main():
    cam = Gemini335()
    cam.start()
    print("相机已启动。请依次点击三个灯的中心。")

    clicks = []
    win = ("灯位置标定 | 依次点击: 灯1(红按钮) 灯2(拨杆) 灯3(绿按钮) "
           "| s=保存 r=重置 q=退出")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((int(x), int(y)))
            print(f"  ✅ 已记录 {LIGHT_ORDER[len(clicks) - 1][1]} @ ({x}, {y})")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)

    try:
        while True:
            img = cam.grab_rgb()
            arr = np.asarray(img.convert("RGB"))[..., ::-1].copy()

            # 已点击的标记
            for i, (x, y) in enumerate(clicks):
                cv2.circle(arr, (x, y), 9, (0, 255, 0), 2)
                cv2.circle(arr, (x, y), 1, (0, 255, 0), 3)
                cv2.putText(arr, f"{i + 1}", (x + 14, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 状态提示
            if len(clicks) < 3:
                tip = f">>> 请点击: {LIGHT_ORDER[len(clicks)][1]}"
                cv2.putText(arr, tip, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (0, 200, 255), 2)
            else:
                cv2.putText(arr, "3 个灯已标定 —— 按 s 保存, r 重来",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, (0, 255, 0), 2)

            cv2.imshow(win, arr)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("退出（未保存）")
                break
            elif key == ord("r"):
                clicks.clear()
                print("已重置，重新点击")
            elif key == ord("s"):
                if len(clicks) < 3:
                    print(f"⚠️ 只点了 {len(clicks)} 个灯，需要 3 个")
                    continue
                out = {}
                for (lid, _label), (x, y) in zip(LIGHT_ORDER, clicks):
                    out[lid] = {"pixel_x": x, "pixel_y": y,
                                "radius": DEFAULT_RADIUS}
                SAVE_PATH.parent.mkdir(exist_ok=True)
                with open(SAVE_PATH, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"✅ 已保存: {SAVE_PATH}")
                print(f"   {json.dumps(out, ensure_ascii=False)}")
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == "__main__":
    main()
