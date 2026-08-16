#!/usr/bin/env python3
"""
生成模拟决赛场景的测试图片集
==============================
三种场景（模拟真实比赛条件）：
1. 开关面板：3个灯（红/黄/绿）+ 对应开关，随机亮1个灯
2. 数字方块：4个长方体顶面数字1-4（哑光表面）
3. 几何体：长方体/正方体/圆柱体/多面体竖直摆放

每类生成多个变体（角度/光照/位置扰动）用于评估鲁棒性。
输出到 tests/test_images/
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

# 中文字体尝试
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/arial.ttf",
]

def get_font(size):
    for f in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def add_noise_lighting(img, brightness_range=(-20, 20), noise_std=5):
    """模拟光照变化+噪声"""
    arr = np.array(img).astype(np.float32)
    arr += random.uniform(*brightness_range)
    arr += np.random.normal(0, noise_std, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ============================================================
# 场景1: 开关面板（3灯 + 开关，随机亮1灯）
# ============================================================
def gen_switch_panel(light_idx=None, n=6, prefix="panel_light"):
    """生成开关面板图，light_idx 0/1/2 或 None(随机)"""
    for i in range(n):
        idx = light_idx if light_idx is not None else random.randint(0, 2)
        W, H = 640, 480
        img = Image.new("RGB", (W, H), (200, 200, 205))  # 浅灰面板
        draw = ImageDraw.Draw(img)

        # 面板框
        draw.rectangle([60, 40, W-60, H-40], outline=(80, 80, 85), width=4)
        # 三个灯（水平排布，参考群聊截图：红黄绿）
        light_colors = [(200, 60, 60), (220, 200, 60), (60, 180, 60)]  # 灭灯色
        lit_colors = [(255, 60, 60), (255, 230, 60), (60, 255, 60)]    # 亮灯色
        light_positions = [(200, 120), (320, 120), (440, 120)]
        for j, (lx, ly) in enumerate(light_positions):
            color = lit_colors[j] if j == idx else light_colors[j]
            draw.ellipse([lx-30, ly-30, lx+30, ly+30], fill=color, outline=(60, 60, 60), width=3)
            if j == idx:  # 亮灯光晕
                draw.ellipse([lx-45, ly-45, lx+45, ly+45], outline=lit_colors[j], width=2)

        # 三个开关（灯下方）：按钮/拨杆/按钮
        switch_positions = [(200, 240), (320, 240), (440, 240)]
        for j, (sx, sy) in enumerate(switch_positions):
            if j == 1:  # 拨杆
                draw.rectangle([sx-25, sy-45, sx+25, sy+45], fill=(190, 190, 195), outline=(70, 70, 70), width=3)
                draw.rectangle([sx-6, sy-38, sx+6, sy+38], fill=(90, 90, 95))
            else:  # 按钮
                btn_color = (230, 80, 80) if j == 0 else (80, 200, 80)
                draw.ellipse([sx-25, sy-25, sx+25, sy+25], fill=btn_color, outline=(70, 70, 70), width=3)

        # 标签
        font = get_font(18)
        draw.text((80, 340), "SWITCH PANEL", fill=(40, 40, 40), font=font)
        img = add_noise_lighting(img)
        img.save(os.path.join(OUT_DIR, f"{prefix}{idx}_v{i}.png"))
    return n


# ============================================================
# 场景2: 数字方块（顶面数字1-4）
# ============================================================
def gen_number_cubes(n=8):
    """生成俯视的4个数字方块，顶面数字1-4"""
    for v in range(n):
        W, H = 640, 480
        img = Image.new("RGB", (W, H), (150, 150, 155))  # 桌面
        draw = ImageDraw.Draw(img)
        font = get_font(70)

        # 4个槽位位置（俯视）
        slots = [(120, 200), (260, 200), (400, 200), (520, 200)]
        random.shuffle(slots)  # 数字位置随机（模拟比赛）
        numbers = [1, 2, 3, 4]

        for num, (sx, sy) in zip(numbers, slots):
            # 方块顶面（正方形，哑光）
            size = 100
            x0, y0 = sx - size//2, sy - size//2
            draw.rectangle([x0, y0, x0+size, y0+size], fill=(190, 190, 192), outline=(70, 70, 72), width=3)
            # 数字
            text = str(num)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((sx - tw//2, sy - th//2 - 5), text, fill=(30, 30, 30), font=font)

        img = add_noise_lighting(img)
        img.save(os.path.join(OUT_DIR, f"cubes_v{v}.png"))
    return n


# ============================================================
# 场景3: 几何体（竖直摆放）
# ============================================================
def gen_shapes(n=8):
    """生成俯视的4个几何体（简化为俯视轮廓+伪3D）"""
    shapes = ["长方体", "正方体", "圆柱体", "三棱柱"]
    for v in range(n):
        W, H = 640, 480
        img = Image.new("RGB", (W, H), (150, 150, 155))
        draw = ImageDraw.Draw(img)

        positions = [(120, 200), (260, 200), (400, 200), (520, 200)]
        random.shuffle(positions)

        for shape, (sx, sy) in zip(shapes, positions):
            # 俯视轮廓（竖直摆放时从上看）
            if shape == "长方体":
                draw.rectangle([sx-50, sy-35, sx+50, sy+35], fill=(180, 130, 90), outline=(70, 50, 35), width=3)
            elif shape == "正方体":
                draw.rectangle([sx-40, sy-40, sx+40, sy+40], fill=(130, 130, 160), outline=(50, 50, 70), width=3)
            elif shape == "圆柱体":
                draw.ellipse([sx-40, sy-40, sx+40, sy+40], fill=(90, 160, 90), outline=(40, 70, 40), width=3)
            elif shape == "三棱柱":
                draw.polygon([(sx, sy-45), (sx+50, sy+40), (sx-50, sy+40)], fill=(160, 120, 160), outline=(70, 50, 70), width=3)

        img = add_noise_lighting(img)
        img.save(os.path.join(OUT_DIR, f"shapes_v{v}.png"))
    return n


if __name__ == "__main__":
    gen_switch_panel(light_idx=None, n=6, prefix="panel_random")  # 6张随机灯(无真值)
    for idx in (0, 1, 2):
        gen_switch_panel(light_idx=idx, n=2)    # 每灯2张已知真值
    n2 = gen_number_cubes(n=8)
    n3 = gen_shapes(n=8)
    total = len(os.listdir(OUT_DIR))
    print(f"✅ 测试图片生成完成: {total} 张 → {OUT_DIR}")
    print("   - 开关面板: 12 张（6随机panel_random+6已知panel_light）")
    print("   - 数字方块: 8 张")
    print("   - 几何体: 8 张")
