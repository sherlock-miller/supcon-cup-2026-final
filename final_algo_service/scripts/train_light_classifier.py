#!/usr/bin/env python3
"""任务1 策略二: 三灯分类模型训练（红/白/绿灯亮）
================================================
小数据集（每类约 10 张手机照片）→ MobileNetV3-Small 迁移学习。

策略（防过拟合）:
  1. ImageNet 预训练 backbone 冻结，只训分类头（数据少, 冻结最稳）
  2. 数据增强: 翻转/旋转/颜色抖动/亮度（模拟相机视角变化）
  3. 验证集: 每类留 2 张; 早停按验证 acc
  4. 输出: weights/light_classifier.pth (state_dict + 类别映射)

用法:
  python scripts/train_light_classifier.py --data "E:/hermes/中控杯决赛" \
      --epochs 20 --out weights/light_classifier.pth
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

CLASS_NAMES = ["green", "red", "white"]  # 字母序固定
CLASS_TO_LIGHT = {"green": "light_3", "red": "light_1", "white": "light_2"}


def build_dataset(data_dir: Path, val_per_class: int = 2, seed: int = 42):
    """从 {data_dir}/{class}/ 加载，每类留 val_per_class 张做验证"""
    rng = random.Random(seed)
    train_imgs, val_imgs = [], []
    for ci, cls in enumerate(CLASS_NAMES):
        cls_dir = data_dir / cls
        files = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        val_files = files[:val_per_class]
        train_files = files[val_per_class:]
        print(f"  {cls}: {len(train_files)} 训练 + {len(val_files)} 验证")
        train_imgs += [(p, ci) for p in train_files]
        val_imgs += [(p, ci) for p in val_files]
    return train_imgs, val_imgs


class LightDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def train(args):
    t0 = time.time()
    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_imgs, val_imgs = build_dataset(data_dir, args.val_per_class)
    if not train_imgs:
        raise SystemExit("无训练数据")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)
    # 冻结 backbone
    for p in model.features.parameters():
        p.requires_grad = False
    model = model.to(device)

    train_ds = LightDataset(train_imgs, build_transforms(True))
    val_ds = LightDataset(val_imgs, build_transforms(False))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=0)

    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            total += len(y)
            correct += (model(x).argmax(1) == y).sum().item()
        sched.step()
        train_acc = correct / max(total, 1)

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                v_total += len(y)
                v_correct += (model(x).argmax(1) == y).sum().item()
        val_acc = v_correct / max(v_total, 1)
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"loss={loss_sum/max(total,1):.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 阶段2: 解冻轻量微调（小学习率）
    if args.finetune_epochs > 0:
        print("阶段2: 解冻 backbone 微调...")
        for p in model.parameters():
            p.requires_grad = True
        opt2 = torch.optim.Adam(model.parameters(), lr=1e-5)
        for epoch in range(1, args.finetune_epochs + 1):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt2.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt2.step()
            model.eval()
            v_total, v_correct = 0, 0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    v_total += len(y)
                    v_correct += (model(x).argmax(1) == y).sum().item()
            val_acc = v_correct / max(v_total, 1)
            print(f"finetune {epoch}/{args.finetune_epochs}  val_acc={val_acc:.3f}")
            if val_acc >= best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存: state_dict + 元数据
    torch.save({
        "state_dict": best_state,
        "class_names": CLASS_NAMES,
        "class_to_light": CLASS_TO_LIGHT,
        "arch": "mobilenet_v3_small",
        "val_acc": round(best_acc, 4),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, out_path)
    print(f"\n✅ 模型已保存: {out_path}  (最佳验证 acc={best_acc:.3f}, "
          f"用时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/hermes/中控杯决赛")
    ap.add_argument("--out", default=str(ROOT / "weights" / "light_classifier.pth"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val-per-class", type=int, default=2)
    args = ap.parse_args()
    train(args)
