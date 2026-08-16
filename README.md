# 中控杯 2026 决赛 — 汪汪队

> 2026 第二届"中控杯"智能制造挑战赛 · 赛道一「AI赋能」赛题2「工业多模态感知与无人化智能操作」
> **决赛**: 2026.8.19-20 杭州白马湖国际会展中心 B 馆（2号馆 = 外操无人化赛区）
> 队伍: 汪汪队 | 初赛仓库: [supcon-cup-2026](https://github.com/sherlock-miller/supcon-cup-2026)（master 分支）

---

## 决赛任务

以**单臂机械臂 + 灵巧手 + Gemini335 深度相机**为载体，三项递进式任务：

| 任务 | 内容 | 判分 |
|------|------|------|
| 任务1 | 拨按开关（视觉定位亮灯→点按/拨动对应开关） | 100 |
| 任务2 | 长方体有序转运（顶面数字 1-4 按序抓取放置） | 100 |
| 任务3 | 几何体无序分拣（形状识别→对应槽位，全部竖直摆放） | 100 |

**竞赛软件只调用 4 个 HTTP 端点**，其余全部由选手算法自主完成：

```
GET  /api/health          → {"success": true, "message": "ready"}
POST /api/task1/execute   → 完成开关操作后返回
POST /api/task2/execute   → 完成长方体转运后返回
POST /api/task3/execute   → 完成几何体分拣后返回
```

## 仓库结构

```
├── final_algo_service/        ← 算法服务核心（FastAPI + 机械臂SDK + 灵巧手SDK + 视觉 + 三任务编排）
│   ├── app.py                ← 4 端点主服务
│   ├── hardware/             ← FTArm B9 机械臂(:8087) + 灵巧手(:8088) HTTP SDK
│   ├── vision/               ← CLIP/GroundingDINO/EasyOCR + eye-in-hand 坐标变换
│   ├── tasks/                ← 三任务编排（安全序列+异常恢复）
│   ├── scripts/              ← 标定半自动化/预热/调试工具/硬件自检
│   ├── tests/                ← 47 个 pytest 用例（hermes verify 全绿）
│   └── Dockerfile + 部署脚本  ← Docker 镜像 / wheels 离线双轨部署
├── 相机服务/                  ← 宿主机相机 HTTP 服务（Docker 部署时 USB 透传方案）
├── 资料整理/                  ← 官方文档+群聊情报 豆包视觉识别存档
├── 架构与开发范围说明.md       ← 队友培训材料
└── 交接清单-现场必读.md        ← 现场必读（含 8.19/8.20 日程、分工、标定流程）
```

## 快速开始

```bash
cd final_algo_service

# 本地 Mock 测试（无需硬件）
pip install -r requirements.txt pytest
python -m pytest tests/ -q        # 56 passed

# 现场部署（三选一，推荐①镜像分发）
bash build_image.sh && bash run_on_site.sh    # Docker 镜像（U盘携带）
bash download_wheels.sh                        # wheels 离线包（Win11 工控机推荐）
docker pull ghcr.io/sherlock-miller/supcon-cup-2026-final:latest   # 见下方「镜像分发」章节
```

## 镜像分发（CI 自动构建，推荐）

CI（GitHub Actions）在每次 push 到 `main` 时自动构建并发布 Docker 镜像到 GHCR。
**镜像内已包含全部模型权重（CLIP + GroundingDINO + EasyOCR）**，队友拉取后即可运行，
无需本地构建（不用装 torch/下载模型，也不用关心构建环境）：

```bash
# 1. 拉取镜像（约 3GB，含模型，首次下载耗时取决于网速）
docker pull ghcr.io/sherlock-miller/supcon-cup-2026-final:latest
# 需要指定版本时用 commit 短号（每次构建对应一个短号标签，GitHub 镜像包页面可查）
docker pull ghcr.io/sherlock-miller/supcon-cup-2026-final:<commit短号>

# 2. 打成本地脚本认识的标签（run_on_site.sh / docker-compose.yml 引用 wangwang-final:latest）
docker tag ghcr.io/sherlock-miller/supcon-cup-2026-final:latest wangwang-final:latest

# 3. 启动（与现场部署流程一致）
cd final_algo_service
bash run_on_site.sh
```

> **维护者注意（一次性操作）**：GHCR 镜像默认私有。首次构建成功后，仓库管理员需到
> GitHub 仓库页 → Packages → 选择 `supcon-cup-2026-final` 镜像 → Package settings →
> Danger Zone → Change visibility → 改为 **Public**，否则队友拉取会报 401 denied。

> **国内拉取加速**：ghcr.io 直连较慢时，可配置 Docker 镜像加速器
> （Linux 编辑 `/etc/docker/daemon.json` 后 `sudo systemctl restart docker`；
> Windows Docker Desktop 在 Settings → Docker Engine 中修改）：
> `{"registry-mirrors": ["https://docker.m.daocloud.io"]}`
> 注意加速器仅对 Docker Hub 生效，ghcr.io 一般走直连（现场有有线网口，通常可达）。

> **离线兜底**：无法联网的环境仍走 U 盘方案 —— 有网电脑 `bash build_image.sh` 构建并导出
> `汪汪队决赛镜像.tar`，现场 `docker load` 后 `bash run_on_site.sh`。

## 关键情报（详见 资料整理/00-关键情报汇总.md）

- 工控机 **Win11 无预装环境**，现场每组仅 1 个有线网口（自备转接头）
- 机械臂 FTArm B9（HTTP :8087）+ 灵巧手（HTTP :8088），官方 HTTP API 跨平台
- 相机 Gemini335 装机械臂末端（eye-in-hand），SDK/内外参自备自标定
- 8.19 全天调试（19:00 前确认交卷锁屏），8.20 上午评委实操演示
- 任务2 数字仅顶面；任务3 几何体全部竖直摆放
