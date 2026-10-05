---
name: topo2sketch
description: 把产品结构拓扑图/专利剖视图转成产品设计线稿（白底黑线 ControlNet 线稿）。当用户提到拓扑图转线稿、剖视图转设计草图、专利图转产品线稿、ControlNet lineart 出图、DreamShaper 线稿，或给出一张带标注的工程示意图并想要"干净的设计线稿"时使用——即使用户只说"把这张图变干净/转成线稿"。
---

# topo2sketch：拓扑图 → 产品设计线稿

输入一张**产品结构拓扑图或专利剖视图**（可带标注文字、引线、剖面线），输出干净的
白底黑线产品设计线稿，可直接作为下游 `sketch2showcase` 渲染管线的控制图。

管线 = OpenCV 预处理（去标注/引线/剖面线）→ ControlNet lineart + SD1.5 底模出图。

## 环境准备（一次性）

- NVIDIA GPU + Python venv。**PyPI 的 torch 是 CPU 版**，Blackwell（RTX 50 系）必须 cu128：
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install "diffusers>=0.40" accelerate transformers safetensors opencv-python pillow
  ```
- 下载模型（SD1.5 + lineart ControlNet，走 hf-mirror，约 6GB）：
  ```bash
  python scripts/download_models.py --root models
  ```

## 工作流（人机协同，逐图调参）

**第 1 步：预处理控制图。** 标注框坐标必须逐图人工调整——先跑一次生成 debug 叠框图，
用 Read 目检框位，再修正坐标重跑，直到标注/引线全部清除且结构线无损：

```bash
python scripts/prep_control.py input/topo.png --out output/control_clean.png \
  --erase-box 2,42,96,80 --erase-box 44,116,114,146 \
  --erase-line 114,134,150,150 \
  --hatch-zone 140,86,390,114 --hatch-zone 148,110,170,445 \
  --debug output/debug_boxes.png
```

- `--erase-box x1,y1,x2,y2`（可重复）：标注文字框，源图坐标，inpaint 抹除
- `--erase-line x1,y1,x2,y2`（可重复）：引线段走廊（画成 5px 宽 mask 后 inpaint）
- `--margin-rect`（可重复）：边缘空白带内完全包含的孤立连通域直接抹白
- `--hatch-zone`（可重复）：**画布坐标系**（默认 512×512 成品图）内的剖面线清除区，
  用横/竖开运算（K=7）保留结构线、抽掉 45° 斜线
- `--thresh`：二值化阈值（默认 170）；`--size`：输出方形画布边长（默认 512）

**第 2 步：出图。** 提示词必须描述产品本体（结构、材质、视角），NEG 已内置
防黑底/防剖面线/防图框：

```bash
python scripts/generate.py output/control_clean.png \
  --prompt "product design sketch of a hydrostatic testing machine, front view, \
            black ink linework on white paper, steel frame base" \
  --out output/sketch.png --seeds 101,202,303 --cfg 6.2 --scale 1.0 --steps 32
```

**第 3 步：逐幅目检**（Read 生成的 PNG），换 seed/调 `--cfg 5.5-7`、`--scale 0.85-1.0`
多出几轮挑优。精选线稿可再经 `sketch2showcase` 的 `prep_control.py` 转纯黑白。

## 必踩坑（都已固化在脚本默认值里，改动前先读懂）

- **剖面线会被 lineart ControlNet 放大成满幅条纹**：控制图必须在限定区域内清除斜线，
  这是 `--hatch-zone` 存在的原因。
- **SD1.5 原始底模在此类提示词下会输出深底白线或自绘专利图框**：NEG 已带
  `(dark background:1.4)`；仍复发就换 DreamShaper-8 等写实底模（`--base` 指向本地目录）。
- **huggingface_hub 新版走 Xet 后端在 hf-mirror 上会 401**：脚本已设
  `HF_HUB_DISABLE_XET=1`，不要删除。
- **每幅图必须 Read 目检**：文字/标号重叠、patch 超界、箭头穿框都不会自动报错。
- 主体结构必须与拓扑图严格 1:1，不许增删零件；发现 AI 自创零件就提高
  `--scale`（控制强度）或换 seed。

## 与下游管线的衔接

产出的设计线稿交给 `sketch2showcase` skill：经其 `prep_control.py` 转纯黑白后作为
渲染控制图；配合说明书文字可进一步用 cadquery 建模投影出标准三视图线稿。
