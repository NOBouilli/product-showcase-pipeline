---
name: sketch2showcase
description: 把产品设计线稿/工程线稿变成产品展示图全套——渲染图、三视图、核心结构细节及放大图。当用户提到产品展示图、三视图渲染、细节放大图、爆炸图线稿渲染、线稿上色渲染、产品摄影风渲染、整体实物渲染、核心结构实物渲染、整体结构等距示意图、ControlNet lineart 渲染、MistoLine、RealVisXL、说明书配图，或给出线稿/工程图想要"真实感产品图/渲染图/效果图"时使用——即使用户只说"把这张线稿变成真机效果图"。
---

# sketch2showcase：线稿 → 产品展示图（三视图/细节放大）

输入**白底黑线的线稿控制图**（上游 `topo2sketch` 的产物、CAD 投影线稿或手绘稿），
输出摄影级产品渲染图、标准三视图与核心结构细节放大图。

**铁律：AI 只负责"怎么好看"，不负责"长什么样"。** 所有控制图必须来自几何投影或既有
线稿，禁止 txt2img 直出——让 AI 变视角或脑补结构必然编造零件。整体结构等距示意图
（解剖图、发丝线 HTML 导出、等距总览）同样是**几何控制图**，先做成白底黑线再进
ControlNet；禁止只把它写进提示词当风格参考。

## 环境准备（一次性）

- NVIDIA GPU（16GB 显存直跑；16GB **物理内存**加载时要靠脚本里的 `device_map`
  流式方案，已内置）。torch 必须 cu128（Blackwell）：
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install "diffusers>=0.40" accelerate transformers safetensors pillow
  ```
- 下载模型（RealVisXL_V5 底模 + MistoLine 线稿控制网 + x4 放大器，约 12GB）：
  ```bash
  python scripts/download_models.py --root models
  ```
- CAD 投影路线（可选）另建一个 venv 装 cadquery：`pip install cadquery`

## 管线总览

```
说明书/拓扑图 ──▶ 结构解析(对话中完成,产出调度JSON) ──▶ 视图规划
线稿来源 A：上游 topo2sketch 产出的设计线稿 ──▶ prep_control 转纯黑白 ─┐
线稿来源 B：cadquery 模型文件 ──▶ cad_views.py 投影三视图线稿 ────────┤
整体结构等距示意图 ──▶ 白底黑线（--iso-ref，第二路 MistoLine）────────┤
                                                                      ▼
                    generate_showcase.py
                      --mode ortho    单线稿（默认，三视图）
                      --mode overall  线稿 + 等距图 → 整体实物渲染
                      --mode core     线稿 + 等距图（可 --iso-box）→ 核心结构实物渲染
                                                                      ▼
                                   detail_zoom.py 裁局部 x4 放大 = 核心结构细节图
```

## 第 1 步：控制图准备

**A. 既有线稿**（设计线稿带浅灰填充时必须转纯黑白，否则灰调泄漏进渲染）：

```bash
python scripts/prep_control.py sketch1.png sketch2.png --out-dir output/control --thresh 218
```

阈值 218 适用于"背景纯白、线条浅灰"；**过低（如 140）会把浅灰结构线当背景丢光**。
图上的尺寸标注等杂物先用 `--erase-box x1,y1,x2,y2`（源图坐标白块）抹掉。

**B. CAD 投影三视图**（推荐用于三视图需求——结构绝对准确）。写一个模型文件
（约定：`build()` 返回 cadquery 形体；单位 mm；Z 轴向上；原点在底面投影中心；
零件全部用 box/cylinder/revolve 基元即可），然后：

```bash
python scripts/cad_views.py my_machine.py --out-dir output/cad --views front,side,top
```

生成 `view_front/side/top.png` 三张正交线稿，直接作为渲染控制图。模型文件写法
参考仓库 `examples/hydrostatic_tester_model.py`（水压试验机，34 实体）。

**不要用 AutoCAD 的 SOLVIEW/VIEWBASE 做脚本投影**——交互提示在 SendCommand 下
不可控，实测会把 AutoCAD 卡死到只能 taskkill。cadquery 全确定性。

## 第 2 步：渲染展示图

```bash
python scripts/generate_showcase.py output/cad/view_front.png \
    --product "automatic hydrostatic testing machine for fire extinguisher cylinders" \
    --view "front view" --seeds 101,202 --scale 0.7
```

- `--style render` 摄影级（默认）| `--style clay` 工业设计灰模
- `--view` 直接进提示词："front view" / "side view" / "top view" / "three-quarter view"
- `--scale 0.6-0.9`：越高越贴线稿、越低越自由
- 不传 `--iso-ref`、`--mode` 保持 `ortho` 时，控制网、提示词和文件名与旧版相同
- 单张 1024×30 步约 1 分钟；每幅 **必须 Read 目检**，多 seed 挑优

**三视图要风格统一**：三个视图共用同一种 `--style`、相近 seed、相同产品提示词，
并且用 `--mode ortho`（默认），不要把等距图混进正交视图。

### 整体实物渲染 / 核心结构实物渲染

等距示意图是第二路**几何**控制，不是风格图。脚本用同一份 MistoLine 做
MultiControlNet：第一路主线稿，第二路等距图转出的白底黑线。16GB 显存只加载一份
ControlNet 权重。提示词只加取景（整机入画 / 核心区特写），不写零件清单。

```bash
# 整体实物渲染。主线稿提供零件轮廓，等距图锁定整机布局。
python scripts/generate_showcase.py output/cad/view_front.png \
    --iso-ref output/iso_overview.png --mode overall \
    --product "automatic hydrostatic testing machine for fire extinguisher cylinders" \
    --view "isometric three-quarter view" --seeds 101,202

# 核心结构实物渲染。--iso-box 在等距图上圈核心区；--control-box 可复用同一细节区裁线稿。
python scripts/generate_showcase.py output/cad/view_front.png \
    --iso-ref output/iso_overview.png --mode core \
    --iso-box 120,80,400,360 --control-box 390,430,680,720 \
    --product "automatic hydrostatic testing machine for fire extinguisher cylinders" \
    --view "close-up of the core assembly" --seeds 101
```

- `--mode overall` 必须有 `--iso-ref`。输出 `*_overall_render_<seed>.png`（clay 则是 `*_overall_clay_<seed>.png`）
- `--mode core` 必须有 `--iso-ref`。`--iso-box x1,y1,x2,y2` 可选，源图像素，裁等距图上的核心区。输出 `*_core_render_<seed>.png`
- `--control-box` 可选，把主线稿裁到细节区（坐标约定与 `detail_zoom.py --box` 相同）。不传则整张线稿进控制网
- `--iso-erase-box` 可重复，先于裁切把标注涂白
- `--iso-method auto|thresh|edges|dark-stroke`：auto 对黑线等距/解剖图留深色笔划，浅灰线稿用阈值，彩色填色图抽边缘。透明底 PNG 会先合成到白底
- `--iso-scale` 默认 overall `0.85`、core `0.75`；主线稿仍用 `--scale`（默认 0.7）
- 两路控制图都会等比放进方形白底（不拉伸）。旁边会落下 `*_overall_line_ctrl.png` / `*_overall_iso_ctrl.png`（core 同理），目检几何再挑 seed
- 两张控制图构图差太远时会互相拉扯：调 `--scale` / `--iso-scale`，不要往提示词里补结构

`detail_zoom.py` 仍是「从已有展示图裁一块再 x4」。`--mode core` 是另一步：直接出核心结构的实物渲染，几何来自等距图裁切 + 线稿。

## 第 3 步：核心结构细节及放大图

```bash
python scripts/detail_zoom.py output/showcase/xxx_render_101.png --box 380,120,760,500
```

`--box` 为原图像素坐标，裁剪区 400-600px 见方（输出 4 倍边长）。先 Read 展示图
确认核心结构位置再定框。输出即"核心结构细节及放大图"。

## 调度清单（多视图任务先写 JSON 再执行）

视图多于一张时，先把规划落成 `showcase_plan.json` 再逐条执行，避免遗漏：

```json
{
  "_meta": {"product": "全氟己酮灭火器瓶体自动化耐压检测设备（直压内测式）"},
  "iso_ref": "output/control/ctrl_iso_overview.png",
  "views": [
    {"view": "front", "mode": "ortho", "control": "output/cad/view_front.png",
     "prompt_view": "front view", "seeds": [101, 202], "best": "…"},
    {"view": "side", "mode": "ortho", "control": "output/cad/view_side.png",
     "prompt_view": "side view", "seeds": [101, 202]},
    {"view": "overall", "mode": "overall", "control": "output/cad/view_front.png",
     "iso_ref": "output/control/ctrl_iso_overview.png",
     "prompt_view": "isometric three-quarter view", "seeds": [101, 202]}
  ],
  "exploded": null,
  "details": [
    {"part": "对接密封头+瓶口区", "src": "output/showcase/…_render_202.png",
     "box": "390,430,680,720"},
    {"part": "核心机架", "mode": "core",
     "control": "output/cad/view_front.png",
     "iso_ref": "output/control/ctrl_iso_overview.png",
     "iso_box": "120,80,400,360",
     "control_box": "390,430,680,720",
     "prompt_view": "close-up of the core assembly", "seeds": [101, 202]}
  ]
}
```

`mode` 为 `ortho`（可省略，等于旧行为）、`overall` 或 `core`。`iso_ref` 可写在计划顶层，
条目里再覆盖。`iso_box` 裁等距示意图，`control_box` 裁主线稿；`details[].box` 仍是
`detail_zoom.py` 用的展示图裁切框，和 `iso_box` 不是同一个坐标。

说明书文本 + 拓扑图 → 这份 JSON 的解析工作在对话中完成（部件清单、装配关系、
核心件识别、视图规划），JSON 落盘后按条目出图并回填 `best` 字段。

## 必踩坑（均已固化在脚本里，改动前先读懂）

- **16GB 物理内存会在加载 SDXL 时 MemoryError**：controlnet 用 `device_map="cuda"`、
  管线用 `device_map="balanced"` 流式进显存，**不能** `.to("cuda")` 前在 CPU 攒齐全套权重。
- **RealVisXL 的 model_index 缺 `feature_extractor`/`image_encoder`**：from_pretrained
  必须显式传 `image_encoder=None, feature_extractor=None`，且 `controlnet=` 要在
  管线构造时传入。
- **hf-mirror 的 Xet 401**：脚本已设 `HF_HUB_DISABLE_XET=1`。
- **乱码铭牌字**：NEG 已带 `(logo:1.4),(sticker:1.3),(nameplate:1.4),(label:1.3)` +
  POS `unbranded`；小铭牌可压掉，大字位（顶板正面）靠挑 seed。
- **AI 会自由发挥"合理"细节**（压力表、控制箱、管束填充）：展示图可接受，
  结构性图纸不接受——那正是控制图必须来自几何投影的原因。等距示意图只走控制网，
  不要把它的零件名单写进 `--product` / `--view`。
- **双控制图不要拉伸**：overall/core 等比垫白进 1024。正交 `--mode ortho` 仍按旧行为直接缩放到方形。
- **等距图和正交图对不齐**时两路残差会打架。整体渲染让 `--iso-scale` 高于 `--scale`；
  仍然乱就收小其中一路，而不是改提示词编结构。
- **爆炸图**：没有成熟的 2D AI 生成方案；正路是 CAD 几何爆炸装配（如 FreeCAD
  ExplodedAssembly）出爆炸布局线稿后，再走本管线 `--view "exploded view"` 渲染。
