"""示例模型文件：水浴式水压试验机（直压内测式）简化几何，34 实体。

配套 sketch2showcase skill 的 cad_views.py 使用：
    python cad_views.py hydrostatic_tester_model.py --out-dir output/cad

模型文件契约：定义 build() 返回 cadquery 形体（Workplane/Shape/list）；
单位 mm，Z 轴向上，原点在底面投影中心；零件全部用 box/cylinder/revolve 基元。
结构来源：专利交底书图1整机结构示意图（机架/工位基座/灭火器瓶体/对接密封总成/
计量增压总成/标定机构/排气支路/回收排空/控制器）。
"""
import math

import cadquery as cq


def box(x1, y1, z1, x2, y2, z2):
    return cq.Workplane("XY", origin=((x1 + x2) / 2, (y1 + y2) / 2, z1)).box(
        x2 - x1, y2 - y1, z2 - z1, centered=(True, True, False))


def cylz(x, y, z1, z2, r):
    return cq.Workplane("XY", origin=(x, y, z1)).circle(r).extrude(z2 - z1)


def cylx(x1, x2, y, z, r):
    c = cq.Workplane("XY", origin=(x1, y, z)).circle(r).extrude(x2 - x1)
    return c.rotate((x1, y, z), (x1, y + 1, z), 90)


def build():
    s = []
    # 机架 100：底座 + 四立柱 + 顶梁
    s.append(box(-1200, -500, 0, 1200, 500, 100))
    for sx in (-900, 900):
        for sy in (-420, 420):
            s.append(box(sx - 40, sy - 40, 100, sx + 40, sy + 40, 1700))
    s.append(box(-950, -360, 1700, 950, 360, 1860))
    # 工位基座 200 定心托架 + 瓶体 10（XZ 面轮廓绕世界 Z 轴回转，直接直立）
    s.append(cylz(0, 0, 100, 200, 160))
    s.append(cq.Workplane("XZ")
             .polyline([(0, 200), (110, 200), (110, 740), (55, 800),
                        (35, 830), (35, 880), (0, 880)])
             .close().revolve(360, (0, 0), (0, 1)))
    # 对接密封总成 300：芯轴/密封圈/压紧盘/接头体/升降杆/压紧驱动件
    s.append(cylz(0, 0, 560, 880, 16))
    s.append(cq.Solid.makeTorus(26, 8, cq.Vector(0, 0, 824), cq.Vector(0, 0, 1)))
    s.append(cylz(0, 0, 880, 904, 90))
    s.append(box(-130, -100, 904, 130, 100, 1074))
    s.append(cylz(0, 0, 1074, 1440, 24))
    s.append(box(-160, -120, 1440, 160, 120, 1660))
    # 计量增压总成 400：支撑台/缸体（水平）/活塞杆/位移传感器/驱动件/压力传感器
    s.append(box(300, -180, 100, 1000, 180, 700))
    s.append(cylx(320, 920, 0, 780, 80))
    s.append(cylx(920, 1140, 0, 780, 22))
    s.append(box(980, -20, 800, 1040, 20, 850))
    s.append(box(1140, -120, 630, 1380, 120, 930))
    s.append(cylz(360, 0, 860, 915, 14))
    # 加压水路示意管
    s.append(cylx(150, 320, 0, 780, 8))
    s.append(cylz(150, 0, 780, 904, 8))
    s.append(cylx(0, 150, 0, 904, 8))
    # 标定机构 700：刚性基准筒 + 切换阀组
    s.append(cylz(650, -330, 100, 320, 60))
    s.append(box(420, -370, 100, 500, -290, 210))
    # 排气支路 500：支架/集气杯/液位开关/排气阀/连管
    s.append(box(-860, -20, 1240, -600, 20, 1270))
    s.append(cylz(-620, 0, 1270, 1380, 40))
    s.append(cylz(-620, 0, 1380, 1410, 12))
    s.append(box(-655, -35, 1180, -585, 35, 1270))
    s.append(cylx(-585, -130, 0, 1335, 7))
    # 回收排空总成 600：回收水箱/单向阀/气源
    s.append(box(-900, -190, 100, -400, 190, 400))
    s.append(box(-360, -180, 100, -290, -110, 170))
    s.append(box(40, -460, 100, 260, -240, 360))
    # 控制器 800：控制柜
    s.append(box(-1160, 220, 100, -880, 480, 950))
    return s
