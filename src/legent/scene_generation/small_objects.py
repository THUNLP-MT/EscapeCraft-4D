import copy
import json
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

from legent.scene_generation.objects import ObjectDB#物体数据
from legent.scene_generation.room import Room#房间信息
from legent.server.rect_placer import RectPlacer#防止重叠

# from legent.utils.io import log

MAX_OBJECT_NUM_ON_RECEPTACLE = 3#定义单个容器表面（比如一张桌子）上面最多可以放几个物体

MAX_PLACE_ON_SURFACE_RETRIES = 10#在一个表面上尝试为单个物体寻找不重叠位置的最大次数，如果尝试10次都失败了，就放弃这个位置

SMALL_OBJECT_MIN_MARGIN = 0.1#放置小物体时，距离其它小物体的最小间距，放置穿模


def log(*args, **kwargs):
    pass

#
def prefab_fit_surface(prefab_size, surface, receptacle):
    px, py, pz = prefab_size.values()#获取小物件的长、高、宽
    # print(px, pz)
    sx = surface["surface"]["x_max"] - surface["surface"]["x_min"]#计算课放置表面的长和宽
    sz = surface["surface"]["z_max"] - surface["surface"]["z_min"]
    #检查容器（比如桌子）是否被旋转过90度或者270度
    #如果是，那么它的局部坐标系下的长和宽需要交换才能匹配世界坐标
    #这是在暗示测量的度量衡是固定的，太精准了
    #从哪里说明了测量的度量衡是固定的？
    if receptacle["receptacle"]["rotation"][1] != 0 and receptacle["receptacle"]["rotation"][1] != 180:
        sx, sz = sz, sx
    #获取当前表面上已经放置了多少个小物件
    total_num = receptacle["small_object_num"]
    #这个式子判断了很多东西
    #1.小物体的长小于表面长度的90%
    #2.小物件的宽度是否小于表面宽度（90%）
    #3.当前表面上已经放置的小物体数量是否小于最大允许数量
    if px < (sx * 0.9) and pz < (sz * 0.9) and total_num < MAX_OBJECT_NUM_ON_RECEPTACLE:
        return True
    return False

# ==========================================================================================
# ================================ 函数改造开始 ============================================
# ==========================================================================================


#这个是3d场景中自动放置小物体的功能
def add_small_objects(
    objects: List[Dict[str, any]],
    odb: ObjectDB,
    rooms: Dict[int, Room],
    max_object_types_per_room: int = 10000,
    placer_bbox=(0, 0, 0, 0),
    # --- 参数修改: object_counts 现在用于通用/未分类物品 ---
    object_counts: Dict[str, int] = {},
    # --- 新增参数: 接收从上游传来的分类好的物品字典 ---
    feature_object_counts: Dict[str, int] = {},
    critical_object_counts: Dict[str, int] = {},
    # --- 新增结束 ---
    specified_object_instances: Dict[str, int] = {},
    receptacle_object_counts: Dict[str, int] = {},
):

    small_objects = []
    placer = RectPlacer(placer_bbox)

    objects_per_room = defaultdict(list)
    for obj in objects:
        room_id = obj["room_id"]
        objects_per_room[room_id].append(obj)
    objects_per_room = dict(objects_per_room)
    receptacles_per_room = {room_id: [obj for obj in objects if odb.OBJECT_TO_TYPE[obj["prefab"]] in odb.RECEPTACLES and obj["is_receptacle"]] for room_id, objects in objects_per_room.items()}

    # --------------------------------------------------------------------------------------
    # --- 核心改造 1: 将原有的放置逻辑封装成一个可复用的内部函数 ---
    # --------------------------------------------------------------------------------------
    def _place_item_category(counts_dict: Dict, available_surfaces: List[Dict]):
        """
        一个通用的放置函数，负责将给定类别(counts_dict)的物品放置到指定的可用表面(available_surfaces)上。
        此函数的核心逻辑完全来自于您提供的原始代码中的 'if object_counts:' 分支。
        """
        if not counts_dict or not available_surfaces:
            return

        for k, v in counts_dict.items():
            if isinstance(v, tuple):
                v, item_id = v
            else:
                item_id = None
            
            for _ in range(v):
                random.shuffle(available_surfaces)
                prefab = odb.PREFABS[k]
                success_flag = False
                for surface in available_surfaces:
                    receptacle = surface
                    if prefab_fit_surface(prefab["size"], surface, receptacle):
                        small_object = {}
                        small_object["prefab"] = prefab["name"]

                        # === 核心坐标计算 (逻辑不变) ===
                        x_min = ((surface["surface"]["x_min"] + receptacle["receptacle"]["position"][0]) if receptacle["receptacle"]["rotation"][1] == 0 or receptacle["receptacle"]["rotation"][1] == 180 else (surface["surface"]["z_min"] + receptacle["receptacle"]["position"][0]))
                        x_max = ((surface["surface"]["x_max"] + receptacle["receptacle"]["position"][0]) if receptacle["receptacle"]["rotation"][1] == 0 or receptacle["receptacle"]["rotation"][1] == 180 else (surface["surface"]["z_max"] + receptacle["receptacle"]["position"][0]))
                        z_min = ((surface["surface"]["z_min"] + receptacle["receptacle"]["position"][2]) if receptacle["receptacle"]["rotation"][1] == 0 or receptacle["receptacle"]["rotation"][1] == 180 else (surface["surface"]["x_min"] + receptacle["receptacle"]["position"][2]))
                        z_max = ((surface["surface"]["z_max"] + receptacle["receptacle"]["position"][2]) if receptacle["receptacle"]["rotation"][1] == 0 or receptacle["receptacle"]["rotation"][1] == 180 else (surface["surface"]["x_max"] + receptacle["receptacle"]["position"][2]))
                        
                        x_margin = prefab["size"]["x"] / 2 + SMALL_OBJECT_MIN_MARGIN
                        z_margin = prefab["size"]["z"] / 2 + SMALL_OBJECT_MIN_MARGIN
                        sample_x_min = x_min + x_margin
                        sample_x_max = x_max - x_margin
                        sample_z_min = z_min + z_margin
                        sample_z_max = z_max - z_margin

                        # 避免采样范围无效
                        if sample_x_min >= sample_x_max or sample_z_min >= sample_z_max:
                            continue

                        for _ in range(MAX_PLACE_ON_SURFACE_RETRIES):
                            x, z = np.random.uniform(sample_x_min, sample_x_max), np.random.uniform(sample_z_min, sample_z_max)
                            if placer.place(k, x, z, prefab["size"]["x"] + 2 * SMALL_OBJECT_MIN_MARGIN, prefab["size"]["z"] + 2 * SMALL_OBJECT_MIN_MARGIN):
                                y = (receptacle["receptacle"]["position"][1] + surface["surface"]["y"] + odb.PREFABS[k]["size"]["y"] / 2)
                                small_object["position"] = (x, y, z)
                                small_object["type"] = "kinematic"
                                small_object["parent"] = receptacle["receptacle"]["prefab"]
                                small_object["scale"] = prefab.get("custom_scale", [1, 1, 1])
                                small_object["rotation"] = prefab.get("custom_rotation", [0, 0, 0])
                                if prefab.get("item_type", False):
                                    small_object["item_type"] = prefab["item_type"]
                                if item_id:
                                    small_object["item_id"] = item_id
                                small_objects.append(small_object)
                                surface["small_object_num"] += 1
                                receptacle["small_object_num"] += 1
                                success_flag = True
                                break
                        if success_flag:
                            break
                if not success_flag:
                    print(f"Warning: Failed to place object {k}")
                    # raise Exception(f"Failed to place object {k}") # 建议改为警告而不是中断，以增加场景生成成功率

    # --------------------------------------------------------------------------------------
    # --- 核心改造 2: 按优先级顺序调用新的放置函数 ---
    # --------------------------------------------------------------------------------------

    # a. 首先，一次性收集场景中所有可用的表面
    all_surfaces = []
    for room_id, room in rooms.items():
        if room_id not in receptacles_per_room:
            continue
        receptacles_in_room = receptacles_per_room[room_id]
        for receptacle in receptacles_in_room:
            placeable_surfaces = odb.PREFABS[receptacle["prefab"]].get("placeable_surfaces", [])
            if not placeable_surfaces:
                continue
            for surface in placeable_surfaces:
                all_surfaces.append({
                    "receptacle": receptacle,
                    "surface": surface,
                    "small_object_num": 0, # 每个表面单独计数
                })
    
    # b. 【第一优先级】放置特色/大件物品 (FEATURE)
    if feature_object_counts:
        # (可选逻辑): 为大件物品筛选出尺寸足够大的 "优质" 表面
        large_surfaces = []
        for s in all_surfaces:
            surface_width = abs(s["surface"]["x_max"] - s["surface"]["x_min"])
            surface_depth = abs(s["surface"]["z_max"] - s["surface"]["z_min"])
            # 示例: 只在长或宽大于0.5米的表面上放置特色物品
            if surface_width > 0.5 or surface_depth > 0.5:
                 large_surfaces.append(s)
        
        print(f"Placing {len(feature_object_counts)} feature items on {len(large_surfaces) if large_surfaces else len(all_surfaces)} surfaces...")
        _place_item_category(feature_object_counts, large_surfaces if large_surfaces else all_surfaces)

    # c. 【第二优先级】放置关键道具 (CRITICAL)
    if critical_object_counts:
        # 关键道具可以使用所有表面。由于placer已经记录了特色物品的位置，这里会自动避开它们。
        print(f"Placing {len(critical_object_counts)} critical items...")
        _place_item_category(critical_object_counts, all_surfaces)

    # d. 【最低优先级】放置其他/通用物品
    if object_counts:
        print(f"Placing {len(object_counts)} other items...")
        _place_item_category(object_counts, all_surfaces)

    # (旧的 `if object_counts:` 逻辑块已被上面的优先级调用取代，因此可以删除)

    # ... 您原有的 `if receptacle_object_counts:` 逻辑可以保留，因为它处理的是更特殊的指定容器的放置 ...
    # ... 您原有的文件末尾的随机填充逻辑也可以保留 ...
    
    # 这里为了清晰，我将省略原有的 receptacle_object_counts 和随机填充部分的代码
    # 在您的实际文件中，应将本节的 a, b, c, d 代码块插入到 `receptacle_object_counts` 逻辑之前
    
    log(f"receptacle_object_counts: {receptacle_object_counts}") # 示例保留
    # ... (您原有的 if receptacle_object_counts: ... 分支代码)
    
    # ... (您原有的文件末尾的随机填充 for 循环代码) ...


    return small_objects, placer