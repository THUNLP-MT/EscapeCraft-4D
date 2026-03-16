import json
import random
import os
from scipy.spatial import ConvexHull

import numpy as np
from legent import generate_scene
from legent import get_mesh_size
from legent.scene_generation.objects import get_default_object_db

from config import ITEM_CATEGORIES 


from config import *
from utils import *
from log_config import configure_logger

import logging

logging.getLogger("PIL").setLevel(logging.INFO)


logger = configure_logger(__name__)


__odb = get_default_object_db()


def apply_rotation_to_size(size, rotation):
    """Apply a 3D rotation to the size vector using the provided rotation (assumed in degrees).

    Args:
        size (list): [x_size, y_size, z_size]
        rotation (list): [x_rot, y_rot, z_rot], in degrees.

    Returns:
        list: size after rotation
    """
    if rotation == [90, 0, 0]:
        return [size[0], size[2], size[1]]

    elif rotation == [0, 90, 0]:
        return [size[2], size[1], size[0]]

    elif rotation == [0, 0, 90]:
        return [size[1], size[0], size[2]]

    else:
        return size


def __add_custom_prefab_to_odb(odb):
    for item in PREFABS_USED:
        name = os.path.normpath(os.path.join(PREFAB_DIR, PREFABS_USED[item]["path"]))
        path = name
        item_info = {
            "name": name,
            "custom_type": "interactable",
            "type": "interactable",
            "placeable_surfaces": [],
            "item_type": item,
        }

        if PREFABS_USED[item].get("scale", False):
            __origin_size = get_mesh_size(path)
            item_info["custom_scale"] = PREFABS_USED[item]["scale"]
            item_info["size"] = [
                PREFABS_USED[item]["scale"][i] * __origin_size[i] for i in range(3)
            ]

        elif PREFABS_USED[item].get("height", False):
            __origin_size = get_mesh_size(path)
            __scale_size = PREFABS_USED[item]["height"] / __origin_size[1]
            item_info["custom_scale"] = [__scale_size for _ in range(3)]
            item_info["size"] = [__scale_size * __origin_size[i] for i in range(3)]

        else:
            item_info["size"] = get_mesh_size(path)
        if PREFABS_USED[item].get("rotation", False):
            item_info["custom_rotation"] = PREFABS_USED[item]["rotation"]
            item_info["size"] = apply_rotation_to_size(
                item_info["size"], PREFABS_USED[item]["rotation"]
            )
        
        if PREFABS_USED[item].get("sound_file", False):
            item_info["sound_file"] = PREFABS_USED[item]["sound_file"]

        if PREFABS_USED[item].get("interaction_angle", False):
            item_info["interaction_angle"] = PREFABS_USED[item]["interaction_angle"]
        
        if PREFABS_USED[item].get("interaction_distance", False):
            item_info["interaction_distance"] = PREFABS_USED[item]["interaction_distance"]
        
        
        __size = {xyz: value for xyz, value in zip(["x", "y", "z"], item_info["size"])}
        item_info["size"] = __size
        odb.PREFABS[f"custom_{item}"] = item_info

    return odb


ODB = __add_custom_prefab_to_odb(__odb)
logger.debug("Custom prefabs added to ODB.")

class DefaultSceneGenerator:
    """
    This class is responsible for generating a game scene based on a game settings file. It initializes the game scene by
    generating a scene layout with specific items and doors, based on the configuration provided in the game settings file.
    """

    def __init__(
        self, game_setting_path: str, prefab_dir: str = PREFAB_DIR, room_num: int = 1
    ):

        self.prefab_dir = PREFAB_DIR
        self.interaction_items = {}
        self.game_setting_path = game_setting_path
        self.room_num = room_num

        self.scene_list = []

        for i in range(self.room_num):
            self.scene_list.append(self.__initialize_scene(i + 1))

        logger.info("Game scene initialized.")


    def save_scene(self):
        """
        Saves the game scene to a JSON file.

        Returns:
            - str: The path to the saved scene file or the first scene file of a list of multirooms saved.
        """
        level = os.path.splitext(os.path.basename(self.game_setting_path))[0]
        folder = os.path.dirname(self.game_setting_path)
        save_folder = os.path.join(folder, level)
        
        # print(level, folder, save_folder)
        
        if self.room_num > 1:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            try:
                folders = [f for f in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, f))]

                numeric_folders = [int(f) for f in folders if f.isdigit()]

                folder_number = max(numeric_folders, default=None) + 1
                
            except:
                folder_number = 1
            
            save_folder = os.path.join(save_folder, str(folder_number))
            
            for i, scene in enumerate(self.scene_list, start=1):
                with open(os.path.join(save_folder, f"{i}.json"), "w", encoding="utf-8") as f:
                    json.dump(scene, f, indent=4, ensure_ascii=False)
                    
            return_path = os.path.join(save_folder, "1.json")
            
        
        else:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            try:
                json_files = [f for f in os.listdir(save_folder) if f.endswith('.json')]

                numeric_filenames = [int(os.path.splitext(f)[0]) for f in json_files if os.path.splitext(f)[0].isdigit()]

                number = max(numeric_filenames, default=None) + 1
                
            except:
                number = 1
            
            with open(os.path.join(save_folder, f"{number}.json"), "w", encoding="utf-8") as f:
                json.dump(self.scene_list[0], f, indent=4, ensure_ascii=False)
                
            return_path = os.path.join(save_folder, f"{number}.json")
            
        logger.info(f"Scene saved.")
        
        return return_path
        
    
    def __initialize_scene(self, room_id=1):
        """
        Initializes the game scene for a given room ID. This function generates a scene layout with specific items and doors,
        based on the configuration provided in the game settings file.

        Parameters:
            - room_id (int): The identifier of the room being initialized. Default is 1. Determines if the room has an entrance door.

        Returns:
            - dict: A dictionary representing the generated scene, including room layout, item instances, and door positions.
        """
       
        feature_object_counts = {}
        critical_object_counts = {}
        decoration_object_counts = {}
        other_object_counts = {} 
        
        level_data = json.load(open(self.game_setting_path, "r", encoding="utf-8"))
        
        level_items_by_id = {}
        for item in level_data.get("room", {}).get("items", []):
            if "id" in item:
                level_items_by_id[item["id"]] = item
        
        # for item in level_data["room"]["items"]:
        #     if item["show"]:
        #         num = object_counts.get(f"custom_{item['type']}", 0) + 1
        #         object_counts[f"custom_{item['type']}"] = (num, item["id"])

  
        for item in level_data["room"]["items"]:
            if item["show"]:
                item_type = item['type']
           
                prefab_name = f"custom_{item_type}"
                item_data = (item.get("count", 1), item["id"])

            
            if item_type in ITEM_CATEGORIES["FEATURE"]:
                feature_object_counts[prefab_name] = item_data
            elif item_type in ITEM_CATEGORIES["CRITICAL"]:
                critical_object_counts[prefab_name] = item_data
            elif item_type in ITEM_CATEGORIES["DECORATION"]:
                decoration_object_counts[prefab_name] = item_data
            else:
                other_object_counts[prefab_name] = item_data

   
        categorized_counts = {
            "feature": feature_object_counts,
            "critical": critical_object_counts,
            "decoration": decoration_object_counts,
            "other": other_object_counts
        }
    
        scene = generate_scene(
           
            room_num=random.randint(ROOM_MIN, ROOM_MAX), 
          
            object_counts=categorized_counts
        )

        for instance in scene["instances"]:
            instance["type"] = "kinematic"
           
            item_type = None
            if instance["prefab"].startswith("custom_"):
                item_type = instance["prefab"].replace("custom_", "")
            elif "item_type" in instance and instance["item_type"] in PREFABS_USED:
                item_type = instance["item_type"]
            elif any(key in instance["prefab"] for key in PREFABS_USED.keys()):
                
                for key in PREFABS_USED.keys():
                    if key in instance["prefab"]:
                        item_type = key
                        break
            
            if item_type and item_type in PREFABS_USED:
                # 获取 item_id，用于从 level JSON 中查找配置
                item_id = instance.get("item_id")
                level_item = level_items_by_id.get(item_id) if item_id else None
                
                # 优先从 level JSON 读取 sound_file，如果没有则从 PREFABS_USED 读取
                # 即使 instance 中已经有 sound_file，也要优先使用 level JSON 中的配置
                if level_item and "sound_file" in level_item:
                    instance["sound_file"] = level_item["sound_file"]
                    logger.debug(f"Using sound_file from level JSON for {item_id}: {level_item['sound_file']}")
                elif "sound_file" not in instance and "sound_file" in PREFABS_USED[item_type]:
                    instance["sound_file"] = PREFABS_USED[item_type]["sound_file"]
                
                # 优先从 level JSON 读取 trigger_mode
                if level_item and "trigger_mode" in level_item:
                    instance["trigger_mode"] = level_item["trigger_mode"]
                    logger.debug(f"Using trigger_mode from level JSON for {item_id}: {level_item['trigger_mode']}")
                
                # interaction_angle 和 interaction_distance 保持原逻辑（从 PREFABS_USED 读取）
                if "interaction_angle" in PREFABS_USED[item_type] and "interaction_angle" not in instance:
                    instance["interaction_angle"] = PREFABS_USED[item_type]["interaction_angle"]
                
                if "interaction_distance" in PREFABS_USED[item_type] and "interaction_distance" not in instance:
                    instance["interaction_distance"] = PREFABS_USED[item_type]["interaction_distance"]


    

        __door_with_exit_idx = self.__get_wall_idx(scene["instances"])

        scene["instances"][__door_with_exit_idx]["prefab"] = EXIT_WALL

        exit_door = self.__get_exit_door(
            EXIT_DOOR, scene["instances"][__door_with_exit_idx]
        )

        if room_id > 1:
            __door_with_entrance_idx = self.__get_wall_idx(
                scene["instances"], exclude_idx=__door_with_exit_idx
            )
            entrance_door = self.__get_exit_door(
                EXIT_DOOR, scene["instances"][__door_with_entrance_idx]
            )

            entrance_door["item_id"] = "entrance"
            entrance_door["item_type"] = "entrance"
            scene["instances"].append(entrance_door)

        self.interaction_items["exit"] = len(scene["instances"]) - 1
        scene["instances"].append(exit_door)


        return scene

    def __get_wall_idx(self, instances, exclude_idx=None):
        """
        Selects a random wall index from the given instances that can potentially have a door.

        Parameters:
            - instances (list): A list of dictionaries representing scene instances, each containing 
                           information such as 'prefab' and 'position'.
            - exclude_idx (int, optional): The index of the instance to be excluded from selection. 
                                      Defaults to None.

        Returns:
            - int: The index of the selected wall instance.
        """
        __walls = []
        for idx, instance in enumerate(instances):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            if "Wall" in instance.get("prefab", ""):
                __walls.append((idx, instance["position"]))

        __points = np.array([[x, z] for _, (x, _, z) in __walls])

        hull = ConvexHull(__points)

        __wall_with_door_id = random.choice(hull.vertices)

        return __walls[__wall_with_door_id][0]

    def __get_exit_door(self, door, wall_instance):
        """
        Generates a door instance for the given wall instance.


        Args:
            - door (str): The path to the door prefab.
            - wall_instance (dict): A dictionary representing the wall instance, containing information such as 'position'.

        Returns:
            - dict: A dictionary representing the door instance, containing information such as 'prefab', 'position', and 'scale'.
        """
        door_prefab = os.path.normpath(os.path.join(PREFAB_DIR, door))
        prefabs = ODB.PREFABS
        DOOR_PREFAB = ODB.MY_OBJECTS["door"][0]
        target_size = [
            prefabs[DOOR_PREFAB]["size"]["x"],
            prefabs[DOOR_PREFAB]["size"]["y"],
            prefabs[DOOR_PREFAB]["size"]["z"],
        ]

        def get_door_scale(prefab, door_height, door_width):
            
            full_path = prefab
           
            __origin_size = get_mesh_size(full_path)
           


            def modify_array(arr):
                arr = np.array(arr)
                min_index = np.argmin([arr[0], arr[2]]) * 2
                max_index = 2 if min_index == 0 else 0

                arr[min_index] = 0.1
                arr[1] = 1
                arr[arr != 0.1] = 1

                return arr, max_index

            __latent_size, __width_index = modify_array(__origin_size)

            __door_scale = [1, door_height, 1]
            __door_scale[__width_index] = door_width
            __door_scale = np.array(__door_scale)

            __final_size = __door_scale * __latent_size

            __scale = __final_size / __origin_size

            return __scale.tolist()

        door_instance = {
            "prefab": door_prefab,
            "position": [
                wall_instance["position"][0],
                target_size[1] / 2,
                wall_instance["position"][2],
            ],
            "scale": get_door_scale(door_prefab, target_size[1], target_size[0]),
            "rotation": [
                wall_instance["rotation"][0],
                wall_instance["rotation"][1] - 90,
                wall_instance["rotation"][2],
            ],
            "item_type": "exit",
            "type": "kinematic",
            "item_id": "exit",
           
            "sound_file": AMBIENT_SOUNDS["exit_door"]["sound_file"],
            "sound_type": "ambient", 
            "volume": AMBIENT_SOUNDS["exit_door"]["volume"],
            "max_distance": AMBIENT_SOUNDS["exit_door"]["max_distance"],
            
        }

        return door_instance