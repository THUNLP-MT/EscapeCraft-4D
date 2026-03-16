import os

from legent import (
    Environment,
    ResetInfo,
    save_image,
    Action,
)
from legent.action.api import (
    HideObject,
    ObjectInView,
)
from legent.utils.math import vec_xz, distance

from config import GAME_CACHE_DIR, MAX_INTERACTION_DISTANCE
from log_config import configure_logger

logger = configure_logger(__name__)


class LegentGame:
    """
    A wrapper for the Legent simulation environment.
    This class handles the low-level interactions with the game engine,
    such as initializing the scene, taking screenshots, and executing actions.
    """
    def __init__(self, scene, camera_resolution_width=2048, camera_resolution_height=1024):
        self.scene = scene
        self.env = Environment(
            env_path="auto",
            camera_resolution_width=camera_resolution_width,
            camera_field_of_view=120,
            camera_resolution_height=camera_resolution_height
        )
        self.scene["player"]["prefab"] = "null"
        self.scene["player"]["position"] = [100, 0, 100]
        self.obs = self.env.reset(ResetInfo(self.scene))
        self.__get_interaction_items()

        self.__hidden_items = []
        self.key_history = []

    def __get_interaction_items(self):
        self.interaction_items = {}
        self.first_interaction_items = {}
        for idx, instance in enumerate(self.scene["instances"]):
            item_type = instance.get("item_type", None)
            if item_type:
                self.interaction_items[instance["item_id"]] = idx
                self.first_interaction_items[idx] = False

    def game_shot(self, step, save_path=None, center_mark=True):
        if not save_path:
            __cache_dir = os.path.join(GAME_CACHE_DIR, "steps")
            if not os.path.exists(__cache_dir):
                os.makedirs(__cache_dir)
        else:
            __cache_dir = os.path.join(save_path)

        save_path = os.path.join(__cache_dir, f"{step}.png")

        save_image(self.obs.image, save_path, center_mark=center_mark)

        return save_path

    def step(self, action: Action = None):
        if action:
            self.obs = self.env.step(action)
        else:
            self.obs = self.env.step()

    def stop(self):
        self.env.close()

    def hide(self, id):
        api_calls = [HideObject(id)]
        self.obs = self.env.step(Action(api_calls=api_calls))
        self.interaction_items.pop(self.scene["instances"][id]["item_id"])

    def agent_grab_object_id(self):
        object_ids = []
        object_in_views = []
        for item_id, object_id in self.interaction_items.items():
            self.obs = self.env.step(Action(api_calls=[ObjectInView(object_id)]))
            if self.obs.api_returns["in_view"]:
                object_in_views.append(object_id)
                if (
                    distance(
                        vec_xz(
                            self.obs.game_states["instances"][object_id]["position"]
                        ),
                        vec_xz(self.obs.game_states["agent"]["position"]),
                    )
                    < MAX_INTERACTION_DISTANCE
                ):
                    object_ids.append(object_id)

        return object_ids, object_in_views

    def get_agent_state(self, reload=False):
        if not reload:
            return self.obs.game_states["agent"]
        ori_state = self.obs.game_states["agent"]
        agent_state = {"prefab": "", "scale": [1, 1, 1], "parent": -1, "type": ""}
        agent_state["position"] = [ori_state["position"][i] for i in ['x','y','z']]
        agent_state["rotation"] = [ori_state["rotation"][i] for i in ['x','y','z']]
        agent_state["forward"] = [ori_state["forward"][i] for i in ['x','y','z']]
        return agent_state


class Timer:
    def __init__(self):
        self._time = 0

    def add_time(self, delta):
        self._time += delta
        logger.debug(f"Timer updated: +{delta:.2f}s, total time: {self._time:.2f}s")

    def get_time(self):
        logger.debug(f"Timer retrieved: {self._time:.2f}s")
        return self._time

    def reset(self):
        self._time = 0

