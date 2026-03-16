import json
import os
import random
import io
import math
import numpy as np
from PIL import Image
import base64

from config import MAX_INTERACTION_DISTANCE
from audio_manager import AudioTriggerManager
from log_config import configure_logger, set_log_level

logger = configure_logger(__name__)


class CombinationLock:
    """Represents a combination lock with a password."""
    def __init__(self, id, **kwargs):
        self.id = id
        password = kwargs.get("password", None)
        self.__password = self.__assign_password(**kwargs) if not password else password

    @property
    def password(self):
        return self.__password

    def __call__(self, user_input):
        return self.check_password(user_input)

    def check_password(self, user_input):
        """Checks if the user input matches the lock's password."""
        if str(user_input) == self.__password:
            return True
        return False

    def __assign_password(self, **kwargs):
        """Generates a random numerical password."""
        length = kwargs.get("length", 4)
        password = "".join([str(random.randint(0, 9)) for _ in range(length)])
        return password

# Backward-compatible alias
ConbinationLock = CombinationLock


class BaseGame:
    """
    Core game logic for an escape room style game.
    Manages game state, items, interactions, and puzzle logic.
    """
    def __init__(self, level_data, hint=False, **kwargs):
        with open(level_data, "r", encoding="utf-8") as f:
            self.__ori_data = json.load(f)

        for name, value in kwargs.items():
            setattr(self, name, value)

        logger.info(f"Loading {level_data} as the base game.")

        self.__add_description()

        self.items = self.__format_items()
        self.__format_link()
        self.__assign_password()
        self.puzzle_images = {}

        self.bag = Bag()
        self.clear = False

        self.hint = hint
        self.audio_manager = AudioTriggerManager()

    @property
    def bag_desc(self):
        return self.bag.get_bag_desc()

    def __add_description(self):
        """Adds a default description to items based on their type."""
        for item in self.__ori_data["room"]["items"]:
            if item["type"] == "box":
                __description = "A box "
                if item["unlock_method"]:
                    __description += f"seems that can be open with a {item['unlock_method']['type']}."
                    item["locked"] = True
                else:
                    __description += "unlocked."
                    item["locked"] = False

            elif item["type"] == "key":
                __description = "A key seems to open something."

            elif item["type"] == "musicbox":
                __description = "A delicate music box. It might play a sound if you get close to it."

            else:
                __description = None

            if __description:
                item["description"] = __description

    def __format_items(self):
        """Converts the list of items from JSON into a dictionary indexed by item ID."""
        result = {}
        for item in self.__ori_data["room"]["items"]:
            result[item["id"]] = item
        return result

    def __format_link(self):
        """Establishes relationships between items (e.g., contents of a box)."""
        for id, item in self.items.items():
            if item["type"] == "box":
                for content in item["contents"]:
                    self.items[content]["putted_in"] = id
            elif item["type"] == "paper":
                for content in item["contents"]:
                    if isinstance(content, str):
                        self.items[content]["carried_on"] = id
                    elif isinstance(content, dict) and content["type"] == "image":
                        self.items[content["password_id"]]["carried_on"] = id

    def __assign_password(self):
        """Assigns a password function to items that are password-protected."""
        for item_id, item in self.items.items():
            if item.get("type") != "password":
                continue

            # Priority: per-item value -> global password -> random fallback
            fixed_password = item.get("value")
            global_password = getattr(self, "password", None)

            if fixed_password is not None:
                assigned_password = str(fixed_password)
            elif global_password is not None:
                assigned_password = str(global_password)
            elif item.get("show"):
                raise ValueError(
                    f"Password item {item_id} is visible but has no fixed password assigned!"
                )
            else:
                assigned_password = None

            if assigned_password is not None:
                lock = CombinationLock(item_id, password=assigned_password)
            else:
                lock = CombinationLock(item_id)
                assigned_password = lock.password

            item["check_func"] = lock

            carried_on = item.get("carried_on")
            if carried_on:
                self.items[carried_on]["password"] = assigned_password

    @property
    def ori_data(self):
        return self.__ori_data

    def open_box(self, box_id):
        """Handles the logic for opening a box and adding its contents to the player's bag."""
        if "box" not in box_id:
            raise ValueError(f"{box_id} is not a box!")

        self.items[box_id]["locked"] = False
        desc = "By opening the box, you got:\n"

        for content in self.items[box_id]["contents"]:
            item_got = self.items[content]
            self.bag.add_item(content, item_got)
            desc += self.bag.get_item_desc(content)
        return desc

    def interaction(self, item_id, **kwargs):
        """
        Processes all player interactions with items in the game.
        This is the main entry point for game actions.
        """
        desc = ""
        get_item = False

        if (not item_id in self.items) and (not item_id == "exit"):
            raise ValueError(f"The game don't have the item {item_id}!")

        logger.warning(
            f"The agent is interacting with objects in the scene. The interactable items: {item_id}"
        )

        user_input = kwargs.get("input", None)
        use_item_id = kwargs.get("use_item_id", None)
        read: bool = kwargs.get("read", False)
        if not self.bag.check_item(use_item_id) and use_item_id:
            desc += f"You don't have the item {use_item_id} in your bag, please try exploring further in the room!",

        if item_id == "entrance":
            desc += f"This is where you start the game. You can explore the room by looking around and interacting with items."

        elif item_id == "exit":
            unlock = self.__ori_data["room"]["exit"]
            desc += f"This door seems to be the exit to get out of here, and can be open with a {unlock['type']}. "

            if unlock["type"] == "password":
                if user_input:
                    user_input = str(user_input)
                    user_input = "".join(user_input.split("-"))
                    if self.items[unlock["unlock_item_id"]]["check_func"](user_input):
                        self.clear = True
                        desc = "You have used the correct password to unlock the door."
                        logger.critical(
                            f"The agent is using the correct password to unlock the door. The game will be cleared ..."
                        )
                    else:
                        desc = f"You use the password {user_input}, but it seems that this password is wrong!"

            elif unlock["type"] in ["key"]:
                if use_item_id == unlock["unlock_item_id"]:
                    desc = f"You have used the item {use_item_id} to unlock the door successfully."
                    logger.critical(
                        f"The agent is using the correct key to unlock the door. The game will be cleared ..."
                    )
                    self.clear = True
                elif use_item_id:
                    desc = f"You use the item {use_item_id}, but it seems that this item is not the correct key!"
                else:
                    desc += f"You need to find the key to unlock the door."

        elif self.items[item_id]["type"] == "box":
            if read:
                desc = f"You can't read a box. Please try to interact with it by setting grab=True."
            elif self.items[item_id]["locked"]:
                unlock_method = self.items[item_id]["unlock_method"]
                if unlock_method["type"] == "key":
                    if use_item_id == unlock_method["item_id"]:
                        desc = self.open_box(item_id)
                        get_item = True
                    elif use_item_id:
                        desc = f"You use the item {use_item_id}, but it seems that this item is not the correct key to open the box!"
                    else:
                        desc = f"The box is locked. You need to find the key to open it."
                elif unlock_method["type"] == "password":
                    if user_input:
                        user_input = str(user_input)
                        user_input = "".join(user_input.split("-"))
                        if self.items[unlock_method["item_id"]]["check_func"](user_input):
                            desc = self.open_box(item_id)
                            get_item = True
                        else:
                            desc = f"You use the password {user_input}, but it seems that this password is wrong!"
                    else:
                        desc = f"The box is locked. You need to find the password to open it."
            else:
                desc = self.open_box(item_id)
                get_item = True

        elif self.items[item_id]["type"] == "paper":
            if read:
                desc = self.bag.get_item_desc(item_id)
            else:
                desc = f"- item_id: {item_id}, item: paper, description: A paper "
                for idx, content in enumerate(self.items[item_id]["contents"]):
                    if isinstance(content, dict):
                        if content["type"] == "story":
                            desc += f"The {idx+1} part of the paper records a story: {content['content']}\n"
                        elif content["type"] == "image":
                            path = content["image_path"]
                            image = Image.open(path)
                            buffered = io.BytesIO()
                            image.save(buffered, format="JPEG")
                            base64_image = base64.b64encode(
                                buffered.getvalue()
                            ).decode("utf-8")
                            self.puzzle_images[item_id] = content["image_path"]
                            desc += f"The {idx+1} part of the paper records an image with {content['content']} attached: <img src='data:image/jpeg;base64,{base64_image}'></img>\n"

                    elif isinstance(content, str):
                        desc += f"The {str(idx+1)} part of the paper records a string of numbers {str(self.items[item_id]['password'])}.\n"

        else:
            if self.items[item_id]["show"]:
                # Sound-emitting devices are treated as fixed scene devices (not collectible).
                # They should be triggered by proximity/angle (or explicit trigger action), not put into the bag.
                if self.items[item_id].get("type") in {"recorder", "radiogram", "musicbox", "tv"}:
                    trigger_mode = self.items[item_id].get("trigger_mode", "trigger")

                    if trigger_mode == "grab":
                        self.bag.add_item(item_id, self.items[item_id])
                        desc = f"You got: {self.bag.get_item_desc(item_id)}"
                        get_item = True
                    else:
                        desc = (
                            f"This {self.items[item_id].get('type')} is a fixed device and cannot be picked up. "
                            "To get clues from it, move closer, face it, and use trigger=true to play its sound."
                        )
                        get_item = False
                        if "change_trigger" in self.items[item_id]:
                            get_item = True
                else:
                    self.bag.add_item(item_id, self.items[item_id])
                    desc = f"You got: {self.bag.get_item_desc(item_id)}"
                    get_item = True

        return desc, get_item

    def __call__(self, item_id, **kwargs):
        return self.interaction(item_id, **kwargs)


class Bag:
    """A simple inventory system to hold items collected by the player."""
    def __init__(self):
        self.items = {}

    def add_item(self, id, item):
        """Adds an item to the bag."""
        logger.item(f"Agent get the item {id}")
        self.items[id] = item

    def check_item(self, id):
        """Checks if an item exists in the bag."""
        return self.items.get(id, False)

    def get_item_desc(self, id):
        """Returns a formatted description of a specific item in the bag."""
        if not self.check_item(id):
            return f"You can't get the information of item {id} because you hasn't collect it. Please try exploring further in the room!"

        item = self.items[id]
        desc = ""
        if item.get("description", False):
            desc += f"- item_id: {id}, item: {item['type']}, description: {item['description']}\n"

        elif item["type"] == "paper":
            desc += f"- id: {id}, item: paper, description: A paper "
            for content in item["contents"]:
                if isinstance(content, str):
                    desc += f"written a string of numbers {item['password']}.\n"
                else:
                    try:
                        desc += f"written a {content['type']}.\n"
                    except Exception:
                        logger.error(f"Unexpected content format in paper item {id}: {content}")
                        continue

        else:
            raise NotImplementedError

        return desc

    def get_bag_desc(self):
        """Returns a formatted description of all items currently in the bag."""
        desc = ""
        for id, item in self.items.items():
            desc += self.get_item_desc(id)
        return desc

