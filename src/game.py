import os
import json
import re
import random
import traceback
from copy import deepcopy

import jsonschema
from jsonschema import validate

from legent import Action

from base_game import BaseGame
from legent_env import LegentGame, Timer
from audio_triggers import AudioTriggerMixin
from action_processor import ActionProcessor
from prompt_config import (
    PromptTemplate_Base, PromptTemplate_Hint,
)
from utils import format_scene, generate_texture
from config import (
    GAME_CACHE_DIR, TEXTURE_PATH, CAPTCHA_SIZE, CAPTCHA_FONT_SIZE,
    CAPTCHA_FONT_PATH, CAPTCHA_WALL_SIZE, EXIT_WALL,
)
from log_config import configure_logger, set_log_level

logger = configure_logger(__name__)

set_log_level("debug")


class Game(AudioTriggerMixin):
    def __init__(
        self,
        agent,
        scene_path: str,
        level_data: str,
        level: str,
        room_num: int = 1,
        max_retry=5,
        hint=False,
        record_path=None,
        scene_id=None,
        story_only=False,
        continue_game=False,
        suffix_level="",
        next_room_id=None,
        skip_story=False,
    ):
        self.agent = agent
        self.level_data = level_data
        self.scene_path = scene_path
        self.scene = format_scene(scene_path)
        if self.scene.get("_password", None):
            self.base_game = BaseGame(level_data, hint=hint, password=self.scene["_password"])
        else:
            self.base_game = BaseGame(level_data, hint=hint)
        self.__load_game()
        self.level = level
        self.scene_id = scene_id
        self.room_num = room_num
        self.next_room_id = next_room_id

        self.timer = Timer()

        self.record_path = record_path
        if record_path:
            logger.warning(f"Game start in record mode!")

        self.json_pattern = re.compile('```json(?P<jstr>[^(```)]+)```')
        self.max_retry = max_retry

        self.steps = -1
        self.__add_steps()

        if hint:
            self.Prompt = PromptTemplate_Hint
        else:
            self.Prompt = PromptTemplate_Base

        if scene_id is not None:
            if room_num > 1:
                if next_room_id:
                    level = f"{room_num}rooms-{level}-{scene_id}-{next_room_id}"
                else:
                    level = f"{room_num}rooms-{level}-{scene_id}"
            else:
                level = f"{level}-{scene_id}" if not suffix_level else f"{level}_{suffix_level}-{scene_id}"

        self.record_save_path = os.path.join(GAME_CACHE_DIR, level, self.agent.model+"_t_1")
        if not os.path.exists(self.record_save_path):
            os.makedirs(self.record_save_path)
        else:
            self.record_save_path = self.check_dirs(self.record_save_path)

        self.story_only = story_only
        self.continue_game = continue_game
        self.skip_story = skip_story

        self.trigger_info = {}
        self.__hidden_items = []
        self.__record_reload = False
        self.__revealed = False
        self.__wall_idx = None

        self.__has_ambient_sound = False
        self.__has_trigger_sound = []

        with open(level_data, 'r') as f:
            self.level_info = json.load(f)

        self._initialize_audio_trigger_system()
        self._sync_trigger_mode_to_items()
        self._trigger_cooldown_until_step = {}
        self._dual_source_trigger_history = []
        self._last_trigger_item_id = None
        self._last_audio_switch_event = None

        self.action_processor = ActionProcessor(self)

    def __load_game(self, tmp_agent=None, hide=False):
        self._fix_audio_paths()

        if tmp_agent:
            self.scene["agent"] = tmp_agent

        if self.agent.model.startswith("claude"):
            self.game = LegentGame(self.scene, camera_resolution_width=1960, camera_resolution_height=980)
        else:
            self.game = LegentGame(self.scene)

        if hide:
            for item_id in self.__hidden_items:
                self.game.hide(item_id)

    def check_dirs(self, path, i=50):
        _path, idx = path.split('_t_')
        idx = int(idx)
        for _i in range(idx, i):
            path = f"{_path}_t_{_i+1}"
            if os.path.exists(path):
                _path, idx = path.split('_t_')
                idx = int(idx)
            else:
                os.makedirs(path)
                return path
        print('exceed test times!')
        exit(1)

    def __verify_format(self, response):
        if response.get("interactions", None) is not None:
            if response["interactions"] == {}:
                response.pop("interactions")
            else:
                if response["interactions"].get("use_item_id", None) is None:
                    response["interactions"]["use_item_id"] = ""
                if response["interactions"].get("input", None) is None:
                    response["interactions"]["input"] = ""
        if any(value is None for value in response.values()):
            keys = list(response.keys())
            for key in keys:
                if response[key] is None:
                    response.pop(key)
        return response

    def __format_response(self, ori_response):
        try:
            ori_response = ori_response.replace(': False', ': false').replace(': True', ': true')
            ori_response = ori_response.strip('</Assistant>')
            ori_response = ori_response.split("</think>", 1)[-1]

            if not ori_response:
                logger.error(f"Step {self.steps}'s response is empty after preprocessing")
                return False

            json_str = None
            if "```json" in ori_response:
                json_response = self.json_pattern.search(ori_response)
                if json_response:
                    json_str = json_response.group('jstr').strip()
            elif ":\n\n" in ori_response:
                json_str = ori_response.split(':\n\n')[-1].strip()
            elif "<|assistant|>" in ori_response:
                json_str = ori_response.split('<|assistant|>')[-1].strip()
            else:
                cleaned = ori_response.strip('`').strip()
                if cleaned.startswith('json'):
                    cleaned = cleaned[4:].strip()
                json_str = cleaned

            if not json_str:
                json_str = ori_response

            first_brace = json_str.find('{')
            if first_brace != -1:
                json_str = json_str[first_brace:]
            else:
                logger.error(f"Step {self.steps}'s response has no JSON object (no '{{' found). Original: {repr(ori_response[:200])}")
                return False

            response = json.loads(json_str)
            response = self.__verify_format(response)
            validate(instance=response, schema=self.Prompt.INTERACTION_SCHEMA)
            logger.info(f"Step {self.steps}'s interaction is legal!")
            return response
        except jsonschema.exceptions.ValidationError as err:
            if err.message.strip().endswith("greater than the maximum of 180"):
                if response.get("rotate_right", 0) > 180:
                    _right = response["rotate_right"]
                    response.pop("rotate_right")
                    if response.get("rotate_left", None) is None:
                        response["rotate_left"] = _right - 180
                if response.get("rotate_left", 0) > 180:
                    _left = response["rotate_left"]
                    response.pop("rotate_left")
                    if response.get("rotate_right", None) is None:
                        response["rotate_right"] = _left - 180
                logger.debug(f" Fix bug for the response from {self.agent.model}: {err.message}")
                logger.info(f"Step {self.steps}'s interaction is corrected and now legal!")
                return response
            logger.error(f"Step {self.steps}'s move is illegal! for {err.message}")
            return False
        except Exception as e:
            logger.error(f"Step {self.steps}'s interaction occurs error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(ori_response)
            if '```json' in ori_response:
                json_response = self.json_pattern.search(ori_response)
                if json_response:
                    try:
                        response = json.loads(json_response.group('jstr').strip())
                    except Exception:
                        return False
                    validate(instance=response, schema=self.Prompt.INTERACTION_SCHEMA)
                    logger.info(f"Step {self.steps}'s interaction is legal!")
                    return response
            return False

    def __add_steps(self):
        self.steps += 1
        self.agent.step_meta_info.append({"key_step": False})

    def __generate_captcha(self, password):
        self.scene["walls"] = self.scene.get("walls", [])

        abs_texture_path = os.path.abspath(os.path.join(TEXTURE_PATH, f"{password}.png"))

        password = generate_texture(
            size=CAPTCHA_SIZE, font_size=CAPTCHA_FONT_SIZE,
            font_path=CAPTCHA_FONT_PATH,
            save_path=abs_texture_path,
            captcha_text=password
        )
        __walls = [(idx, instance) for idx, instance in enumerate(self.scene["instances"]) if "Wall" in instance.get("prefab", "") and EXIT_WALL not in instance.get("prefab", "")]
        __wall_with_captcha_idx = random.choice(__walls)[0]
        __wall_with_captcha = self.scene["instances"][__wall_with_captcha_idx]
        __ori_rotation = __wall_with_captcha["rotation"]
        if __ori_rotation[1] == 180:
            __rotation = [180, 0, 0]
        elif __ori_rotation[1] == 0:
            __rotation = [180, 180, 0]
        elif __ori_rotation[1] == 270:
            __rotation = [0, -90, 180]
        elif __ori_rotation[1] == 90:
            __rotation = [0, 90, 180]
        __captcha_wall = {
            "position": [
                __wall_with_captcha["position"][0],
                __wall_with_captcha["position"][1] + 0.2,
                __wall_with_captcha["position"][2]
            ],
            "rotation": __rotation,
            "size": CAPTCHA_WALL_SIZE,
            "material": abs_texture_path
        }
        self.scene["walls"].append(__captcha_wall)

        return len(self.scene["walls"]) - 1

    def get_action(self, response):
        return self.action_processor.process(response)

    def step(self, response):
        if not self.story_only:
            logger.debug("Taking actions ...")

        action, desc, obj_interact, obj_interact_fail = self.get_action(response)
        self.game.step(action)
        self.__add_steps()

        save_path = self.game.game_shot(self.steps, save_path=self.record_save_path)
        logger.info(f"{self.steps} moved and saved successfully!")

        return desc, save_path, obj_interact, obj_interact_fail

    def replace_base64_with_placeholder(self, text, placeholder="---image---"):
        if not isinstance(text, str):
            raise ValueError("text is not a string")
        pattern = r"(data:image\/[a-zA-Z]+;base64,)([A-Za-z0-9+/=]+)"
        replaced_text = re.sub(pattern, r"\1" + placeholder, text)
        return replaced_text

    def ask_for_action(self, desc, save_path, obj_interact, obj_interact_fail):
        if desc:
            if obj_interact:
                if not obj_interact_fail:
                    print_desc = self.replace_base64_with_placeholder(desc)
                    logger.debug(f"The agent get response: {print_desc}")
                    interaction_desc = f"After the last step of interaction, you find:\n{desc}"
                else:
                    logger.debug(f"obj_interact_fail: {desc}")
                    interaction_desc = f"{desc}"
            else:
                interaction_desc = f"{desc} You did not interact with any objects in the last step."
        else:
            interaction_desc = (
                "The last time your action environment was not responsive."
            )

        bag_desc = (
            self.base_game.bag_desc
            if self.base_game.bag_desc
            else "Nothing in your bag."
        )

        step_prompt = self.Prompt.STEP_PROMPT.format(
            interaction_result=interaction_desc, bag_desc=bag_desc
        )

        audio_info = self.get_current_audio_info()

        self.__has_ambient_sound = audio_info["ambient_sounds"]
        self.__has_trigger_sound = audio_info["trigger_sounds"]

        self.agent.add_problem(step_prompt, save_path, audio_info=audio_info)

        retry = 0
        while retry < self.max_retry:
            ori_response = self.agent.ask()
            if ori_response is not None:
                response = self.__format_response(ori_response)
            else:
                response = None

            if response:
                return response, step_prompt
            else:
                print(ori_response)
                retry += 1

        return {}, step_prompt

    def read_note(self):
        return self.agent.notes

    def story_recovery(self):
        logger.info("Start story recovery.")
        desc = self.Prompt.story_prompt

        bag_desc = (
            self.base_game.bag_desc
            if self.base_game.bag_desc
            else "There is notiong in your bag."
        )
        recovery_prompt = desc + bag_desc
        self.agent.add_problem(recovery_prompt)
        story = self.agent.ask()
        if story:
            logger.info("Story recovered successfully.")
            return story
        else:
            logger.error("Story recovered failed.")
            return ""

    def check_new_room_desc(self, desc, escaped_rooms, room_left_to_escape):
        if self.base_game.clear and room_left_to_escape > 1:
            escaped_rooms += 1
            if escaped_rooms == 1:
                escaped_rooms_str = "1st"
            elif escaped_rooms == 2:
                escaped_rooms_str = "2nd"
            elif escaped_rooms == 3:
                escaped_rooms_str = "3rd"
            else:
                escaped_rooms_str = f"{escaped_rooms_str}th"
            desc = f"You have successfully escaped from the {escaped_rooms} room. You are now entering the next room. The initial scene in the new room is shown in the picture."
            return desc
        return desc

    def main(self):
        room_left_to_escape, escaped_rooms = self.room_num, 0

        logger.info(f"Start playing the game. There are {room_left_to_escape} rooms.")

        results = []

        save_path = self.game.game_shot(self.steps, save_path=self.record_save_path)
        desc = "The initial scene is shown in the picture."
        obj_interact, obj_interact_fail = False, False

        if self.record_path:
            record_steps = json.load(open(self.record_path, "r", encoding="utf-8"))
            if self.story_only:
                self.agent.interactions = deepcopy(self.agent.system_messages)

        # for multi-room
        level_data_list = []
        scene_path_list = []
        if room_left_to_escape > 1:
            if self.next_room_id:
                level_data_list = [self.level_data, self.level_data.replace(f"1_1.json", f"1_{self.next_room_id}.json")]
                scene_path_list = [self.scene_path, self.scene_path.replace(f"1_1.json", f"1_{self.next_room_id}.json")]
            else:
                for i in range(2, room_left_to_escape + 1):
                    new_level_data = re.sub(r"(\d+)(?=\.json$)", str(i), self.level_data)
                    new_scene_path = re.sub(r"(\d+)(?=\.json$)", str(i), self.scene_path)
                    level_data_list.append(new_level_data)
                    scene_path_list.append(new_scene_path)

        # Force-exit thresholds by level family
        if self.level.startswith("level1"):
            max_allowed_steps = 50
        elif self.level.startswith("level2"):
            max_allowed_steps = 65
        elif self.level.startswith("level3"):
            max_allowed_steps = 80
        else:
            max_allowed_steps = 80

        grab_tp = 0

        while not self.base_game.clear:
            if self.record_path:
                if self.steps >= len(record_steps):
                    if self.continue_game:
                        self.record_path = None
                    else:
                        break

                if self.record_path:
                    response = record_steps[self.steps]["response"]
                    desc, save_path, obj_interact, obj_interact_fail = self.step(response)
                    self.agent.add_problem(record_steps[self.steps].get("desc", ""))
                    self.agent.add_response(json.dumps(response))

                    if self.__record_reload:
                        self.__record_reload = False

                    if self.base_game.clear and room_left_to_escape > 1:
                        self.base_game.clear = False
                        room_left_to_escape -= 1
                        tmp_bag = self.base_game.bag
                        self.base_game.clear = False
                        self.game.stop()

                        scene_path = scene_path_list.pop(0)
                        level_data = level_data_list.pop(0)
                        logger.warning(f"In a new scene: {scene_path}\nnew level: {level_data}")
                        self.scene = json.load(open(scene_path))
                        self.__load_game()
                        self.base_game.bag = tmp_bag

                    if self.steps >= len(record_steps):
                        if self.continue_game:
                            self.record_path = None
                        else:
                            break
            else:
                bag_len = len(self.base_game.bag_desc)
                response, step_prompt = self.ask_for_action(desc, save_path, obj_interact, obj_interact_fail)
                self.agent.step_meta_info[-1]['step_prompt'] = step_prompt
                self.agent.step_meta_info[-1]['response'] = response

                self.agent.add_response(json.dumps(response))
                if len(self.base_game.bag_desc) > bag_len:
                    grab_tp += 1
                results.append(
                    {
                        "step": self.steps,
                        "desc": self.replace_base64_with_placeholder(step_prompt),
                        "save_path": save_path,
                        "response": response,
                        "bag": self.base_game.bag_desc,
                        "used_history": len(self.agent.interactions) // 2,
                        "grab_tp": grab_tp,
                        "__record_reload": self.__record_reload,
                        "timer": self.timer.get_time(),
                        "ambient_sound": self.__has_ambient_sound,
                        "trigger_sound": self.__has_trigger_sound,
                        "audio_switch_event": self._last_audio_switch_event,
                        "model_request": self.agent.get_last_sent_message_snapshot(),
                    }
                )
                self._last_audio_switch_event = None
                if self.__wall_idx is not None:
                    results[-1]["__wall_idx"] = self.__wall_idx
                if response == {}:
                    logger.info("Retry failed. Proceed to story recovery.")
                    with open(os.path.join(self.record_save_path, "records.json"), "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                    break

                desc, save_path, obj_interact, obj_interact_fail = self.step(response)

                self._check_audio_triggers()
                self._update_ambient_sounds()

                desc = self.check_new_room_desc(desc, escaped_rooms, room_left_to_escape)

                with open(os.path.join(self.record_save_path, "records.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                if self.steps > max_allowed_steps:
                    logger.info(f'\n\n{self.steps} steps, force exit!!!\n\n')
                    break

                if self.base_game.clear:
                    if room_left_to_escape > 1:
                        room_left_to_escape -= 1
                        tmp_bag = self.base_game.bag
                        self.base_game.clear = False
                        self.game.stop()

                        scene_path = scene_path_list.pop(0)
                        level_data = level_data_list.pop(0)
                        logger.warning(f"In a new scene: {scene_path}\nnew level: {level_data}")
                        self.scene = json.load(open(scene_path))
                        self.__load_game()
                        self.base_game.bag = tmp_bag

                    else:
                        break

        if not self.record_path:
            if self.base_game.clear:
                results.append({'info': f"Game stop at step {self.steps}. Escaped succesfully!"})
            else:
                results.append({'info': f"Game stop at step {self.steps}. Force exit!"})
            with open(os.path.join(self.record_save_path, "records.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        self.base_game.audio_manager.stop_all_sounds()

        self.game.stop()
        if not self.skip_story and ((not self.record_path) or self.story_only or self.continue_game):
            story = self.story_recovery()
            print(story)
            step_wise_note = self.read_note()
            print(step_wise_note)
            story = {
                "story": story,
                "step_wise_note": step_wise_note,
            }
            with open(os.path.join(self.record_save_path, "story.json"), "w", encoding="utf-8") as f:
                json.dump(story, f, ensure_ascii=False, indent=4)

