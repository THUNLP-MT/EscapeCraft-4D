import math
import time
import traceback
from copy import deepcopy

import numpy as np

from legent import Action
from legent.utils.math import vec_xz

from config import FORWARD_SPEED, ROTATION_SPEED, GRAB_TIME
from audio_triggers import (
    INTENT_THRESHOLD, DWELL_TIME_FOR_MAX_SCORE,
    TRIGGER_SWITCH_WINDOW_STEPS, TRIGGER_SWITCH_MIN_ATTEMPTS,
)
from log_config import configure_logger

logger = configure_logger(__name__)


class ActionProcessor:
    """
    Processes a parsed JSON response from the agent into a Legent Action
    plus descriptive text about what happened.
    """

    def __init__(self, game):
        self.game = game

    # ------------------------------------------------------------------
    # Helper methods (formerly nested functions inside Game.get_action)
    # ------------------------------------------------------------------

    def _get_trigger_item_ids(self):
        return {
            s.get("item_id")
            for s in self.game.sound_objects
            if (not s.get("is_ambient")) and s.get("item_id")
        }

    def _get_currently_playing_trigger_item_ids(self):
        playing = set()
        trigger_item_ids = self._get_trigger_item_ids()
        with self.game.base_game.audio_manager.lock:
            for _item_id, state in self.game.base_game.audio_manager.sound_states.items():
                if _item_id in trigger_item_ids and state.get("is_playing", False):
                    if state.get("sound_type", "trigger") != "ambient":
                        playing.add(_item_id)
        return playing

    def _stop_all_trigger_sounds(self):
        stopped = []
        for _item_id in self._get_currently_playing_trigger_item_ids():
            self.game.base_game.audio_manager.stop_sound(_item_id)
            stopped.append(_item_id)
        return stopped

    def _play_exclusive_trigger_sound(self, item_id, sound_file, loop=True):
        stopped_ids = self._stop_all_trigger_sounds()
        self.game.base_game.audio_manager.play_sound(item_id, sound_file, loop=loop)
        self.game._last_trigger_item_id = item_id
        return stopped_ids

    def _find_trigger_candidates_in_view(self):
        try:
            _, object_in_views = self.game.game.agent_grab_object_id()
        except Exception:
            return []

        if not object_in_views:
            return []

        by_instance = {
            s["instance_id"]: s
            for s in self.game.sound_objects
            if not s.get("is_ambient") and s.get("trigger_mode") == "trigger"
        }
        agent_pos = np.array(vec_xz(self.game.game.obs.game_states["agent"]["position"]))

        candidates = []
        for instance_id in object_in_views:
            s_obj = by_instance.get(instance_id)
            if not s_obj:
                continue
            obj_pos = np.array(vec_xz(self.game.game.obs.game_states["instances"][instance_id]["position"]))
            dist = float(np.linalg.norm(agent_pos - obj_pos))
            candidates.append((s_obj, instance_id, dist))

        candidates.sort(key=lambda x: x[2])
        return candidates

    def _find_best_trigger_target(self):
        candidates = self._find_trigger_candidates_in_view()
        if not candidates:
            return None, None, None
        return candidates[0]

    def _compute_angle_ok(self, instance_id, sound_obj, agent_pos):
        obj_instance = self.game.game.obs.game_states["instances"][instance_id]
        obj_pos = np.array(vec_xz(obj_instance["position"]))

        if isinstance(obj_instance.get("rotation"), list) and len(obj_instance["rotation"]) > 1:
            rotation_y = obj_instance["rotation"][1]
        elif isinstance(obj_instance.get("rotation"), dict) and "y" in obj_instance["rotation"]:
            rotation_y = obj_instance["rotation"]["y"]
        else:
            rotation_y = 0

        obj_forward = self.game._get_forward_vector(rotation_y)
        vector_to_agent = agent_pos - obj_pos
        if np.linalg.norm(vector_to_agent) <= 1e-6:
            return False
        norm_vector_to_agent = vector_to_agent / np.linalg.norm(vector_to_agent)
        dot_product = float(np.dot(obj_forward, norm_vector_to_agent))
        angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
        return angle < sound_obj.get("interaction_angle", 120.0) / 2

    def _handle_trigger_action(self, ctx):
        """
        Implements explicit agent trigger intent.
        ctx is a dict with keys 'desc' and 'obj_interact' that get mutated.
        """
        g = self.game
        ctx['obj_interact'] = True
        g._last_audio_switch_event = None

        candidates = self._find_trigger_candidates_in_view()
        if not candidates:
            ctx['desc'] += (
                "Audio trigger result: FAILED (no trigger sound object in view). "
                "Action required: rotate/look around to find a sound-emitting object (recorder/radiogram/musicbox), "
                "center it, move closer, then trigger again.\n"
            )
            return
        target, instance_id, dist = candidates[0]
        target_item_id = target.get("item_id")

        is_dual_zone = len(candidates) >= 2
        if is_dual_zone:
            g._dual_source_trigger_history = [
                h for h in g._dual_source_trigger_history
                if g.steps - h["step"] <= TRIGGER_SWITCH_WINDOW_STEPS
            ]
            g._dual_source_trigger_history.append(
                {"step": g.steps, "item_id": target_item_id}
            )
        else:
            g._dual_source_trigger_history = []

        same_item_attempts = sum(
            1
            for h in g._dual_source_trigger_history
            if h.get("item_id") == target_item_id
        )

        force_switch = (
            is_dual_zone and
            target_item_id is not None and
            same_item_attempts >= TRIGGER_SWITCH_MIN_ATTEMPTS
        )

        if force_switch:
            currently_playing = self._get_currently_playing_trigger_item_ids()
            if currently_playing:
                from_item = next(iter(currently_playing))
            elif g._last_trigger_item_id:
                from_item = g._last_trigger_item_id
            else:
                from_item = target.get("item_id")

            for cand_target, cand_instance_id, cand_dist in candidates:
                cand_item_id = cand_target.get("item_id")
                if cand_item_id and cand_item_id != from_item:
                    target, instance_id, dist = cand_target, cand_instance_id, cand_dist
                    g._last_audio_switch_event = {
                        "type": "forced_trigger_switch",
                        "reason": "repeated_trigger_same_source_near_dual_sources",
                        "step": g.steps,
                        "from_item_id": from_item,
                        "to_item_id": cand_item_id,
                        "window_steps": TRIGGER_SWITCH_WINDOW_STEPS,
                        "attempts_on_source_in_window": same_item_attempts,
                    }
                    logger.info(
                        f"[AUDIO-TRIGGER] Forced switch: {from_item} -> {cand_item_id} "
                        f"(same-source attempts={same_item_attempts} in {TRIGGER_SWITCH_WINDOW_STEPS} steps)"
                    )
                    g._dual_source_trigger_history = []
                    break

        cooldown_step = g._trigger_cooldown_until_step.get(target_item_id, -1)
        if g.steps < cooldown_step:
            ctx['desc'] += (
                f"Audio trigger result: COOLDOWN (wait {cooldown_step - g.steps} more step(s) "
                f"before re-triggering {target_item_id}).\n"
            )
            return

        limit = float(target.get("distance_limit", target.get("interaction_distance", 5.0)))
        agent_pos = np.array(vec_xz(g.game.obs.game_states["agent"]["position"]))
        angle_ok = self._compute_angle_ok(instance_id, target, agent_pos)

        item_id = target.get("item_id")
        tracker = g.intent_tracker.get(instance_id, None)
        dwell_duration = 0.0
        is_moving_towards = False
        if tracker:
            is_in_zone_now = (dist < limit) and angle_ok
            if is_in_zone_now and tracker.get("time_entered_zone", 0) > 0:
                dwell_duration = max(0.0, time.time() - tracker["time_entered_zone"])
            is_moving_towards = dist < float(tracker.get("last_distance", float("inf")))

        intent_score = 0.0
        if dist < limit and angle_ok:
            distance_score = 1.0 - (dist / limit if limit else 1.0)
            dwell_score = min(1.0, dwell_duration / DWELL_TIME_FOR_MAX_SCORE)
            move_score = 0.6 if is_moving_towards else 0.3
            intent_score = (distance_score * 0.5) + (dwell_score * 0.3) + (move_score * 0.2)

        if intent_score > INTENT_THRESHOLD:
            logger.info(
                f"[AUDIO-TRIGGER] Explicit trigger SUCCESS for {item_id}: "
                f"dist={dist:.2f} limit={limit} angle_ok={angle_ok} intent={intent_score:.2f}"
            )
            stopped_ids = self._play_exclusive_trigger_sound(item_id, target.get("sound_file", ""))
            ctx['desc'] += (
                f"Audio trigger result: SUCCESS ({item_id} is playing). "
                f"(dist={dist:.2f}m, limit={limit}m, intent_score={intent_score:.2f} > {INTENT_THRESHOLD})\n"
            )
            if g._last_audio_switch_event:
                ctx['desc'] += (
                    f"Audio switch applied: from {g._last_audio_switch_event.get('from_item_id')} "
                    f"to {g._last_audio_switch_event.get('to_item_id')} due to frequent trigger attempts nearby. "
                    f"Stopped trigger sounds: {stopped_ids}.\n"
                )
        else:
            g._trigger_cooldown_until_step[item_id] = g.steps + 1
            reason = []
            if dist >= limit:
                reason.append(f"too far: {dist:.2f}m > {limit}m")
            if not angle_ok:
                reason.append("not facing the object (angle not ok)")
            if dist < limit and angle_ok:
                reason.append(f"intent_score={intent_score:.2f} <= {INTENT_THRESHOLD} (try dwell ~2s)")
            reason_str = "; ".join(reason) if reason else "unknown"
            ctx['desc'] += (
                f"Audio trigger result: FAILED ({reason_str}). "
                "Action required: move closer and face the object; stay for ~2 seconds, then trigger again.\n"
            )

    # ------------------------------------------------------------------
    # Main process method
    # ------------------------------------------------------------------

    def process(self, response):
        """
        Main entry point. Translates agent response dict into
        (Action, desc, obj_interact, obj_interact_fail).
        """
        g = self.game

        action_list = {}
        ctx = {
            'desc': "",
            'obj_interact': False,
            'obj_interact_fail': True if "grab" in response or "read" in response else False,
        }

        reload_needed = False
        last_expired_item_id = None
        for item_id, _trigger_info in g.trigger_info.items():
            if _trigger_info["type"] == "time_limit":
                elapsed_time = g.timer.get_time() - _trigger_info["start_time"]
                if elapsed_time > _trigger_info["limit"]:
                    reload_needed = True
                    last_expired_item_id = item_id
                    wall_idx = _trigger_info["fade"]
                    g.scene["walls"].pop(wall_idx)
                    logger.info(f"Captcha wall at index {wall_idx} removed after time limit for item {item_id}.")
                    g._Game__revealed = False
                    g._Game__record_reload = False
                    g._Game__wall_idx = None

        if reload_needed:
            tmp_bag = deepcopy(g.base_game.bag)
            tmp_agent = g.game.get_agent_state(reload=True)
            g.game.stop()
            g._Game__load_game(tmp_agent=tmp_agent, hide=True)
            g.base_game.bag = tmp_bag
            g.trigger_info.pop(last_expired_item_id)

        for key in response:
            try:
                if not response[key]:
                    continue
                if key == "trigger":
                    if response.get("trigger") is True:
                        self._handle_trigger_action(ctx)
                    continue
                if key in ["move_forward", "move_right", "rotate_right", "rotate_down", "jump"]:
                    if key == "move_forward":
                        action_list["use_teleport"] = True
                        action_list["teleport_forward"] = float(response[key])
                        g.timer.add_time(abs(float(response[key]) / FORWARD_SPEED))
                    elif key == "jump":
                        action_list[key] == bool(response[key])
                    else:
                        action_list[key] = float(response[key])
                    if key == "rotate_right":
                        g.timer.add_time(abs(float(response[key]) / ROTATION_SPEED))
                    if "Successfully moved." not in ctx['desc']:
                        ctx['desc'] += "Successfully moved."

                elif key == "look_at":
                    if response[key][0] == 0.5 and response[key][1] == 0.5:
                        continue
                    action_list["look_x"], action_list["look_y"] = response[key]
                    action_list["use_look_at"] = True
                    if "View moved." not in ctx['desc']:
                        ctx['desc'] += "View moved."

                elif key == "grab":
                    if response[key]:
                        g.timer.add_time(GRAB_TIME)
                        ctx['obj_interact'] = True
                        logger.warning("The agent try to grab.")
                        logger.debug(f"Rationale of actions: {response['rationale']}")
                        object_ids, object_in_views = g.game.agent_grab_object_id()
                        if object_ids:
                            _desc = None
                            for id, object_id in enumerate(object_ids):
                                _desc, get_item = g.base_game.interaction(
                                    g.scene["instances"][object_id]["item_id"],
                                    **response.get("interactions", {}),
                                )

                                if get_item:
                                    g.game.hide(object_id)
                                    g._Game__hidden_items.append(object_id)
                                    get_item_id = g.scene["instances"][object_id]["item_id"]

                                    # Check grab-mode audio objects
                                    for sound_obj in g.sound_objects:
                                        if (sound_obj.get("item_id") == get_item_id and
                                            sound_obj.get("trigger_mode") == "grab" and
                                            not sound_obj.get("is_ambient")):
                                            logger.info(
                                                f"[AUDIO-TRIGGER] Grab trigger SUCCESS for {get_item_id}: "
                                                f"trigger_mode=grab"
                                            )
                                            self._play_exclusive_trigger_sound(
                                                get_item_id,
                                                sound_obj.get("sound_file", ""),
                                                loop=False
                                            )
                                            ctx['desc'] += (
                                                f"Audio triggered by grab: {get_item_id} is now playing.\n"
                                            )
                                            break

                                    for _item in g.level_info["room"]["items"]:
                                        if _item["id"] == get_item_id:
                                            break
                                    if _item.get("change_trigger", None):
                                        if _item["change_trigger"]["type"] == "reveal":
                                            logger.warning("Revealing new items triggered by interaction.")
                                            g.base_game.clear = False
                                            tmp_bag = deepcopy(g.base_game.bag)
                                            tmp_agent = g.game.get_agent_state(reload=True)
                                            g.game.stop()
                                            target_item_id = _item["change_trigger"]["target_item_id"]
                                            target_item_info = [item for item in g.level_info["room"]["items"] if item["id"] == target_item_id][0]
                                            if _item["change_trigger"]["target_item_type"] == "password":
                                                password = target_item_info["value"]
                                                wall_idx = g._Game__generate_captcha(password)
                                                g.trigger_info[target_item_id] = {
                                                    "type": "time_limit",
                                                    "limit": target_item_info["on_trigger"]["limit_setting"]["limit_seconds"],
                                                    "start_time": g.timer.get_time(),
                                                    "trigger_times": target_item_info["on_trigger"]["trigger_times"],
                                                    "fade": wall_idx
                                                }
                                                logger.debug(f"Captcha wall for password '{password}' generated at wall index {wall_idx}.")

                                                g._Game__load_game(tmp_agent=tmp_agent, hide=True)
                                                g.base_game.bag = tmp_bag
                                                g._Game__revealed = True
                                                g._Game__record_reload = True
                                                g._Game__wall_idx = wall_idx

                                if not g.game.first_interaction_items[object_id]:
                                    g.agent.step_meta_info[-1]['key_step'] = True
                                    g.game.first_interaction_items[object_id] = True

                                ctx['desc'] += f"Interaction triggered {id+1} returns information: {_desc}\n"
                                ctx['obj_interact_fail'] = False
                        elif object_in_views:
                            ctx['desc'] += "You try to interact with some object in the scene, but there seems no response. Please try stepping closer towards the object. If you already step closer but find the object not interactable, you should explore elsewhere in the room."
                        else:
                            ctx['desc'] += "There is no interactable objects in the scene or you are too far away from your target, and your interactive action got no responses."

                elif key == "read":
                    desc_r, get_item = g.base_game.interaction(
                        response[key], read=True
                    )
                    ctx['obj_interact'] = True
                    if desc_r:
                        ctx['obj_interact_fail'] = False
                    ctx['desc'] += desc_r if desc_r else ""

                elif key == "rationale":
                    action_list["text"] = response[key]

            except Exception as e:
                logger.warning(
                    f"There exits an error: {str(e)} while processing the key {key}. For the sake of continuity, we will skip this key."
                )
                print(traceback.format_exc())
                continue

        if g._Game__revealed:
            ctx['desc'] += "If you see the password mentioned in the hint, please explicitly state 'I found the password mentioned before' in the rationale."

        if not g.story_only:
            print('===>', response)
            print('===>', action_list)
            print('===>', response.get('rationale', None))
        return Action(**action_list), ctx['desc'], ctx['obj_interact'], ctx['obj_interact_fail']

