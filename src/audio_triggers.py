import os
import time
import math
import numpy as np

from legent.utils.math import vec_xz

from config import ASSETS_DIR
from log_config import configure_logger

logger = configure_logger(__name__)

INTENT_THRESHOLD = 0.6
DWELL_TIME_FOR_MAX_SCORE = 2.0
AGENT_HISTORY_LENGTH = 10
TRIGGER_SWITCH_WINDOW_STEPS = 5
TRIGGER_SWITCH_MIN_ATTEMPTS = 3


class AudioTriggerMixin:
    """
    Mixin that provides audio trigger system functionality for the Game class.
    Expects the host class to have: self.scene, self.game, self.base_game,
    self.level_info, self.sound_objects, self.intent_tracker, self.agent_pos_history.
    """

    def _initialize_audio_trigger_system(self):
        """Scans the scene for objects with audio files and initializes their intent trackers."""
        self.sound_objects = []
        self.intent_tracker = {}
        self.agent_pos_history = []

        for idx, instance in enumerate(self.scene["instances"]):
            if "sound_file" in instance:

                is_ambient = instance.get("sound_type") == "ambient"
                item_id = instance.get("item_id", f"sound_object_{idx}")

                trigger_mode = "trigger"  # default
                sound_file = instance["sound_file"]  # default from scene
                level_has_trigger_mode = False

                if hasattr(self, 'level_info') and self.level_info:
                    for item in self.level_info.get("room", {}).get("items", []):
                        if item.get("id") == item_id:
                            if "sound_file" in item:
                                level_sound_file = item.get("sound_file")
                                if level_sound_file != sound_file:
                                    logger.debug(f"Using sound_file from level JSON for {item_id}: {level_sound_file} (was {sound_file})")
                                sound_file = level_sound_file
                            if "trigger_mode" in item:
                                trigger_mode = item.get("trigger_mode", "trigger")
                                level_has_trigger_mode = True
                                logger.debug(f"Using trigger_mode from level JSON for {item_id}: {trigger_mode}")
                            break

                if not level_has_trigger_mode and "trigger_mode" in instance:
                    trigger_mode = instance.get("trigger_mode", "trigger")

                obj_info = {
                    "instance_id": idx,
                    "item_id": item_id,
                    "sound_file": sound_file,
                    "trigger_mode": trigger_mode,  # "trigger" or "grab"
                    "distance_limit": instance.get("max_distance") if is_ambient else instance.get("interaction_distance", 5.0),
                    "interaction_angle": instance.get("interaction_angle", 120.0),
                    "is_ambient": is_ambient,
                    "sound_type": instance.get("sound_type", "trigger")
                }
                self.sound_objects.append(obj_info)

                if not is_ambient:
                    self.intent_tracker[idx] = {
                        'time_entered_zone': 0,
                        'last_distance': float('inf'),
                        'is_in_zone_prev_frame': False
                    }
        if self.sound_objects:
            ambient_count = sum(1 for obj in self.sound_objects if obj.get("is_ambient"))
            trigger_count = len(self.sound_objects) - ambient_count
            logger.info(f"Audio trigger system initialized. Found {ambient_count} ambient sounds and {trigger_count} trigger sounds.")

    def _sync_trigger_mode_to_items(self):
        """Sync trigger_mode from scene instances to BaseGame.items."""
        for sound_obj in self.sound_objects:
            item_id = sound_obj.get("item_id")
            trigger_mode = sound_obj.get("trigger_mode", "trigger")
            if item_id in self.base_game.items:
                self.base_game.items[item_id]["trigger_mode"] = trigger_mode
                logger.debug(f"Synced trigger_mode={trigger_mode} to item {item_id}")

    def _update_ambient_sounds(self):
        """Update ambient sounds based on agent position (distance-based, no intent required)."""
        if not self.sound_objects or not self.game.obs.game_states:
            return

        agent_pos = np.array(vec_xz(self.game.obs.game_states["agent"]["position"]))

        for sound_obj in self.sound_objects:
            if sound_obj.get("is_ambient"):
                instance_id = sound_obj['instance_id']
                obj_pos = np.array(vec_xz(self.game.obs.game_states["instances"][instance_id]["position"]))
                dist = np.linalg.norm(agent_pos - obj_pos)
                limit = sound_obj['distance_limit']

                if dist < limit:
                    self.base_game.audio_manager.play_ambient_sound(
                        sound_obj['item_id'],
                        sound_obj['sound_file']
                    )

                    self.base_game.audio_manager.update_ambient_volume(
                        sound_obj['item_id'],
                        dist,
                        limit
                    )
                    volume_ratio = 1.0 - (dist / limit if limit else 0.0)
                    volume_ratio = max(0.0, min(1.0, volume_ratio))
                    try:
                        with self.base_game.audio_manager.lock:
                            state = self.base_game.audio_manager.sound_states.get(sound_obj['item_id'], {})
                            state['current_volume'] = volume_ratio
                            state['sound_type'] = 'ambient'
                            state['sound_file'] = sound_obj['sound_file']
                            self.base_game.audio_manager.sound_states[sound_obj['item_id']] = state
                    except Exception as e:
                        logger.debug(f"Failed to update ambient volume cache for {sound_obj['item_id']}: {e}")
                else:
                    self.base_game.audio_manager.stop_sound(sound_obj['item_id'])
                    try:
                        with self.base_game.audio_manager.lock:
                            state = self.base_game.audio_manager.sound_states.get(sound_obj['item_id'], {})
                            state['current_volume'] = 0.0
                            self.base_game.audio_manager.sound_states[sound_obj['item_id']] = state
                    except Exception:
                        pass

    def _get_forward_vector(self, rotation_y):
        """Converts a Y-axis rotation angle (in degrees) to a 2D forward vector."""
        rad = math.radians(rotation_y)
        return np.array([math.sin(rad), math.cos(rad)])

    def _check_audio_triggers(self):
        """
        Core function called in the game loop to evaluate agent intent and trigger audio.
        It calculates an intent score based on distance, viewing angle, dwell time,
        and movement direction relative to sound-emitting objects.
        """
        if not self.sound_objects or not self.game.obs.game_states:
            return

        agent_state = self.game.obs.game_states["agent"]
        agent_pos = np.array(vec_xz(agent_state["position"]))

        self.agent_pos_history.append(agent_pos)
        if len(self.agent_pos_history) > AGENT_HISTORY_LENGTH:
            self.agent_pos_history.pop(0)

        for sound_obj in self.sound_objects:
            if sound_obj.get("is_ambient"):
                continue

            instance_id = sound_obj['instance_id']
            obj_instance = self.game.obs.game_states["instances"][instance_id]
            obj_pos = np.array(vec_xz(obj_instance["position"]))

            if isinstance(obj_instance["rotation"], list) and len(obj_instance["rotation"]) > 1:
                rotation_y = obj_instance["rotation"][1]
            elif isinstance(obj_instance["rotation"], dict) and "y" in obj_instance["rotation"]:
                rotation_y = obj_instance["rotation"]["y"]
            else:
                rotation_y = 0

            obj_forward = self._get_forward_vector(rotation_y)

            dist = np.linalg.norm(agent_pos - obj_pos)

            vector_to_agent = agent_pos - obj_pos
            is_in_angle = False
            if np.linalg.norm(vector_to_agent) > 1e-6:
                norm_vector_to_agent = vector_to_agent / np.linalg.norm(vector_to_agent)
                dot_product = np.dot(obj_forward, norm_vector_to_agent)
                angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                if angle < sound_obj['interaction_angle'] / 2:
                    is_in_angle = True

            is_in_zone_now = (dist < sound_obj['distance_limit']) and is_in_angle

            tracker = self.intent_tracker[instance_id]

            if is_in_zone_now and not tracker['is_in_zone_prev_frame']:
                tracker['time_entered_zone'] = time.time()

            dwell_duration = 0
            if is_in_zone_now:
                dwell_duration = time.time() - tracker['time_entered_zone']

            is_moving_towards = dist < tracker['last_distance']

            intent_score = 0
            if is_in_zone_now:
                distance_score = 1.0 - (dist / sound_obj['distance_limit'])
                dwell_score = min(1.0, dwell_duration / DWELL_TIME_FOR_MAX_SCORE)
                move_score = 0.6 if is_moving_towards else 0.3

                intent_score = (distance_score * 0.5) + (dwell_score * 0.3) + (move_score * 0.2)

            tracker['last_distance'] = dist
            tracker['is_in_zone_prev_frame'] = is_in_zone_now

    def _fix_audio_paths(self):
        """Fix relative audio file paths in scene data to absolute paths."""
        if "instances" not in self.scene:
            return

        for instance in self.scene["instances"]:
            if "sound_file" in instance:
                sound_file = instance["sound_file"]
                if isinstance(sound_file, str) and sound_file.startswith("assets/"):
                    instance["sound_file"] = os.path.join(ASSETS_DIR, sound_file[7:])

    def get_current_audio_info(self):
        """
        Collect information about currently playing sounds from the audio manager.
        Returns a dict with trigger_sounds and ambient_sounds lists.
        """
        audio_info = {
            "trigger_sounds": [],
            "ambient_sounds": []
        }

        if not hasattr(self.base_game, 'audio_manager') or not self.base_game.audio_manager:
            logger.debug("[AUDIO-DETECT] No audio_manager available")
            return audio_info

        audio_manager = self.base_game.audio_manager
        audio_manager.prune_finished_sounds()

        with audio_manager.lock:
            for item_id, sound_state in audio_manager.sound_states.items():
                if sound_state.get('is_playing', False):
                    sound_type = sound_state.get('sound_type', 'trigger')

                    description = "playing"
                    for instance in self.scene.get("instances", []):
                        if instance.get("item_id") == item_id:
                            item_type = instance.get("item_type", "")
                            if item_type == "musicbox":
                                description = "music box playing"
                            elif item_type == "recorder":
                                description = "recorder playing message"
                            elif item_type == "radiogram":
                                description = "radio playing message"
                            elif item_type == "tv":
                                description = "TV playing"
                            break

                    sound_file = sound_state.get('sound_file', '')

                    sound_info = {
                        "item_id": item_id,
                        "description": description,
                        "sound_file": sound_file,
                        "current_volume": sound_state.get('current_volume', 0.0)
                    }

                    if sound_type == 'ambient':
                        audio_info["ambient_sounds"].append(sound_info)
                        logger.info(f"[AUDIO-DETECT] Found ambient sound: {item_id} ({sound_file})")
                    else:
                        audio_info["trigger_sounds"].append(sound_info)
                        logger.info(f"[AUDIO-DETECT] Found trigger sound: {item_id} ({sound_file})")

        logger.info(f"[AUDIO-DETECT] Collected {len(audio_info['trigger_sounds'])} trigger sound(s) and {len(audio_info['ambient_sounds'])} ambient sound(s)")
        return audio_info

