import json
import sys
import argparse
import math
import numpy as np
import time
import os
import math

sys.path.append("..")

from legent import (
    Environment,
    ResetInfo
)
from legent.utils.math import vec_xz
from audio_manager import AudioTriggerManager
from config import PREFABS_USED

import logging
from log_config import configure_logger
from utils import format_scene

logger = configure_logger(__name__)

logging.getLogger("PIL").setLevel(logging.WARNING)


def build_scene_with_lights(scene):

    scene["walls"] = scene.get("walls", [])
    scene["lights"] = scene.get("lights", [])
    
    # light_position, you can change it
    light_position = [6.5, 1.6, 3.60]
    light_rotation = [90, 0, 0]
    scene["walls"].append(
        {"position": light_position, "rotation": light_rotation, "size": [0.001, 0.001, 0.001], "material": "Light"}
    )
    scene["lights"].append(
        {
            "name": "PointLight0",
            "lightType": "Point",
            "position": light_position,
            "rotation": light_rotation,
            "useColorTemperature": True,
            "colorTemperature": 5500.0,
            "color": [0.86, 0.75, 0.39],
            "intensity": 10,  # brightness
            "range": 50,
            "shadowType": "Soft",
        }
    )
    
    return scene
"audio"
INTENT_THRESHOLD, DWELL_TIME_FOR_MAX_SCORE, AGENT_HISTORY_LENGTH = 0.6, 2.0, 10

def get_forward_vector(r): 
    rad = math.radians(r)
    return np.array([math.sin(rad), math.cos(rad)])

def initialize_audio_trigger_system(scene):
    sound_objects, intent_tracker = [], {}
    ambient_sounds = []
    
    for i, instance in enumerate(scene.get("instances", [])):
        if "sound_file" in instance:
            p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", instance["sound_file"])
            if not os.path.exists(p): 
                print(f"[Warning] Sound file not found: {p}")
                continue
            
            sound_type = instance.get("sound_type", "trigger")
            
            if sound_type == "ambient":
                ambient_sounds.append({
                    "instance_id": i,
                    "item_id": instance.get("item_id", f"ambient_{i}"),
                    "sound_file": p,
                    "volume": instance.get("volume", 0.6),
                    "max_distance": instance.get("max_distance", 8.0)
                })
            else:
                sound_objects.append({
                    "instance_id": i, 
                    "item_id": instance.get("item_id", f"s_{i}"), 
                    "sound_file": p, 
                    "interaction_angle": instance.get("interaction_angle", 120.0), 
                    "interaction_distance": instance.get("interaction_distance", 5.0)
                })
                intent_tracker[i] = {
                    "time_entered_zone": 0, 
                    "last_distance": float("inf"), 
                    "is_in_zone_prev_frame": False
                }

    print(f"[INFO] Audio trigger system initialized. Found {len(sound_objects)} trigger sound objects and {len(ambient_sounds)} ambient sound objects.")
    return sound_objects, intent_tracker, ambient_sounds

def check_audio_triggers(obs, sound_objects, intent_tracker, audio_manager, agent_pos_history, ambient_sounds=[]):
    """Check audio trigger conditions and update ambient sound volume."""
    if not getattr(obs, "game_states", None):
        return

    # Prefer player position, fall back to agent position if not available
    if "player" in obs.game_states and "position" in obs.game_states["player"]:
        agent_pos = np.array(vec_xz(obs.game_states["player"]["position"]))
    elif "agent" in obs.game_states and "position" in obs.game_states["agent"]:
        agent_pos = np.array(vec_xz(obs.game_states["agent"]["position"]))
    else:
        return

    # Update ambient sound volume (based on distance)
    for amb_obj in ambient_sounds:
        try:
            obj_inst = obs.game_states["instances"][amb_obj["instance_id"]]
            obj_pos = np.array(vec_xz(obj_inst["position"]))
            dist = np.linalg.norm(agent_pos - obj_pos)

            # Update volume (closer distance means louder)
            audio_manager.update_ambient_volume(
                amb_obj["item_id"], 
                dist, 
                amb_obj["max_distance"]
            )
        except (IndexError, KeyError):
            continue
        
    agent_pos_history.append(agent_pos)
    if len(agent_pos_history) > AGENT_HISTORY_LENGTH: 
        agent_pos_history.pop(0)
    
    for s_obj in sound_objects:
        try: 
            obj_inst = obs.game_states["instances"][s_obj["instance_id"]]
        except (IndexError, KeyError): 
            continue
        
        obj_pos = np.array(vec_xz(obj_inst["position"]))
        
        rotation = obj_inst.get("rotation", [0, 0, 0])
        if isinstance(rotation, dict):
            y_rotation = rotation.get("y", rotation.get("yaw", 0))
        elif isinstance(rotation, (list, tuple)) and len(rotation) >= 2:
            y_rotation = rotation[1]
        else:
            y_rotation = 0
            
        obj_fwd = get_forward_vector(y_rotation)
        dist = np.linalg.norm(agent_pos - obj_pos)
        vec_to_agent = agent_pos - obj_pos
        is_in_angle = False
        
        if np.linalg.norm(vec_to_agent) > 1e-6:
            angle = math.degrees(math.acos(np.clip(np.dot(obj_fwd, vec_to_agent/np.linalg.norm(vec_to_agent)), -1.0, 1.0)))
            if angle < s_obj["interaction_angle"] / 2: 
                is_in_angle = True
        
        is_now = (dist < s_obj["interaction_distance"]) and is_in_angle
        t = intent_tracker[s_obj["instance_id"]]
        
        if dist < s_obj["interaction_distance"] * 1.5:
            print(f"\r[DEBUG] Distance to {s_obj['item_id']}: {dist:.2f}m (Trigger Distance: {s_obj['interaction_distance']}m, Angle: {angle:.1f}°, Trigger Angle: {s_obj['interaction_angle']}°)", end="", flush=True)

        if is_now and not t["is_in_zone_prev_frame"]:
            t["time_entered_zone"] = time.time()
            print(f"\n[DEBUG] Entered audio trigger zone of {s_obj['item_id']}!")

        dwell = time.time() - t["time_entered_zone"] if is_now else 0
        is_moving_towards = dist < t["last_distance"]
        intent = 0.0
        
        if is_now: 
            intent = (1.0 - dist/s_obj["interaction_distance"]) * 0.5 + min(1.0, dwell/DWELL_TIME_FOR_MAX_SCORE) * 0.3 + (0.6 if is_moving_towards else 0.3) * 0.2
            print(f"\n[DEBUG] {s_obj['item_id']} intent score: {intent:.2f} (Threshold: {INTENT_THRESHOLD})")

            if intent > INTENT_THRESHOLD:
                audio_manager.play_sound(s_obj["item_id"], s_obj["sound_file"])
                print(f"\n[INFO] Playing {s_obj['item_id']} -> (Score={intent:.2f})")
        else:
            if t["is_in_zone_prev_frame"]:
                audio_manager.pause_sound(s_obj["item_id"])
                print(f"\n[DEBUG] Left audio trigger zone of {s_obj['item_id']}, paused playback")

        t["last_distance"] = dist
        t["is_in_zone_prev_frame"] = is_now

##########################
parser = argparse.ArgumentParser()
parser.add_argument("--scene_path", type=str, help="scene path to load")
parser.add_argument("--enable_audio", action="store_true", help="enable audio trigger system")
args = parser.parse_args()

scene = format_scene(args.scene_path)

if args.enable_audio:
    for instance in scene.get("instances", []):
        if instance.get("item_type") == "musicbox":
            musicbox_config = PREFABS_USED.get("musicbox", {})
            if "sound_file" in musicbox_config:
                instance["sound_file"] = musicbox_config["sound_file"]
            if "interaction_angle" in musicbox_config:
                instance["interaction_angle"] = musicbox_config["interaction_angle"]
            if "interaction_distance" in musicbox_config:
                instance["interaction_distance"] = musicbox_config["interaction_distance"]
            print(f"[INFO] Added audio configuration for music box: {instance.get('item_id', 'unknown')}")

scene = build_scene_with_lights(scene)

# Explore
env = Environment(
    env_path="auto", camera_resolution_width=1024, camera_field_of_view=120, rendering_options={"use_default_light": 0}
)

# Initialize audio system
audio_manager = None
sound_objects = []
intent_tracker = {}
agent_pos_history = []
ambient_sounds = []

if args.enable_audio:
    try:
        from audio_manager import AudioTriggerManager
        audio_manager = AudioTriggerManager()
        sound_objects, intent_tracker, ambient_sounds = initialize_audio_trigger_system(scene)
        
        for amb_obj in ambient_sounds:
            audio_manager.play_ambient_sound(
                amb_obj["item_id"],
                amb_obj["sound_file"],
                amb_obj["volume"]
            )
            if amb_obj["item_id"] in audio_manager.sound_states:
                audio_manager.sound_states[amb_obj["item_id"]]['base_volume'] = amb_obj["volume"]

        print(f"[INFO] Audio system initialized. Found {len(sound_objects)} trigger sound objects and {len(ambient_sounds)} ambient sound objects.")
    except Exception as e:
        print(f"[WARN] Failed to initialize audio system: {e}")
        audio_manager = None

logger.warning("Please press Q on the keyboard to start/exit the light mode.")
logger.warning("Please press X on the keyboard to start/exit the full-screen mode.")
logger.warning("Please press ESC on the keyboard to release the mouse.")
if args.enable_audio:
    logger.warning("Audio trigger system is enabled. Walk close to music box to trigger sound.")

try:
    obs = env.reset(ResetInfo(scene))
    while True:
        obs = env.step()
        
        if args.enable_audio and audio_manager:
            check_audio_triggers(obs, sound_objects, intent_tracker, audio_manager, agent_pos_history, ambient_sounds)

finally:
    if audio_manager:
        audio_manager.stop_all_sounds()
    env.close()