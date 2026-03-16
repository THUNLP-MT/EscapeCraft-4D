import os

from legent.environment.env_utils import get_default_env_data_path

from log_config import configure_logger

logger = configure_logger(__name__)
logger.info("Initializing the configs......")
logger.debug(f"Default env path: {get_default_env_data_path()}")


# openai / gemini config
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = "" # You **cannot** leave it blank even if you are using the official openai api.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

#qwen3api
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "") 

# WebSocket config
DASHSCOPE_WEBSOCKET_URL = ""  
DASHSCOPE_REALTIME_MODEL = ""  


# __config_path = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(__file__)
if os.path.isabs(PROJECT_PATH):
    try:
        PROJECT_PATH = os.path.relpath(PROJECT_PATH, os.getcwd())
    except ValueError:
        pass

__PREFAB_ROOT = os.path.join(PROJECT_PATH, "../prefabs")
PREFAB_DIR = os.path.normpath(os.path.join(__PREFAB_ROOT, "prefabs"))
ASSETS_DIR = os.path.normpath(os.path.join(PROJECT_PATH, "../assets"))

TEXTURE_PATH = os.path.join(__PREFAB_ROOT, "texture")
CAPTCHA_SIZE = (200, 200)
CAPTCHA_WALL_SIZE = [1.5, 1.5, 0.1]
CAPTCHA_FONT_SIZE = 36
CAPTCHA_FONT_PATH = "path/to/you/font.ttf" # Please change it to the path of a valid .ttf font file in your env



EXIT_WALL = "LowPolyInterior2_Wall1_C1_02" # The wall material for the exit wall
# The door material for the exit door. If you want to use your own door prefab, please change the way to calculate the door size in SceneGeneration.py/DefaultSceneGenerator.__get_exit_door()
EXIT_DOOR = "door/door_1.glb"



ROOM_MIN, ROOM_MAX = (1, 1)


# door_size
DOOR_WIDTH = 1.5
DOOR_HEIGHT = 2.5


# box_size
BOX_HEIGHT = 0.2


# Game Configs
GAME_CACHE_DIR = "./game_cache" # The path for saving the records

PREFABS_USED = {
    "box": {"path": "box/box_1.glb", "height": BOX_HEIGHT},
    "key": {
        "path": "key/key_1.glb",
        "rotation": [90, 0, 0],
        "scale": [0.0005, 0.0005, 0.0005],
    },
    "paper": {"path": "paper/paper_1.glb"},
    "musicbox": {
        "path": "musicbox/musicbox_1.glb",
        "rotation": [0, 0, 0],
        "scale": [0.02, 0.02, 0.02],
        "sound_file": "assets/musicbox_sound.wav",
        "interaction_angle": 360,
        "interaction_distance": 4.0,
    },
    "recorder": {
        "path": "recorder/recorder_withsound.glb",
        "rotation": [0, 0, 0],
        "scale": [1.0, 1.0, 1.0],
        "sound_file": "assets/recorder_speech_value.wav",
        "interaction_angle": 360,
        "interaction_distance": 4.0,
    },
    "radiogram": {
        "path": "radiogram/radiogram_withsound.glb",
        "rotation": [0, 0, 0],
        "scale": [0.005, 0.005, 0.005],
        "sound_file": "assets/radiogram_speech.wav",
        "interaction_angle": 360,
        "interaction_distance": 4.0,
    },
    "tv": {
        "path": "tv/tv_withsound.glb",
        "rotation": [0, 180, 0],
        "scale": [0.0005, 0.0004, 0.0005],
        "sound_file": "assets/tv_music.wav",
        "interaction_angle": 270,
        "interaction_distance": 4.0,
    },
}


# This is the max distance for interaction. If you want to be strict to the model evaluation, you can set it smaller.
MAX_INTERACTION_DISTANCE = 4


ROTATION_SPEED = 60 # degrees per second
FORWARD_SPEED = 2.0 # meters per second
GRAB_TIME = 0.5  # seconds added when grabbing an item


# Object type definitions for scene generation priority:
# Feature -> Critical -> Decoration
# Feature items are placed first, followed by critical escape items, then decorations.

ITEM_CATEGORIES = {
    "FEATURE": ["musicbox", "box", "recorder","radiogram","tv"], 
    "CRITICAL": ["key","paper"],
    "DECORATION": []
}

# Ambient audio configuration 
# Maximum hearing distance in meters
# Enable loop playback
AMBIENT_SOUNDS = {
    "exit_door": {
        "sound_file": "assets/gentle_breeze.mp3",
        "volume": 0.6,
        "max_distance": 8.0,
        "loop": True,
    }
}

logger.info("All configs loaded.")
