import json
import sys
import argparse

sys.path.append("..")


from legent import Environment, ResetInfo

from SceneGeneration import *
from log_config import configure_logger
import logging

logging.getLogger("PIL").setLevel(logging.INFO)

logger = configure_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--setting_path", type=str, help="setting path")
parser.add_argument("--level_data", type=str, help="level data path (alternative to setting_path)")
args = parser.parse_args()

if args.level_data:
    setting_path = args.level_data
elif args.setting_path:
    setting_path = args.setting_path
else:
    print("Error: Please provide either --setting_path or --level_data")
    sys.exit(1)

generator = DefaultSceneGenerator(
    setting_path,
)

path = generator.save_scene()
logger.info(f"Load scene generated to play......")

scene = format_scene(path)

# Explore the scene
env = Environment(
    env_path="auto", camera_resolution_width=1024, camera_field_of_view=120
)
try:
    obs = env.reset(ResetInfo(scene))
    while True:
        obs = env.step()
        print(obs.game_states["agent"])

finally:
    env.close()
