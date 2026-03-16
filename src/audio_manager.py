import os
import threading
import time

from log_config import configure_logger

try:
    import pygame
except ImportError:
    print("Pygame not found. Audio features will be disabled. Please install it using: pip install pygame")
    pygame = None

logger = configure_logger(__name__)


class AudioTriggerManager:
    """
    Manages audio playback in a separate thread.
    Uses a channel system to support independent control (play/pause/stop) for multiple sounds.
    """
    def __init__(self):
        if pygame:
            # Heuristic to detect headless server:
            # 1. Linux OS
            # 2. No DISPLAY environment variable (common in SSH/servers)
            # 3. No /dev/snd directory
            is_headless = False
            if hasattr(os, 'uname') and os.uname().sysname == "Linux":
                if os.environ.get("DISPLAY") is None:
                    is_headless = True
                if not os.path.exists("/dev/snd"):
                    is_headless = True

            if is_headless and os.environ.get("SDL_AUDIODRIVER") is None:
                os.environ["SDL_AUDIODRIVER"] = "dummy"
                logger.info("Headless environment detected. Using 'dummy' audio driver.")

            try:
                pygame.mixer.init()
                pygame.display.init()
            except (pygame.error, Exception) as e:
                # Fallback if init fails
                logger.warning(f"Audio/Display init failed: {e}. Falling back to dummy driver.")
                os.environ["SDL_AUDIODRIVER"] = "dummy"
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                try:
                    pygame.mixer.init()
                    pygame.display.init()
                except (pygame.error, Exception) as e2:
                     logger.error(f"Failed to initialize pygame mixer/display even with dummy driver: {e2}. Audio features might be broken.")

            if pygame.mixer.get_init():
                pygame.mixer.set_num_channels(16)
        self.sound_states = {}
        self.sounds = {}
        self.channels = {}
        self.lock = threading.Lock()

    def play_sound(self, item_id, sound_file, loop=True):
        # loop=False for grab mode. Play sound once.
        if not pygame:
            return
        with self.lock:
            if self.sound_states.get(item_id, {}).get('is_playing', False):
                return

            if self.sound_states.get(item_id, {}).get('is_paused', False):
                if item_id in self.channels:
                    channel = self.channels[item_id]
                    channel.unpause()
                    self.sound_states[item_id]['is_playing'] = True
                    self.sound_states[item_id]['is_paused'] = False
                    logger.info(f"Audio resumed for item: {item_id}")
                    return

            try:
                resolved_sound_file = sound_file
                if not os.path.exists(resolved_sound_file):
                    potential_path = os.path.join("..", resolved_sound_file)
                    if os.path.exists(potential_path):
                        resolved_sound_file = potential_path

                if item_id not in self.sounds:
                    sound = pygame.mixer.Sound(resolved_sound_file)
                    self.sounds[item_id] = sound
                else:
                    sound = self.sounds[item_id]

                channel = pygame.mixer.find_channel()
                if channel is None:
                    channel = pygame.mixer.Channel(0)

                loops = -1 if loop else 0
                channel.play(sound, loops)
                self.channels[item_id] = channel

                duration = sound.get_length()
                self.sound_states[item_id] = {
                    'is_playing': True,
                    'is_paused': False,
                    'sound_file': resolved_sound_file,
                    'loop': loop,
                    'expected_end_ts': None if loop else time.time() + duration + 0.1
                }
                logger.info(f"Audio started for item: {item_id} on channel {channel} (loop={loop})")
            except Exception as e:
                logger.error(f"Error playing sound {resolved_sound_file} for {item_id}: {e}")

    def pause_sound(self, item_id):
        """Pauses the sound for a specific item_id."""
        if not pygame:
            return
        with self.lock:
            if self.sound_states.get(item_id, {}).get('is_playing', False):
                if item_id in self.channels:
                    channel = self.channels[item_id]
                    channel.pause()
                    self.sound_states[item_id]['is_playing'] = False
                    self.sound_states[item_id]['is_paused'] = True
                    logger.info(f"Audio paused for item: {item_id}")

    def stop_sound(self, item_id):
        """Stops the sound for a specific item_id completely."""
        if not pygame:
            return
        with self.lock:
            if item_id in self.channels:
                channel = self.channels[item_id]
                channel.stop()
                self.sound_states[item_id] = {
                    'is_playing': False,
                    'is_paused': False
                }
                logger.info(f"Audio stopped for item: {item_id}")

    def stop_all_sounds(self):
        """Stops all currently playing sounds and cleans up resources."""
        if pygame:
            with self.lock:
                for item_id, channel in self.channels.items():
                    channel.stop()
                self.channels.clear()
                self.sound_states.clear()
                self.sounds.clear()
            pygame.mixer.quit()
        logger.info("All sounds stopped and mixer uninitialized.")

    def play_ambient_sound(self, item_id, sound_file, volume=1.0):
        """Plays a continuous, looping ambient sound."""
        if not pygame:
            return
        with self.lock:
            if self.sound_states.get(item_id, {}).get('is_playing', False):
                return

            try:
                resolved_sound_file = sound_file
                if not os.path.exists(resolved_sound_file):
                    potential_path = os.path.join("..", resolved_sound_file)
                    if os.path.exists(potential_path):
                        resolved_sound_file = potential_path

                if item_id not in self.sounds:
                    sound = pygame.mixer.Sound(resolved_sound_file)
                    self.sounds[item_id] = sound
                else:
                    sound = self.sounds[item_id]

                sound.set_volume(volume)

                channel = pygame.mixer.find_channel()
                if channel is None:
                    channel = pygame.mixer.Channel(0)

                channel.play(sound, -1)
                self.channels[item_id] = channel

                self.sound_states[item_id] = {
                    'is_playing': True,
                    'is_paused': False,
                    'sound_file': resolved_sound_file,
                    'sound_type': 'ambient'
                }
                logger.info(f"Ambient audio started for item: {item_id}")
            except Exception as e:
                logger.error(f"Error playing ambient sound {resolved_sound_file} for {item_id}: {e}")

    def update_ambient_volume(self, item_id, distance, max_distance):
        """Updates ambient sound volume based on distance, creating a falloff effect."""
        if not pygame:
            return
        with self.lock:
            if item_id in self.channels and item_id in self.sounds:
                if distance < max_distance:
                    volume = 1.0 - (distance / max_distance)
                    volume = max(0.0, min(1.0, volume))

                    base_volume = self.sound_states[item_id].get('base_volume', 0.6)
                    final_volume = base_volume * volume

                    self.sounds[item_id].set_volume(final_volume)
                else:
                    self.sounds[item_id].set_volume(0.0)

    def prune_finished_sounds(self):
        """Checks channels and updates is_playing state for sounds that have finished naturally."""
        if not pygame:
            return

        with self.lock:
            for item_id, channel in list(self.channels.items()):
                state = self.sound_states.get(item_id, {})
                if state.get('is_playing', False):
                    is_finished = not channel.get_busy()

                    if not is_finished and not state.get('loop', True):
                        end_ts = state.get('expected_end_ts')
                        if end_ts is not None and time.time() >= end_ts:
                            is_finished = True
                            channel.stop()  # ensure channel is freed

                    if is_finished:
                        self.sound_states[item_id]['is_playing'] = False
                        self.sound_states[item_id]['is_paused'] = False
                        logger.debug(f"Audio finished naturally for item: {item_id}")

