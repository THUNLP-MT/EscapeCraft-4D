import os
import glob
import shutil
import subprocess
import sys
import time
import signal

# Configuration
LEVELS = [
    "level1_audio",
    "level2_audio",
    "level2.5_audio",
    "level3_note_first_audio",
    "level3.5_note_first_audio",
    "level3_time"
]

MODEL_NAME = "gemini-3-pro-preview"
LOG_DIR_SUFFIX = "_t_1"
MAX_SCRIPT_RETRIES = 3

def free_port(port=50051):
    """
    Kills any process using the specified port (macOS/Linux).
    """
    try:
        # Find PID(s) using the port
        result = subprocess.run(
            ["lsof", "-t", "-i", f":{port}"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p] # Filter empty strings
        
        if pids:
            print(f"Found process(es) {pids} on port {port}. Killing...")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass # Process already gone
            time.sleep(2) # Give OS time to release port
            print(f"Port {port} cleaned up.")
            
    except Exception as e:
        print(f"Warning: Error checking/killing port {port}: {e}")

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    src_dir = os.path.join(project_root, "src")
    levels_dir = os.path.join(project_root, "levels")
    
    # Change working directory to src
    if not os.path.exists(src_dir):
        print(f"Error: src directory not found at {src_dir}")
        return

    os.chdir(src_dir)
    print(f"Working directory changed to: {os.getcwd()}")
    
    # Initialize environment for Gemini (via OpenAI-compatible API)
    os.environ["OPENAI_BASE_URL"] = "" # Change to your Gemini-compatible API base URL if needed
    
    print("Starting Gemini Robotics model tests...")
    
    for level in LEVELS:
        print("="*40)
        print(f"Testing scenes for Level: {level}")
        print("="*40)
        
        scene_dir = os.path.join(levels_dir, level)
        if not os.path.exists(scene_dir):
            print(f"Directory {scene_dir} does not exist, skipping...")
            continue
            
        # Sort files to ensure deterministic order
        scene_files = sorted(glob.glob(os.path.join(scene_dir, "*.json")))
        
        for scene_file in scene_files:
            filename = os.path.basename(scene_file)
            scene_id = os.path.splitext(filename)[0]
            
            # Construct expected log directory path
            game_level_id = f"{level}-{scene_id}"
            
            # Path to cache dir (relative to src/ since we chdir'd)
            cache_dir_name = f"{game_level_id}/{MODEL_NAME}{LOG_DIR_SUFFIX}"
            cache_dir_path = os.path.join("game_cache", cache_dir_name)
            story_json_path = os.path.join(cache_dir_path, "story.json")
            
            # Retry loop
            for attempt in range(MAX_SCRIPT_RETRIES):
                # Check if successful run exists
                if os.path.exists(story_json_path):
                    if attempt == 0:
                        print(f"Skipping Level: {level}, Scene ID: {scene_id} (Already exists: {story_json_path})")
                    else:
                        print(f"Run successful for {game_level_id}")
                    break
                
                # Check for incomplete run
                if os.path.exists(cache_dir_path):
                    # We only delete if it exists but story.json does NOT exist (checked above)
                    print(f"Incomplete run detected at {cache_dir_path}. Deleting and retrying...")
                    try:
                        shutil.rmtree(cache_dir_path)
                    except Exception as e:
                        print(f"Failed to delete {cache_dir_path}: {e}")
                
                print("-" * 40)
                print(f"Running test for Level: {level}, Scene ID: {scene_id} (Attempt {attempt + 1}/{MAX_SCRIPT_RETRIES})")
                print("-" * 40)
                
                # Ensure port is free before starting
                free_port(50051)

                cmd = [
                    "python", "main.py",
                    "--level", level,
                    "--scene_id", scene_id,
                    "--room_num", "1",
                    "--model", MODEL_NAME,
                    "--max_retry", "3"
                ]
                
                try:
                    subprocess.run(cmd, check=False)
                except Exception as e:
                    print(f"Error running command: {e}")
                
                # After run, loop continues to check story.json existence
                
            else:
                # If loop finishes without break, it failed all attempts
                print(f"Failed to complete Level: {level}, Scene ID: {scene_id} after {MAX_SCRIPT_RETRIES} attempts.")

    print("All Gemini Robotics tests completed.")

if __name__ == "__main__":
    main()
