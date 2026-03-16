import os
import time
import glob
import subprocess
import sys
import signal

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    levels_dir = os.path.abspath(os.path.join(current_dir, "../../levels"))
    
    level_files = glob.glob(os.path.join(levels_dir, "*.json"))
    
    level_files = ["../../levels/" + os.path.basename(f) for f in level_files]
    
    level_files.sort()

    print(f"Found {len(level_files)} level files: {[os.path.basename(f) for f in level_files]}")

    for level_file in level_files[:]:
        level_name = os.path.basename(level_file)
        print(f"\nProcessing level: {level_name}")
        
        for i in range(1):
            print(f"  Generating scene {i+1}/10 for {level_name}...")

            cmd = [sys.executable, "generate_scene.py", "--setting_path", level_file]
            process = subprocess.Popen(
                cmd,
                cwd=current_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = None, None
            try:
                try:
                    stdout, stderr = process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    print("    Timeout reached (15s), killing process...")
                    process.kill()
                    stdout, stderr = process.communicate()
            except Exception as e:
                print(f"    Error: {e}")
                process.kill()
            
            if stdout:
                print(f"    Stdout: {stdout.decode('utf-8', errors='replace')}")
            if stderr:
                print(f"    Stderr: {stderr.decode('utf-8', errors='replace')}")

            time.sleep(1)

    print("\nAll scenes generated.")

if __name__ == "__main__":
    main()
