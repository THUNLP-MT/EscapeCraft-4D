import os
import re
import json
import argparse
from collections import defaultdict


SOUND_TYPES = {"recorder", "radiogram", "musicbox", "tv"}
NON_PROP_TYPES = SOUND_TYPES | {"password", "box"}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _max_step_for_level(level_name):
    if "level1" in level_name:
        return 50
    if "level2" in level_name:
        return 65
    return 80


def _collect_prop_ids(level_json_path):
    level_data = _load_json(level_json_path)
    prop_ids = set()
    for item in level_data.get("room", {}).get("items", []):
        item_type = item.get("type", "")
        if item_type in NON_PROP_TYPES:
            continue
        prop_ids.add(item.get("id", ""))
    prop_ids.discard("")
    # print(prop_ids)
    return prop_ids


def _collect_key_sound_ids(level_json_path):
    level_data = _load_json(level_json_path)
    sound_ids = set()
    for item in level_data.get("room", {}).get("items", []):
        item_type = item.get("type", "")
        if item_type in SOUND_TYPES and item.get("sound_trigger", False):
            sound_ids.add(item.get("id", ""))
    sound_ids.discard("")
    return sound_ids


def _extract_bag_item_ids(bag_text):
    if not bag_text:
        return set()
    ids = set()
    for m in re.finditer(r"(?:item_id|id):\s*([A-Za-z0-9_]+)", str(bag_text)):
        ids.add(m.group(1))
    return ids


def _load_record_or_default(record_path):
    if not os.path.exists(record_path):
        return None
    try:
        data = _load_json(record_path)
        if isinstance(data, list) and len(data) > 0:
            return data
    except Exception:
        return None
    return None


def _summarize_one_record(record, level_name):
    max_step = _max_step_for_level(level_name)

    success = 0
    escaped = False

    if record[-1].get("info") is not None:
        if len(record) >= 2:
            step_count = record[-2].get("step", 0) + 1
            last_inter = record[-2]
        else:
            step_count = record[-1].get("step", 0) + 1
            last_inter = record[-1]
        if "Escaped succesfully!" in str(record[-1].get("info", "")):
            success = 1
            escaped = True
    else:
        last_step = record[-1].get("step", 0)
        step_count = min(max_step, last_step + 1)
        last_inter = record[-1]

    grab_attempts = 0
    for r in record:
        if "info" in r or "response" not in r:
            continue
        if r["response"].get("grab", False):
            grab_attempts += 1

    record_len = len(record)
    if record[-1].get("info") is not None:
        record_len -= 1
    grab_ratio = (grab_attempts / float(record_len)) if record_len > 0 else 0.0

    if grab_attempts > 0:
        if escaped:
            if "level1" in level_name:
                grab_tp = 1.0
            elif "level2" in level_name:
                grab_tp = 1.0
            else:
                grab_tp = 2.0
        else:
            bag = str(last_inter.get("bag", ""))
            grab_tp = 0.0
            if "key_1" in bag:
                grab_tp += 1
            if "key_2" in bag:
                grab_tp += 1
            if "note_1" in bag:
                grab_tp += 1
            if "note_2" in bag:
                grab_tp += 1
        grab_success = grab_tp / grab_attempts
    else:
        grab_success = 0.0

    last_bag = str(last_inter.get("bag", ""))
    return {
        "success": success,
        "steps": step_count,
        "grab_ratio": grab_ratio,
        "grab_success": grab_success,
        "last_bag": last_bag,
    }


def _init_bucket():
    return {
        "total": 0,
        "success": 0,
        "steps": 0.0,
        "grab_ratio_sum": 0.0,
        "grab_success_sum": 0.0,
        "prop_gain_sum": 0.0,
        "prop_gain_count": 0,
        "trigger_rate_sum": 0.0,
        "trigger_success_ratio_sum": 0.0,
    }


def _finalize_bucket(bucket):
    n = bucket["total"]
    if n == 0:
        return None
    prop_gain = None
    if bucket["prop_gain_count"] > 0:
        prop_gain = (bucket["prop_gain_sum"] / bucket["prop_gain_count"]) * 100.0

    return {
        "n": n,
        "er": (bucket["success"] / n) * 100.0,
        "steps": bucket["steps"] / n,
        "grab_sr": (bucket["grab_success_sum"] / n) * 100.0,
        "grab_ratio": (bucket["grab_ratio_sum"] / n) * 100.0,
        "prop_gain": prop_gain,
        "trigger_rate": (bucket["trigger_rate_sum"] / n) * 100.0,
        "trigger_success_ratio": (bucket["trigger_success_ratio_sum"] / n) * 100.0,
    }


def _parse_scene_dir_name(name):
    if "-" not in name:
        return None, None
    try:
        level_name, scene_str = name.rsplit("-", 1)
        if not scene_str.isdigit():
            return None, None
        return level_name, int(scene_str)
    except ValueError:
        return None, None


def _audio_utility_from_record(record, limit_seconds, marker_phrase):
    start_timer = None
    marker_timer = None
    marker_l = marker_phrase.lower()

    for frame in record:
        if "info" in frame:
            continue
        if frame.get("__record_reload", False) and start_timer is None:
            start_timer = float(frame.get("timer", 0.0))

        response = frame.get("response", {})
        rationale = str(response.get("rationale", ""))
        if marker_l in rationale.lower():
            if start_timer is not None:
                marker_timer = float(frame.get("timer", 0.0))
                break

    if start_timer is None or marker_timer is None:
        return 1.0
    elapsed = max(0.0, marker_timer - start_timer)
    return min(1.0, elapsed / float(limit_seconds))


def _extract_triggered_sound_ids(record):
    ids = set()
    unknown_trigger_hit = False

    for frame in record:
        if "info" in frame:
            continue
        ts = frame.get("trigger_sound", None)
        if isinstance(ts, list):
            for entry in ts:
                if isinstance(entry, dict):
                    item_id = entry.get("item_id")
                    if item_id:
                        ids.add(item_id)
                elif isinstance(entry, str) and entry:
                    ids.add(entry)
        elif isinstance(ts, dict):
            item_id = ts.get("item_id")
            if item_id:
                ids.add(item_id)
        elif ts is True:
            unknown_trigger_hit = True
        elif isinstance(ts, str) and ts.strip():
            # fallback for loosely formatted logs
            for m in re.finditer(r"(recorder_[0-9]+|radiogram_[0-9]+|musicbox_[0-9]+|tv_[0-9]+)", ts):
                ids.add(m.group(1))
            if not ids:
                unknown_trigger_hit = True

        # backup parse from textual desc when trigger_sound is absent/incomplete
        desc = str(frame.get("desc", ""))
        for m in re.finditer(r"Audio trigger result: SUCCESS \(([^ )]+)", desc):
            ids.add(m.group(1))
        for m in re.finditer(r"Audio triggered by grab: ([A-Za-z0-9_]+)", desc):
            ids.add(m.group(1))

    return ids, unknown_trigger_hit


def _extract_trigger_action_stats(record):
    trigger_true = 0
    trigger_success = 0

    for frame in record:
        if "info" in frame:
            continue

        response = frame.get("response", {})
        if isinstance(response, dict) and response.get("trigger", False) is True:
            trigger_true += 1

        desc = str(frame.get("desc", ""))
        trigger_success += len(re.findall(r"Audio trigger result: SUCCESS \(", desc))

    if trigger_success > trigger_true:
        trigger_success = trigger_true

    return trigger_true, trigger_success


def eval_all_v2(game_cache, round_id=1):
    if not os.path.exists(game_cache):
        print(f"Path {game_cache} does not exist.")
        return

    level2_prop_ids = _collect_prop_ids(os.path.join("levels", "level2_audio.json"))
    level3_note_prop_ids = _collect_prop_ids(os.path.join("levels", "level3_note_first_audio.json"))
    level3_time_data = _load_json(os.path.join("levels", "level3_time.json"))
    level3_time_limit = (
        level3_time_data["room"]["items"][1]["on_trigger"]["limit_setting"]["limit_seconds"]
    )
    marker_phrase = "I found the password mentioned before"

    # model -> difficulty -> bucket
    stats = defaultdict(lambda: {
        "Difficulty-1": _init_bucket(),
        "Difficulty-2": _init_bucket(),
        "Difficulty-3": _init_bucket(),
    })
    # model -> audio utility list
    audio_utility = defaultdict(list)
    # model -> level -> count of samples
    model_level_counts = defaultdict(lambda: defaultdict(int))

    all_dirs = [d for d in os.listdir(game_cache) if os.path.isdir(os.path.join(game_cache, d))]
    all_dirs.sort()

    for scene_dir in all_dirs:
        level_name, _scene_id = _parse_scene_dir_name(scene_dir)
        if level_name is None:
            continue

        if level_name == "level1_audio":
            difficulty = "Difficulty-1"
        elif level_name == "level2_audio":
            difficulty = "Difficulty-2"
        elif level_name == "level3_note_first_audio":
            difficulty = "Difficulty-3"
        elif level_name == "level3_time":
            difficulty = None
        else:
            continue

        scene_path = os.path.join(game_cache, scene_dir)
        for model_dir in os.listdir(scene_path):
            if model_dir.startswith("."):
                continue
            if f"_t_{round_id}" not in model_dir:
                continue

            model_name = model_dir.split(f"_t_{round_id}")[0]
            record_file = os.path.join(scene_path, model_dir, "records.json")
            record = _load_record_or_default(record_file)

            if level_name == "level3_time":
                if record is not None:
                    au = _audio_utility_from_record(record, level3_time_limit, marker_phrase)
                else:
                    au = 1.0
                audio_utility[model_name].append(au)
                model_level_counts[model_name]["level3_time"] += 1
                continue

            bucket = stats[model_name][difficulty]
            bucket["total"] += 1
            model_level_counts[model_name][level_name] += 1

            if record is None:
                bucket["steps"] += _max_step_for_level(level_name)
                continue

            s = _summarize_one_record(record, level_name)
            bucket["success"] += s["success"]
            bucket["steps"] += s["steps"]
            bucket["grab_ratio_sum"] += s["grab_ratio"]
            bucket["grab_success_sum"] += s["grab_success"]

            bag_ids = _extract_bag_item_ids(s["last_bag"])
            if level_name == "level2_audio":
                prop_ids = level2_prop_ids
            elif level_name == "level3_note_first_audio":
                prop_ids = level3_note_prop_ids
            else:
                prop_ids = level3_note_prop_ids

            total_props = len(prop_ids)
            if total_props > 0:
                got_props = len(bag_ids & prop_ids)
                prop_gain = got_props / float(total_props)
                bucket["prop_gain_sum"] += prop_gain
                bucket["prop_gain_count"] += 1

            record_len = len(record) - (1 if record and record[-1].get("info") is not None else 0)
            trigger_true, trigger_success = _extract_trigger_action_stats(record)
            trigger_rate = (trigger_true / float(record_len)) if record_len > 0 else 0.0
            trigger_success_ratio = (trigger_success / float(trigger_true)) if trigger_true > 0 else 0.0
            bucket["trigger_rate_sum"] += trigger_rate
            bucket["trigger_success_ratio_sum"] += trigger_success_ratio

    all_models = sorted(set(list(stats.keys()) + list(audio_utility.keys())))
    if not all_models:
        print("No matched records found.")
        return

    header = (
        f"{'Models':<30} | "
        f"{'D1 ER(%)':>8} {'D1 Steps':>9} {'D1 GrabSR(%)':>13} {'D1 GrabRatio':>13} | "
        f"{'D2 ER(%)':>8} {'D2 Steps':>9} {'D2 GrabSR(%)':>13} {'D2 GrabRatio':>13} {'D2 TrigSR(%)':>13} {'D2 TrigRate(%)':>15} | "
        f"{'D3 ER(%)':>8} {'D3 Prop(%)':>10} {'D3 Steps':>9} {'D3 GrabSR(%)':>13} {'D3 GrabRatio':>13} {'D3 TrigSR(%)':>13} {'D3 TrigRate(%)':>15} | "
        f"{'AVG ER(%)':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for model in all_models:
        d1 = _finalize_bucket(stats[model]["Difficulty-1"])
        d2 = _finalize_bucket(stats[model]["Difficulty-2"])
        d3 = _finalize_bucket(stats[model]["Difficulty-3"])

        def fmt(v, k, default="-"):
            if v is None:
                return default
            if v.get(k) is None:
                return default
            return f"{v[k]:.2f}"

        er_vals = [x["er"] for x in [d1, d2, d3] if x is not None]
        avg_er = sum(er_vals) / len(er_vals) if er_vals else 0.0

        line = (
            f"{model:<30} | "
            f"{fmt(d1, 'er'):>8} {fmt(d1, 'steps'):>9} {fmt(d1, 'grab_sr'):>13} {fmt(d1, 'grab_ratio'):>13} | "
            f"{fmt(d2, 'er'):>8} {fmt(d2, 'steps'):>9} {fmt(d2, 'grab_sr'):>13} {fmt(d2, 'grab_ratio'):>13} {fmt(d2, 'trigger_success_ratio'):>13} {fmt(d2, 'trigger_rate'):>15} | "
            f"{fmt(d3, 'er'):>8} {fmt(d3, 'prop_gain'):>10} {fmt(d3, 'steps'):>9} {fmt(d3, 'grab_sr'):>13} {fmt(d3, 'grab_ratio'):>13} {fmt(d3, 'trigger_success_ratio'):>13} {fmt(d3, 'trigger_rate'):>15} | "
            f"{avg_er:>10.2f}"
        )
        print(line)

    print("=" * len(header))

    print("\nLevel3-Time TCSS")
    print("=" * 70)
    print(f"{'Models':<30} | {'Scenes':>8} | {'TCSS(%)':>16}")
    print("-" * 70)
    for model in all_models:
        vals = audio_utility.get(model, [])
        if len(vals) == 0:
            print(f"{model:<30} | {0:>8} | {'-':>16}")
            continue
        avg_au = (sum(vals) / len(vals)) * 100.0
        print(f"{model:<30} | {len(vals):>8} | {100-avg_au:>16.2f}")
    print("=" * 70)

    print("\nSamples per Model per Level")
    print("=" * 100)
    
    level_name_mapping = {
        "level1_audio": "Level1",
        "level2_audio": "Level2",
        "level3_note_first_audio": "Level3-Note",
        "level3_time": "Level3-Time",
    }

    levels_order = [
        "level1_audio",
        "level2_audio",
        "level3_note_first_audio",
        "level3_time",
    ]
    header = f"{'Model':<30} | " + " | ".join(f"{level_name_mapping[l]:>12}" for l in levels_order) + " | " + f"{'Total':>12}"
    print(header)
    print("-" * 100)
    
    for model in all_models:
        row = f"{model:<30} | "
        total = 0
        for level_name in levels_order:
            count = model_level_counts[model].get(level_name, 0)
            row += f"{count:>12} | "
            total += count
        row += f"{total:>12}"
        print(row)
    
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models in game cache (v2)")
    parser.add_argument("game_cache", type=str, nargs="?", default="src/game_cache", help="path to game_cache")
    parser.add_argument("--round_id", type=int, default=1, help="round id filter")
    args = parser.parse_args()
    eval_all_v2(args.game_cache, round_id=args.round_id)
