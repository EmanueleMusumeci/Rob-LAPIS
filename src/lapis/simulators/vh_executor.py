"""
VirtualHome plan executor and video generator.

Converts a PDDL plan (from LAPIS) to a VirtualHome action script,
executes it in the Unity simulator, and saves a rendered video.

Usage (standalone):
    python -m src.lapis.simulators.vh_executor \\
        --plan results_virtualhome/.../plan_0.out \\
        --scene 1 \\
        --output videos/

Import usage:
    from src.lapis.simulators.vh_executor import run_plan_in_unity
    success, video_path = run_plan_in_unity(plan_path, scene_id, output_dir)
"""

import os
import re
import sys
import glob
import subprocess
from pathlib import Path
from typing import Optional

VH_SIM_DIR = Path("/DATA/CoSTL/third-party/virtualhome/simulation/unity_simulator")
VH_BINARY   = VH_SIM_DIR / "linux_exec.v2.3.0.x86_64"
VH_COMM_DIR = Path("/DATA/CoSTL/third-party/virtualhome/virtualhome/simulation/unity_simulator")

# ---------------------------------------------------------------------------
# PDDL plan → VirtualHome script conversion
# ---------------------------------------------------------------------------

# Maps PDDL action name → VH action name (None = skip)
_ACTION_MAP = {
    "walk_into":         "Walk",
    "walk_towards":      "Walk",
    "find":              None,       # no-op in VH
    "grab":              "Grab",
    "open_obj":          "Open",
    "close_obj":         "Close",
    "put_on":            "PutBack",  # place on surface
    "put_inside":        "PutIn",
    "put_inside_no_open":"PutIn",
    "switch_on":         "SwitchOn",
    "switch_off":        "SwitchOff",
    "plug_in":           "PlugIn",
    "plug_out":          "PlugOut",
    "sit":               "Sit",
    "standup":           "StandUp",
    "lie":               "Lie",
    "drink":             "Drink",
    "drink_from_recipient": "Drink",
    "read":              "Read",
    "eat":               "Eat",
    "wash":              "Wash",
    "pour":              "Pour",
    "drop":              "Drop",
    "turn_to":           "TurnTo",
    "look_at":           "LookAt",
    "watch":             "Watch",
}

def _parse_plan(plan_path: str) -> list[tuple[str, list[str]]]:
    """Parse a PDDL plan file into (action_name, [params]) tuples."""
    actions = []
    with open(plan_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # Format: action_name(param1, param2, ...) or (action_name param1 ...)
            m = re.match(r'(\w+)\(([^)]*)\)', line)
            if m:
                name = m.group(1).lower()
                params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            else:
                # Space-separated format: (action param1 param2)
                m2 = re.match(r'\((\S+)\s*(.*)\)', line)
                if m2:
                    name = m2.group(1).lower()
                    params = m2.group(2).split()
                else:
                    continue
            actions.append((name, params))
    return actions


def _to_vh_line(action_name: str, params: list[str], char_id: int = 0) -> Optional[str]:
    """Convert one PDDL action to a VH script line."""
    vh_action = _ACTION_MAP.get(action_name)
    if vh_action is None:
        return None  # skip

    char = f"<char{char_id}>"
    # params[0] is always the character — strip it
    obj_params = params[1:]

    def _norm(name: str) -> str:
        """Normalize PDDL object name to VH object name (underscores and hyphens → spaces)."""
        return name.replace("_", " ").replace("-", " ")

    if vh_action == "StandUp":
        return f"{char} [{vh_action}]"
    elif len(obj_params) == 1:
        obj = _norm(obj_params[0])
        return f"{char} [{vh_action}] <{obj}> (1)"
    elif len(obj_params) == 2:
        obj1 = _norm(obj_params[0])
        obj2 = _norm(obj_params[1])
        return f"{char} [{vh_action}] <{obj1}> (1) <{obj2}> (1)"
    else:
        return None


def plan_to_vh_script(plan_path: str, char_id: int = 0) -> list[str]:
    """Convert a PDDL plan file to a VH script (list of strings)."""
    actions = _parse_plan(plan_path)
    script = []
    for name, params in actions:
        line = _to_vh_line(name, params, char_id)
        if line:
            script.append(line)
    return script


# ---------------------------------------------------------------------------
# Unity execution
# ---------------------------------------------------------------------------

def _import_comm():
    """Import UnityCommunication, bypassing the broken top-level VH __init__.py."""
    sim_dir = str(VH_COMM_DIR.parent)   # .../virtualhome/simulation
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    from unity_simulator.comm_unity import UnityCommunication
    return UnityCommunication


def _frames_to_video(frames_dir: str, output_path: str, fps: int = 5) -> bool:
    """Combine PNG frames from VH output into an MP4 video using ffmpeg.

    VH saves frames as: <output_folder>/<prefix>/<char_id>/Action_XXXX_0_normal.png
    """
    # VH frame layout: frames_dir/<prefix>/<char_id>/Action_*_normal.png
    frames = sorted(glob.glob(os.path.join(frames_dir, "**", "*_normal.png"), recursive=True))
    if not frames:
        # Fallback: any PNG
        frames = sorted(glob.glob(os.path.join(frames_dir, "**", "*.png"), recursive=True))
    if not frames:
        return False

    pattern = os.path.dirname(frames[0])
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", os.path.join(pattern, "*_normal.png"),
        "-vf", "scale=640:480:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def run_plan_in_unity(
    plan_path: str,
    scene_id: int,
    output_dir: str,
    port: str = "8080",
    x_display: str = "1",
    image_width: int = 640,
    image_height: int = 480,
    frame_rate: int = 5,
    char_model: str = "Chars/Female1",
) -> tuple[bool, Optional[str]]:
    """
    Execute a PDDL plan in the VirtualHome Unity simulator and save a video.

    Args:
        plan_path:   Path to PDDL plan file (.out)
        scene_id:    VH scene number from EAI filename (1-indexed, e.g. 1 or 2)
        output_dir:  Directory to save the output video
        port:        Port for Unity communication
        x_display:   X display number (without ':')
        image_width/height: Render resolution
        frame_rate:  Frames per second
        char_model:  Character model to use

    Returns:
        (success, video_path)  where video_path is None on failure
    """
    script = plan_to_vh_script(plan_path)
    if not script:
        return False, None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = Path(plan_path).parent.parent.parent.name  # e.g. Go_to_sleep_1027_2
    video_path = str(output_dir / f"{task_name}.mp4")
    frames_output = str(VH_SIM_DIR / "Output" / task_name)

    try:
        UnityCommunication = _import_comm()

        comm = UnityCommunication(
            file_name=str(VH_BINARY),
            port=port,
            x_display=x_display,
            timeout_wait=60,
        )

        # VH scenes are 0-indexed; EAI filenames use 1-indexed
        comm.reset(scene_id - 1)
        comm.add_character(char_model)

        success, message = comm.render_script(
            script,
            recording=True,
            find_solution=True,
            output_folder=frames_output,
            file_name_prefix=task_name,
            frame_rate=frame_rate,
            image_width=image_width,
            image_height=image_height,
            image_synthesis=["normal"],
            camera_mode=["AUTO"],
        )

        comm.close()

        if not success:
            print(f"  [VH] render_script failed: {message}")
            return False, None

        # Convert frames → video
        ok = _frames_to_video(frames_output, video_path, fps=frame_rate)
        if ok:
            print(f"  [VH] Video saved: {video_path}")
            return True, video_path
        else:
            print(f"  [VH] Frame-to-video conversion failed (frames in {frames_output})")
            return True, None  # execution succeeded, just no video

    except Exception as e:
        print(f"  [VH] Unity execution error: {e}")
        return False, None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Execute a PDDL plan in VirtualHome and record video")
    parser.add_argument("--plan",    required=True, help="Path to PDDL plan file")
    parser.add_argument("--scene",   type=int, default=1, help="VH scene ID (1-7)")
    parser.add_argument("--output",  default="videos/", help="Output directory for videos")
    parser.add_argument("--port",    default="8080")
    parser.add_argument("--display", default="1", help="X display number")
    parser.add_argument("--dry_run", action="store_true", help="Print VH script without executing")
    args = parser.parse_args()

    script = plan_to_vh_script(args.plan)
    print("VH script:")
    for line in script:
        print(" ", line)

    if not args.dry_run:
        success, video = run_plan_in_unity(
            args.plan, args.scene, args.output,
            port=args.port, x_display=args.display,
        )
        print(f"Execution: {'OK' if success else 'FAILED'}")
        if video:
            print(f"Video: {video}")
