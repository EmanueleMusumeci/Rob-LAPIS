import os
import copy
import numpy as np

from typing import Dict, Set, Union
from collections import defaultdict
from pathlib import Path


def filter_graph(graph: Dict, labels: Set[str]) -> Dict:
    """
    Filter the 3DSG graph to only include objects with labels in the 'labels' set

    :param graph: Dictionary containing the 3DSG
    :param labels: Set of labels (strings) to consider, e.g. {"room", "object"}
    :return new_graph: Dictionary containing the filtered 3DSG
    """
    new_graph = {}
    for key, item in graph.items():
        if key in labels:
            new_graph[key] = item
    return new_graph


def save_graph(graph: Dict, path: str):
    """
    Save the 3DSG graph to a file

    :param graph: Dictionary containing the 3DSG
    :param path: Path to the file
    """
    np.savez(path, output=graph)


def collapse_scene_graph(graph: Dict) -> Dict:
    """
    Collapse the nodes of a 3DSG by removing its objects.
    
    :param graph: Dictionary containing the 3DSG
    :return: 3DSG with unexpanded room nodes A
    """
    csg = copy.deepcopy(graph)
    for k, v in csg.items():
        csg[k] = []

    return csg


def update_scene_graph(sg: Dict, orig_sg: Dict, command: str, room: str) -> Dict:
    if command == "expand":
        if room not in orig_sg:
            orig_sg[room] = []
            sg[room] = []
            sg[room] = orig_sg[room]
        else:
            assert sg[room] == [], "Room is not empty!"
            sg[room] = orig_sg[room]
    elif command == "contract":
        sg[room] = []
    else:
        raise Exception("Invalid command!")
    return sg


def read_graph_from_path(path: Path) -> Dict:
    """
    Read 3DSG from file (.npz) and returns it stored in a dictionary

    :param path: Path to the .npz file
    :return: Dictionary containing the 3DSG
    """

    assert isinstance(path, Path), "Input file is not a Path"
    assert str(path).endswith(".npz"), "Input file is not .npz object"

    graph = np.load(path, allow_pickle=True)["output"].item()

    keeps = set(["object", "room"])
    graph = filter_graph(graph, keeps)

    return graph


def get_verbose_scene_graph(graph: Dict, as_string: bool = True, include_descriptions: bool = True) -> Union[str, Dict]:
    """
    Given a 3DSG, return a verbose, discursive description of the scene graph, or a dict.

    :param graph: Dictionary containing the 3DSG
    :param as_string: Whether to output as a string or dict
    :return: String with the verbose scene graph or dict mapping rooms to objects
    """
    rooms = graph.get("room", {})
    objects = graph.get("object", {})

    # 1. Create labels for rooms and objects
    room_id_to_label = {
        r_id: f"{info.get('scene_category', 'UnnamedRoom')}_{r_id}"
        for r_id, info in rooms.items()
    }
    obj_id_to_label = {
        o_id: f"{info.get('class_', 'UnnamedObject')}_{o_id}"
        for o_id, info in objects.items()
    }

    # 2. Group objects by their parent room
    room_to_objects = defaultdict(list)
    for o_id, info in objects.items():
        r_id = info.get('parent_room')
        if r_id in rooms:
            label = obj_id_to_label[o_id]
            # Prevent the label from containing spaces instead of underscores
            label = label.replace(" ", "_")
            if include_descriptions:
                desc = info.get('description', None)
                room_to_objects[r_id].append((label, desc))
            else:
                room_to_objects[r_id].append(label)

    if not as_string:
        # Return raw dict
        return {
            room_id_to_label[r_id]: room_to_objects.get(r_id, [])
            for r_id in rooms
        }

    # 3. Build discursive string
    # List of room labels in order
    labels = [room_id_to_label[r_id] for r_id in rooms]
    # Intro sentence
    output = []
    output.append(
        "The 3DSG is made of these rooms: " + ", ".join(labels) + "."
    )

    # One sentence per room
    for r_id in rooms:
        room_label = room_id_to_label[r_id]
        items = room_to_objects.get(r_id, [])
        if items:
            # Describe objects
            parts = [f"{name} ({desc})" if desc is not None else f"{name}" for name, desc in items]
            line = f"The {room_label} contains " + ", ".join(parts) + "."
        else:
            line = f"The {room_label} has no objects."
        output.append(line)

    # Join into single paragraph
    return "\n".join(output)

def prune_sg_with_item(sg: dict, item_keep: list, is_extracted_sg: bool = False):
    if is_extracted_sg:
        pruned_sg = copy.deepcopy(sg)
    else:
        pruned_sg = copy.deepcopy(sg["rooms"])

    for room_name, room_items in pruned_sg.items():
        pruned_items = []
        for item_name in room_items:
            # Only keep items that are in the item_keep list
            if item_name in item_keep:
                pruned_items.append(item_name)
        pruned_sg[room_name] = pruned_items

    return pruned_sg

def export_obj_list(response: str, is_extracted_sg : bool = False):
    """
    Extract a Python list of object names from a (possibly formatted) LLM response string.

    This function is defensive: it finds the first `[` and the last `]`, extracts the
    bracketed content, and tries to parse it using ast.literal_eval (safe). If that
    fails it falls back to extracting quoted tokens or splitting on commas/newlines.

    Returns: list of strings
    Raises: AssertionError if no bracketed list is present or if no items can be parsed.
    """
    import ast
    import re

    # quick check
    assert "[" in response and "]" in response, "No list found in response!"

    # get the substring that contains the list (include the brackets)
    start_idx = response.find("[")
    end_idx = response.rfind("]")
    list_sub = response[start_idx:end_idx + 1]

    # remove common markdown fencing around content (```)
    list_sub = re.sub(r"```+", "", list_sub)

    # Try to safely evaluate the list literal
    try:
        parsed = ast.literal_eval(list_sub)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except Exception:
        pass

    # Fallback 1: extract quoted tokens
    quoted = re.findall(r'["\']([^"\']+)["\']', list_sub)
    if quoted:
        return quoted

    # Fallback 2: split by commas/newlines and clean tokens
    parts = re.split(r"[,\n]", list_sub)
    items = []
    for p in parts:
        token = p.strip()
        token = token.strip('[]() \"\'`')
        if not token:
            continue
        items.append(token)

    assert items, "Could not parse any list items from response"
    return items