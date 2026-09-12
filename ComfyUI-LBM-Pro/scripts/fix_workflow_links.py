"""Repair link dst_slot indices in example workflows.

ComfyUI assigns a single integer slot index to every entry in
``INPUT_TYPES()["required"]`` followed by every entry in
``["optional"]``, in declaration order — INCLUDING widgets (like the
``INT steps`` entry). Workflows that were generated when a node had a
different INPUT_TYPES layout can end up with ``dst_slot`` pointing at
the wrong input name.

This script reads each workflow JSON, looks up the actual slot for the
input name claimed by the destination node, and rewrites both the
top-level ``links[]`` array and each destination node's
``inputs[].link`` reference to match. It also normalises each node's
``inputs[]`` array to match the order of ``INPUT_TYPES()`` for our 8
nodes (so newly-added widgets and re-ordered fields are reflected).

Run from the repo root: ``python scripts/fix_workflow_links.py``.
"""
from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(REPO_ROOT, "example_workflows")
sys.path.insert(0, REPO_ROOT)

# The custom-node modules import ``comfy.*`` at top level. This script
# only needs their INPUT_TYPES schema (not the comfy runtime), so
# install stubs for the few names the nodes touch if real ComfyUI
# isn't on sys.path.
if "comfy" not in sys.modules:
    import types
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.model_management = types.ModuleType("comfy.model_management")
    fake_comfy.model_management.soft_empty_cache = lambda: None
    fake_comfy.utils = types.ModuleType("comfy.utils")

    class _FakeProgressBar:
        def __init__(self, *_args, **_kwargs):
            pass

        def update_absolute(self, *_args, **_kwargs):
            pass

    fake_comfy.utils.ProgressBar = _FakeProgressBar
    fake_comfy.utils.load_torch_file = lambda *_a, **_kw: {}

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_folder_paths = lambda *_a, **_kw: []
    fake_folder_paths.get_input_directory = lambda: ""
    fake_folder_paths.get_output_directory = lambda: ""
    fake_folder_paths.get_temp_directory = lambda: ""

    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_management"] = fake_comfy.model_management
    sys.modules["comfy.utils"] = fake_comfy.utils
    sys.modules["folder_paths"] = fake_folder_paths

import lbm_nodes.lbm_relighting_pro as r
import lbm_nodes.lbm_depth_normal_pro as dn
import lbm_nodes.lbm_light_preset as lp
import lbm_nodes.lbm_model_loader as ml
import lbm_nodes.lbm_depth_visualizer as dv
import lbm_nodes.lbm_normal_visualizer as nv
import lbm_nodes.lbm_compare_grid as cg
import lbm_nodes.lbm_batch_processor as bp


NODES_BY_TYPE = {
    "LBM_Relighting_Pro": r.LBM_Relighting_Pro,
    "LBM_DepthNormal_Pro": dn.LBM_DepthNormal_Pro,
    "LBM_Light_Preset": lp.LBM_Light_Preset,
    "LBM_Model_Loader": ml.LBM_Model_Loader,
    "LBM_Depth_Visualizer": dv.LBM_Depth_Visualizer,
    "LBM_Normal_Visualizer": nv.LBM_Normal_Visualizer,
    "LBM_Compare_Grid": cg.LBM_Compare_Grid,
    "LBM_Batch_Processor": bp.LBM_Batch_Processor,
}


def slot_for_input(node_type: str, input_name: str):
    """Return the ComfyUI-assigned slot index for ``input_name`` on
    ``node_type``, or ``None`` if the name is not declared."""
    if node_type not in NODES_BY_TYPE:
        return None
    its = NODES_BY_TYPE[node_type].INPUT_TYPES()
    slot = 0
    for cat in ("required", "optional"):
        if cat not in its:
            continue
        for name in its[cat]:
            if name == input_name:
                return slot
            slot += 1
    return None


def fix_workflow(path: str) -> bool:
    """Return True if the workflow was modified."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    nodes_by_id = {n["id"]: n for n in d.get("nodes", [])}

    modified = False

    # Walk the top-level links[] and rewrite dst_slot to match the slot
    # of the destination input name declared on the destination node.
    new_links = []
    for link in d.get("links", []):
        if len(link) < 5:
            new_links.append(link)
            continue
        lid, src, src_slot, dst, dst_slot, *rest = link
        dst_node = nodes_by_id.get(dst)
        # If the link points at one of our 8 nodes, look up the input
        # name claimed by the destination for this link id, and resolve
        # its real slot.
        claimed_name = None
        if dst_node is not None:
            for inp in dst_node.get("inputs", []) or []:
                if inp.get("link") == lid:
                    claimed_name = inp.get("name")
                    break
        if dst_node and claimed_name and dst_node["type"] in NODES_BY_TYPE:
            real_slot = slot_for_input(dst_node["type"], claimed_name)
            if real_slot is not None and real_slot != dst_slot:
                link = [lid, src, src_slot, dst, real_slot] + list(rest)
                modified = True
        new_links.append(link)
    d["links"] = new_links

    # Reorder each of our nodes' inputs[] so it follows INPUT_TYPES
    # order. The names are unchanged; only the array order and any
    # missing-but-declared entries (with no link) are normalised.
    for n in d.get("nodes", []):
        if n["type"] not in NODES_BY_TYPE:
            continue
        its = NODES_BY_TYPE[n["type"]].INPUT_TYPES()
        order = []
        for cat in ("required", "optional"):
            if cat not in its:
                continue
            for name in its[cat]:
                order.append(name)
        existing = {inp.get("name"): inp for inp in n.get("inputs", []) or []}
        new_inputs = []
        for name in order:
            inp = existing.get(name)
            if inp is not None:
                new_inputs.append(inp)
            else:
                # Declared in INPUT_TYPES but not in the JSON's inputs[].
                # Synthesize a stub for completeness (ComfyUI tolerates
                # either way; including it keeps the workflow truthful).
                new_inputs.append({"name": name, "type": "*", "link": None})
        if [i.get("name") for i in new_inputs] != [i.get("name") for i in n.get("inputs", []) or []]:
            n["inputs"] = new_inputs
            modified = True

    if modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
    return modified


def main() -> int:
    files = sorted(
        f for f in os.listdir(WORKFLOW_DIR) if f.endswith(".json")
    )
    changed = []
    for name in files:
        path = os.path.join(WORKFLOW_DIR, name)
        if fix_workflow(path):
            changed.append(name)
    if changed:
        print("Updated:")
        for n in changed:
            print(f"  {n}")
    else:
        print("All example workflows already had correct slot indices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
