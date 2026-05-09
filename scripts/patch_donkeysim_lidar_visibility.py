#!/usr/bin/env python3
"""
Patch DonkeySim prefab assets so vehicle colliders are easier for LiDAR to hit.

This script targets the packaged Unity asset bundle when Unity project sources
are unavailable. It performs a conservative patch on the shared car hitbox:

- `body` BoxCollider center.y
- `body` BoxCollider size.y

Optionally it can also patch the prefab-default LiDAR local Y mount. That
default only matters when runtime lidar_config does not override offsets.

The script always creates a timestamped backup before replacing the asset.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import UnityPy


DEFAULT_SIM_DATA_DIR = "/home/longzhao/DonkeySim/DonkeySimLinux/donkey_sim_Data"
ASSET_NAME = "sharedassets1.assets"


@dataclass
class ColliderPatch:
    path_id: int
    game_object: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]


@dataclass
class TransformPatch:
    path_id: int
    game_object: str
    local_position: Tuple[float, float, float]


def _load_env(asset_path: str):
    env = UnityPy.load(asset_path)
    object_map = {obj.path_id: obj for obj in env.objects}
    return env, object_map


def _game_object_name(object_map, game_object_ptr) -> Optional[str]:
    if game_object_ptr is None:
        return None
    path_id = getattr(game_object_ptr, "path_id", 0)
    if not path_id:
        return None
    obj = object_map.get(path_id)
    if obj is None:
        return None
    data = obj.read()
    return getattr(data, "m_Name", None)


def _find_body_box_collider(env, object_map) -> ColliderPatch:
    for obj in env.objects:
        if obj.type.name != "BoxCollider":
            continue
        data = obj.read()
        go_name = _game_object_name(object_map, getattr(data, "m_GameObject", None))
        if go_name != "body":
            continue
        center = data.m_Center
        size = data.m_Size
        return ColliderPatch(
            path_id=obj.path_id,
            game_object=go_name,
            center=(float(center.x), float(center.y), float(center.z)),
            size=(float(size.x), float(size.y), float(size.z)),
        )
    raise RuntimeError("Could not find BoxCollider on GameObject 'body'")


def _find_lidar_transform(env, object_map) -> TransformPatch:
    for obj in env.objects:
        if obj.type.name != "Transform":
            continue
        data = obj.read()
        go_ptr = getattr(data, "m_GameObject", None)
        go_name = _game_object_name(object_map, go_ptr)
        if go_name != "Lidar":
            continue
        pos = data.m_LocalPosition
        return TransformPatch(
            path_id=obj.path_id,
            game_object=go_name,
            local_position=(float(pos.x), float(pos.y), float(pos.z)),
        )
    raise RuntimeError("Could not find Transform on GameObject 'Lidar'")


def _apply_patch(
    asset_path: str,
    body_center_y: float,
    body_size_y: float,
    lidar_local_y: Optional[float],
    dry_run: bool,
) -> None:
    env, object_map = _load_env(asset_path)
    body_before = _find_body_box_collider(env, object_map)
    lidar_before = _find_lidar_transform(env, object_map)

    print("Before:")
    print(
        f"  body BoxCollider path={body_before.path_id} "
        f"center={body_before.center} size={body_before.size}"
    )
    print(
        f"  lidar Transform path={lidar_before.path_id} "
        f"local_position={lidar_before.local_position}"
    )

    body_obj = object_map[body_before.path_id]
    body_data = body_obj.read()
    body_data.m_Center.y = float(body_center_y)
    body_data.m_Size.y = float(body_size_y)
    body_data.save()

    if lidar_local_y is not None:
        lidar_obj = object_map[lidar_before.path_id]
        lidar_data = lidar_obj.read()
        lidar_data.m_LocalPosition.y = float(lidar_local_y)
        lidar_data.save()

    if dry_run:
        print("Dry run only. No files written.")
        return

    backup_suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{asset_path}.bak_{backup_suffix}"
    shutil.copy2(asset_path, backup_path)
    print(f"Backup: {backup_path}")

    out_dir = tempfile.mkdtemp(prefix="donkeysim_lidar_patch_")
    env.save(out_path=out_dir)
    patched_path = os.path.join(out_dir, ASSET_NAME)
    if not os.path.exists(patched_path):
        raise RuntimeError(f"Patched asset missing: {patched_path}")
    shutil.copy2(patched_path, asset_path)

    verify_env, verify_map = _load_env(asset_path)
    body_after = _find_body_box_collider(verify_env, verify_map)
    lidar_after = _find_lidar_transform(verify_env, verify_map)

    print("After:")
    print(
        f"  body BoxCollider path={body_after.path_id} "
        f"center={body_after.center} size={body_after.size}"
    )
    print(
        f"  lidar Transform path={lidar_after.path_id} "
        f"local_position={lidar_after.local_position}"
    )
    print("DonkeySim asset patched. Restart the simulator to pick up the change.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-data-dir",
        default=DEFAULT_SIM_DATA_DIR,
        help="Path to donkey_sim_Data directory",
    )
    parser.add_argument(
        "--body-center-y",
        type=float,
        default=0.6,
        help="New BoxCollider center.y for GameObject 'body'",
    )
    parser.add_argument(
        "--body-size-y",
        type=float,
        default=1.2,
        help="New BoxCollider size.y for GameObject 'body'",
    )
    parser.add_argument(
        "--lidar-local-y",
        type=float,
        default=None,
        help="Optional prefab-default Lidar localPosition.y override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and patch in memory without writing the asset",
    )
    args = parser.parse_args()

    asset_path = os.path.join(args.sim_data_dir, ASSET_NAME)
    if not os.path.exists(asset_path):
        raise FileNotFoundError(asset_path)

    _apply_patch(
        asset_path=asset_path,
        body_center_y=float(args.body_center_y),
        body_size_y=float(args.body_size_y),
        lidar_local_y=args.lidar_local_y,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
