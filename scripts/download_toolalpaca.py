#!/usr/bin/env python3
"""
Utility script to clone the ToolAlpaca repository if not already present.
"""

import os
import subprocess
import sys


def ensure_toolalpaca(target_dir: str = "ToolAlpaca") -> str:
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "data", "train_data.json")):
        print(f"ToolAlpaca dataset already exists at {os.path.abspath(target_dir)}")
        return os.path.join(target_dir, "data", "train_data.json")

    print(f"Cloning ToolAlpaca into {target_dir}...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tangqiaoyu/ToolAlpaca.git", target_dir], check=True)
    json_path = os.path.join(target_dir, "data", "train_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Failed to locate {json_path} after cloning.")
    print("ToolAlpaca downloaded successfully.")
    return json_path


if __name__ == "__main__":
    ensure_toolalpaca()
