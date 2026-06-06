from pathlib import Path
import json
from typing import Any, Dict

def load_json_tree(root_dir: str | Path) -> Dict[str, Any]:
    """
    Recursively find and load all JSON files under root_dir.

    Each JSON file is inserted into a nested dict based on its relative path.
    The filename without .json is used as the final key.

    Example:
        ./subdir1/subdir2/name.json

    Produces:
        {
            "subdir1": {
                "subdir2": {
                    "name": <json object>
                }
            }
        }
    """
    root = Path(root_dir).resolve()
    result: Dict[str, Any] = {}

    for json_path in root.rglob("*.json"):
        relative_path = json_path.relative_to(root)

        # Directory parts plus filename stem
        parts = list(relative_path.parts)
        parts[-1] = json_path.stem

        with json_path.open("r", encoding="utf-8") as f:
            json_obj = json.load(f)

        current = result

        # Create nested directories
        for part in parts[:-1]:
            current = current.setdefault(part, {})

        # Insert JSON object using filename without extension
        current[parts[-1]] = json_obj

    return result


if __name__ == "__main__":
    data = load_json_tree("./output_SVR")

    print(json.dumps(data, indent=2, ensure_ascii=False))
    best = float("-inf")
    best_file = None
    for subdir in data:
        for json_name in data[subdir]:
            json_data = data[subdir][json_name]
            score = json_data["score"]
            if score > best:
                best = score
                best_file = f"{subdir}/{json_name}"
    print(f"Best score: {best}")
    print(f"Best score file: {best_file}")