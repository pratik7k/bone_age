import csv
from typing import Dict, Tuple
import os


def load_ground_truth(csv_path: str) -> Dict[str, Dict]:
    """
    Loads CSV into a lookup dictionary.

    CSV format:
    id,boneage,male

    Returns:
    {
      "1377": {"bone_age": 180, "gender": "female"},
      ...
    }
    """
    lookup = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row["id"])
            lookup[image_id] = {
                "bone_age": float(row["boneage"]),
                "gender": "male" if row["male"] == "True" else "female"
            }

    return lookup


def get_actual_from_image(image_path: str, ground_truth: dict):
    """
    Uses image filename as ID to fetch ground truth.

    Example:
    image_path = "images/1377.png"
    """

    image_id = os.path.splitext(os.path.basename(image_path))[0]

    if image_id not in ground_truth:
        raise ValueError(f"No ground truth found for ID: {image_id}")

    return ground_truth[image_id]
