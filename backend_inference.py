# backend_inference.py
import torch
from io import BytesIO
from PIL import Image

from ml_utils import (
    load_classification_model,
    load_regression_model,
    preprocess_for_classification,
    preprocess_for_vit,
    AGE_GROUPS,
    device
)


def predict_bone_age_from_bytes(image_bytes: bytes, gender: str):
    """
    PURE backend inference
    Uses existing reusable functions
    """

    gender_tensor = torch.tensor(
        [1 if gender.lower() == "male" else 0],
        dtype=torch.float32
    ).to(device)

    # Load image
    image_gray = Image.open(BytesIO(image_bytes)).convert("L")
    image_rgb = Image.open(BytesIO(image_bytes)).convert("RGB")

    # ---- Classification ----
    classification_image = preprocess_for_classification(image_gray).to(device)
    classifier = load_classification_model()

    with torch.no_grad():
        class_output = classifier(
            classification_image,
            gender_tensor.unsqueeze(0)
        )
        age_group = torch.argmax(class_output).item()

    # ---- Regression ----
    regression_image = preprocess_for_vit(image_rgb).to(device)
    regressor = load_regression_model(age_group)

    with torch.no_grad():
        bone_age = regressor(
            regression_image,
            gender_tensor.unsqueeze(0)
        ).item()

    min_age, max_age, label = AGE_GROUPS[age_group]

    return {
        "age_group_id": age_group,
        "age_group_label": label,
        "age_range_months": [min_age, max_age],
        "bone_age_months": round(bone_age, 1),
        "bone_age_years": round(bone_age / 12, 1)
    }
