from ground_truth_loader import load_ground_truth, get_actual_from_image
from backend_inference import predict_bone_age_from_bytes

# Load CSV once
ground_truth = load_ground_truth("boneage-training-dataset.csv")

# Load image
image_path = "images\\1380.png"
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Get actual values
actual = get_actual_from_image(image_path, ground_truth)

# Predict
prediction = predict_bone_age_from_bytes(
    image_bytes=image_bytes,
    gender=actual["gender"]
)

# Compare
print("Actual Bone Age:", actual["bone_age"])
print("Predicted Bone Age:", prediction["bone_age_months"])
print(
    "Absolute Error:",
    abs(actual["bone_age"] - prediction["bone_age_months"])
)
