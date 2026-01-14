# api.py
from fastapi import FastAPI, UploadFile, File, Form
from backend_inference import predict_bone_age_from_bytes

app = FastAPI(
    title="Pediatric Bone Age Prediction API",
    version="1.0"
)


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    gender: str = Form(...)
):
    image_bytes = await image.read()

    result = predict_bone_age_from_bytes(
        image_bytes=image_bytes,
        gender=gender
    )

    return {
        "status": "success",
        "data": result
    }


@app.get("/health")
def health():
    return {"status": "ok"}
