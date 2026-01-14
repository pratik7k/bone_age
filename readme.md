# Pediatric Bone Age Prediction Service

## Overview
Backend ML service for pediatric bone age estimation using CNN + Vision Transformer.

## Tech Stack
- FastAPI
- PyTorch
- Vision Transformer (ViT)


## Run Backend
```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
