import torch
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.translate import Translator
import uvicorn

# 1. Define the Input Format (Validation)
class TranslationRequest(BaseModel):
    text: str

app = FastAPI(title="Hindi Planner API", version="1.0")

model_instance = None

@app.on_event("startup")
def load_model():
    """Run this when the server starts."""
    global model_instance
    print("  Loading Model... This might take a minute.")
    try:
        if not os.path.exists("checkpoints/vae_epoch_3.pth"):
            print(" Critical: VAE Checkpoint missing!")
        
        # Initialize your Translator class
        model_instance = Translator()
        print(" Model Loaded and Ready!")
    except Exception as e:
        print(f" Failed to load model: {e}")

@app.get("/")
def home():
    return {"status": "running", "docs_url": "/docs"}

@app.post("/translate")
def translate_text(payload: TranslationRequest):
    """The Inference Endpoint"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")
    
    try:
        # Run inference
        hindi_translation = model_instance.translate(payload.text)
        return {
            "input": payload.text,
            "translation": hindi_translation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)