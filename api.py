import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from predict import predict_reciter

app = FastAPI(
    title="Quran Reciter Identification API",
    description="REST API for classifying Quran reciters from audio clips using PyTorch CNN",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Quran Reciter Identification API",
        "supported_classes": 12,
        "endpoint": "/predict (POST file upload)"
    }


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload (WAV, MP3, OGG, M4A, FLAC)
    and returns predicted Qari/Reciter along with confidence scores.
    """
    allowed_extensions = [".wav", ".mp3", ".ogg", ".m4a", ".flac"]
    file_ext = os.path.splitext(file.filename)[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{file_ext}'. Allowed: {allowed_extensions}"
        )

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
            temp_audio.write(contents)
            temp_audio_path = temp_audio.name

        try:
            result = predict_reciter(temp_audio_path)
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        return JSONResponse(content={
            "predicted_class": result["predicted_class"],
            "predicted_name_en": result["predicted_name_en"],
            "predicted_name_ar": result["predicted_name_ar"],
            "description": result["description"],
            "confidence": result["confidence"],
            "probabilities": {
                k: {
                    "name_en": v["name_en"],
                    "name_ar": v["name_ar"],
                    "probability": v["probability"]
                }
                for k, v in result["probabilities"].items()
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
