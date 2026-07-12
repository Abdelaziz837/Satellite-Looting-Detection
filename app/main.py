import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.model_helper import LootingClassifier
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)  #

# Global model variable to hold the loaded model weights in memory
classifier = None

# Define the absolute directory path where models will be cached/loaded
ENSEMBLE_DIR = os.path.join(os.path.dirname(__file__), "..", "models_ensemble")

# Create a list containing the absolute file paths for all 5 folds
MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, f"resnet50_fold_{i}.pth") 
    for i in range(1, 6)
]

# Use the modern FastAPI lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("Initializing LootingClassifier...")
    print("Checking local storage (missing weights will be securely downloaded from S3)...")
    try:
        # Code here runs BEFORE the API starts accepting requests (Startup)
        classifier = LootingClassifier(MODEL_PATHS)
        print("Model loaded successfully and ready for inference!")
    except Exception as e:
        print(f"Fatal startup error: {e}")
        raise e
        
    yield  # The server runs and handles requests while paused here
    
    # Code here runs AFTER the API stops accepting requests (Shutdown)
    print("Shutting down and cleaning up resources...")

# Initialize FastAPI app with the lifespan handler
app = FastAPI(
    title="Archaeological Looting Detection API",
    description="A production-grade REST API utilizing a fine-tuned ResNet-50 model to classify archaeological looting signatures in Egypt.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    """Simple API status check endpoint."""
    return {"status": "healthy", "model_loaded": classifier is not None}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (PNG, JPG, JPEG) and returns the model prediction.
    """
    if not classifier:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model is not loaded on the server."
        )

    # 1. Validate file format
    extension = file.filename.split(".")[-1].lower() if file.filename else ""
    if extension not in ["png", "jpg", "jpeg"]:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a PNG, JPG, or JPEG image."
        )
        
    try:
        # 2. Read raw binary image data asynchronously
        image_bytes = await file.read()
        
        # 3. Pass bytes to helper and run inference
        result = classifier.predict(image_bytes)
        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
            "threshold_applied": result["decision_threshold_applied"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failure: {str(e)}")