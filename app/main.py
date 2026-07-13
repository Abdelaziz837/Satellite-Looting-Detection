import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query  # Added Query import
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
from app.model_helper import LootingClassifier

# 1. Load the environment variables from the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Global token for Mapbox (defined at module level so all functions can access it)
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

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

# 2. Initialize FastAPI app FIRST so we can configure middleware on it
app = FastAPI(
    title="Archaeological Looting Detection API",
    description="A production-grade REST API utilizing a fine-tuned ResNet-50 model to classify archaeological looting signatures in Egypt.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Add CORS Middleware to the initialized app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/predict-coordinates")
async def predict_coords(lat: float = Query(...), lon: float = Query(...), zoom: int = 16):
    """
    Fetches a satellite image from Mapbox for the given coordinates and runs inference.
    """
    if not classifier:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model is not loaded on the server."
        )

    if not MAPBOX_TOKEN:
        raise HTTPException(status_code=500, detail="Mapbox token not configured on server.")

    # 1. Construct the Mapbox Static Imagery URL (256x256 patch)
    # This fetches a high-res satellite patch centered on the coordinates
    static_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/256x256?access_token={MAPBOX_TOKEN}"
    
    try:
        # 2. Download the image from the Map Provider
        response = requests.get(static_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch satellite imagery from provider.")
            
        image_bytes = response.content
        
        # 3. Run your existing model ensemble inference
        result = classifier.predict(image_bytes)
        
        return {
            "latitude": lat,
            "longitude": lon,
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
            "threshold_applied": result["decision_threshold_applied"],
            "scanned_image_url": static_url  # <-- ADD THIS LINE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring pipeline failure: {str(e)}")