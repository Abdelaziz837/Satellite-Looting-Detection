# Egyptian Archaeological Site Looting Dataset

![Project Banner](data/NotLooted/NotLooted105.jpg)

[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/abdelazizamr837/egyptian-archaeological-site-looting) 
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

## Project Overview

This project focuses on identifying archaeological looting pits in Egypt using satellite imagery. Following the security vacuum of 2011, many historical sites were subjected to illegal excavations. These "pits" are visible from space, and this dataset was created to provide a foundation for automated detection.

## Dataset Details

The dataset consists of **508 image patches** categorized into two classes:
* **Looted:** Image patches showing clear evidence of illegal excavation pits.
* **Not Looted:** Image patches showing pristine terrain or natural desert features.

**Dataset Link:** [Access on Kaggle](https://www.kaggle.com/datasets/abdelazizamr837/egyptian-archaeological-site-looting)

## Data Sourcing & Methodology

* **Source:** Captured from **Google Earth Pro’s** historical imagery archive.
* **Timeframe:** Focused on imagery from **2010 to 2016**.
* **Locations:** Key archaeological sites including **Dashur, Lisht, and Saqqara**.

---

## Interactive Monitoring System Architecture

The application is built on a production-grade, decoupled cloud architecture designed to keep binary files and secrets completely out of source control.

```text
[ Client Browser ] <───( https / HTML / Leaflet.js Map )───> [ GitHub Pages ]
       │                                                           │
       │ (GET /predict-coordinates)                                │
       ▼                                                           ▼
[ Google Cloud Run ] (FastAPI Backend Docker Container)
       │
       ├───( Securely fetches 5-fold weights on boot )───> [ AWS S3 Bucket ]
       │
       └───( Fetches 256x256 satellite patch by coords )──> [ Mapbox API ]
```

* **Frontend (UI):** Hosted on GitHub Pages. Features an interactive satellite map centered on Egypt, bounded to geographic borders, with markers pointing to the top 10 target archaeological sites.
* **Backend (REST API):** A Dockerized FastAPI web service hosted on Google Cloud Run.
* **Model Weights Storage:** Kept private and secure on AWS S3. On container boot, the FastAPI server securely downloads the 5-fold model weights into memory before accepting traffic.
* **Satellite Provider:** Connects to the Mapbox Static Images API dynamically to fetch high-resolution satellite imagery patches on-demand.

### Repository Structure

```text
├── app/
│   ├── main.py               # FastAPI application, routing, and CORS configuration
│   └── model_helper.py       # PyTorch LootingClassifier & AWS S3 download controller
├── Dockerfile                # Production compilation recipe (Port 8080)
├── requirements.txt          # CPU-optimized PyTorch and Python dependencies
├── index.html                # Interactive Leaflet.js frontend map dashboard
├── .gitignore                # Prevents tracking .env and models_ensemble/ folders
├── .gcloudignore             # Optimizes Google Cloud Build source uploads
├── results/
│   └── confusion_matrix.png  # Evaluation confusion matrix visualization
└── README.md                 # Project documentation
```

---

## Model Performance & Evaluation

The model predictions were evaluated across 504 validation samples using out-of-fold cross-validation. By applying an optimized decision threshold of 0.4007 (to reduce false negatives and prioritize site protection), the classification balance of the ResNet-50 ensemble is as follows:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~83.5% (421 / 504) |
| **Precision (Looted)** | ~83.5% |
| **Recall (Looted)** | ~84.4% |
| **F1-Score (Looted)** | ~83.9% |

### Optimized Confusion Matrix (Threshold: 0.4007)

Below is the confusion matrix generated from the validation results:

![Confusion Matrix](results/confusion_matrix.png)

* **True Negatives** (NotLooted correctly identified): 204
* **True Positives** (Looted correctly identified): 217
* **False Negatives** (Looted missed): 40
* **False Positives** (NotLooted flagged as Looted): 43

---

## API Documentation & Usage

The API is fully documented and interactive out-of-the-box. Once deployed, you can access the interactive Swagger UI documentation at: `https://<your-cloud-run-url>/docs`

### 1. Scan by Coordinates (GET)

Fetches a high-resolution satellite patch from Mapbox centered on the coordinates and runs ensemble inference.

* **Endpoint:** `GET /predict-coordinates`
* **Parameters:** `lat` (float), `lon` (float), `zoom` (int, default: 16)

#### Sample Response:
```json
{
  "latitude": 29.8712,
  "longitude": 31.2140,
  "prediction": "Looted",
  "probabilities": {
    "NotLooted": 0.1245,
    "Looted": 0.8755
  },
  "threshold_applied": 0.4007,
  "scanned_image_url": "https://api.mapbox.com/..."
}
```

### 2. Scan by Uploaded Image (POST)

Accepts a raw binary satellite patch image and runs ensemble inference.

* **Endpoint:** `POST /predict`
* **Content-Type:** `multipart/form-data`
* **Payload:** `file` (image format: PNG, JPG, or JPEG)

#### Sample Response:
```json
{
  "filename": "satellite_patch.jpg",
  "prediction": "NotLooted",
  "probabilities": {
    "NotLooted": 0.9421,
    "Looted": 0.0579
  },
  "threshold_applied": 0.4007
}
```

---

## Local Development & Setup

### Prerequisites
* Python 3.10
* An AWS account with an S3 Bucket (to store model weights)
* A Mapbox Account (for the public access token)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/abdelazizamr837/Satellite-Looting-Detection.git
cd Satellite-Looting-Detection
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Create a `.env` file in your root folder:
```text
S3_BUCKET_NAME=your-s3-bucket-name
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
AWS_REGION=us-east-1
MAPBOX_ACCESS_TOKEN=your-mapbox-public-token
```

### 3. Run the Server
Launch the FastAPI development server:
```bash
uvicorn app.main:app --reload
```
> **Note:** If the 5 `.pth` model weight files are already present in your local `models_ensemble/` folder, the server will load them directly. If not, it will connect to your S3 bucket and download them automatically.

---

## License

This project and the provided dataset are licensed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License. See the LICENSE file for details.

**Dataset and Model Created by:** Abdelaziz Amr
