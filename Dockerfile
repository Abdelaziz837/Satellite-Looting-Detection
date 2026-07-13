# 1. Start with an official, lightweight Python base image
FROM python:3.10-slim

# 2. Prevent Python from writing .pyc files and buffering terminal output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /workspace

# 4. Install system tools required for building standard Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your Python application code into the container
COPY app/ ./app/

# 7. Expose the API port (7860 is required by Hugging Face Spaces)
EXPOSE 8080

# 8. Start the FastAPI server using Uvicorn on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]