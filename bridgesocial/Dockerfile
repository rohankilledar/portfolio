# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies and Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* 

# Set the working directory
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /app

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Set the environment variable for the OpenAI API key
ENV OPENAI_API_KEY YOUR_OPENAI_API_KEY

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
