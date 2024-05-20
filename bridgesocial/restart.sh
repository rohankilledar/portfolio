#!/bin/bash

# Stop the running container
docker stop ocr-fastapi-app
docker rm ocr-fastapi-app

# Remove the old image
docker rmi ocr-fastapi-app

# Build the new image
docker build -t ocr-fastapi-app .

# Run the new container
docker run -d -p 8000:8000 --name ocr-fastapi-app ocr-fastapi-app


# Function to check if the FastAPI application is up
function wait_for_fastapi() {
    local retries=10
    local count=0
    local status=1
    while [ $count -lt $retries ]; do
        echo "Checking if FastAPI is up... (attempt $((count + 1))/$retries)"
        status=$(curl -o /dev/null -s -w "%{http_code}\n" http://localhost:8000/docs)
        if [ "$status" == "200" ]; then
            echo "FastAPI is up and running."
            return 0
        fi
        count=$((count + 1))
        sleep 3
    done
    echo "FastAPI did not start within expected time."
    return 1
}

# Wait for the FastAPI application to be up and running
if wait_for_fastapi; then
    # Open the FastAPI application in the default browser
    open http://localhost:8000/docs
else
    echo "Failed to start FastAPI application."
fi