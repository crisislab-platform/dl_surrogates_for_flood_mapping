#!/bin/bash

echo "Building Carlisle Docker image..."

# Remove duplicate docker-compose files if they exist
if [ -f "docker-compose.yaml" ] && [ -f "docker-compose.yml" ]; then
    echo "Removing duplicate docker-compose.yaml file..."
    rm docker-compose.yaml
fi

docker compose build

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo ""
    echo "To run the container:"
    echo "  ./run.sh"
    echo ""
    echo "To start Jupyter Lab:"
    echo "  ./run.sh jupyter"
    echo ""
    echo "Note: GPU support requires NVIDIA Docker runtime to be installed"
else
    echo "❌ Build failed!"
    echo ""
    echo "If you need GPU support, you can install NVIDIA Docker runtime later."
    echo "The container will still work for CPU-based operations."
    exit 1
fi
