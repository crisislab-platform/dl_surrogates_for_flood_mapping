#!/bin/bash

# Build the Docker image
build() {
    echo "Building Docker image..."
    docker-compose build
}

# Run the container interactively
run() {
    echo "Starting Carlisle container..."
    docker-compose run --rm carlisle
}

# Start Jupyter Lab
jupyter() {
    echo "Starting Jupyter Lab on port 8889..."
    docker-compose up carlisle-jupyter
}

# Stop all containers
stop() {
    echo "Stopping all containers..."
    docker-compose down
}

# Show logs
logs() {
    docker-compose logs -f carlisle-jupyter
}

# Execute command in running container
exec() {
    docker-compose exec carlisle bash
}

case "$1" in
    build)
        build
        ;;
    run)
        run
        ;;
    jupyter)
        jupyter
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    exec)
        exec
        ;;
    *)
        echo "Usage: $0 {build|run|jupyter|stop|logs|exec}"
        echo "  build   - Build the Docker image"
        echo "  run     - Run container interactively"
        echo "  jupyter - Start Jupyter Lab"
        echo "  stop    - Stop all containers"
        echo "  logs    - Show container logs"
        echo "  exec    - Execute bash in running container"
        exit 1
esac
