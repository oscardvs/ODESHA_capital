#!/bin/bash

# Deployment script for Quant ML Options Trading System
# This script helps with deploying the system in various environments

# Set default values
ENV="local"
DATA_DIR="./data"
CONFIG_FILE="./quant_ml_trader/config/settings.yaml"
USE_DATABASE=false
REBUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --env)
      ENV="$2"
      shift
      shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift
      shift
      ;;
    --config)
      CONFIG_FILE="$2"
      shift
      shift
      ;;
    --use-db)
      USE_DATABASE=true
      shift
      ;;
    --rebuild)
      REBUILD=true
      shift
      ;;
    --help)
      echo "Usage: ./deploy.sh [options]"
      echo "Options:"
      echo "  --env ENV         Deployment environment (local, dev, prod) [default: local]"
      echo "  --data-dir DIR    Data directory path [default: ./data]"
      echo "  --config FILE     Configuration file path [default: ./quant_ml_trader/config/settings.yaml]"
      echo "  --use-db          Enable database service [default: false]"
      echo "  --rebuild         Force rebuild of Docker images [default: false]"
      echo "  --help            Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Set environment-specific variables
case $ENV in
  local)
    echo "Deploying in local environment"
    COMPOSE_FILE="docker-compose.yml"
    ;;
  dev)
    echo "Deploying in development environment"
    COMPOSE_FILE="docker-compose.dev.yml"
    if [ ! -f "$COMPOSE_FILE" ]; then
      echo "Development compose file not found, using default"
      COMPOSE_FILE="docker-compose.yml"
    fi
    ;;
  prod)
    echo "Deploying in production environment"
    COMPOSE_FILE="docker-compose.prod.yml"
    if [ ! -f "$COMPOSE_FILE" ]; then
      echo "Production compose file not found, using default"
      COMPOSE_FILE="docker-compose.yml"
    fi
    ;;
  *)
    echo "Unknown environment: $ENV"
    echo "Using default compose file"
    COMPOSE_FILE="docker-compose.yml"
    ;;
esac

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file not found: $CONFIG_FILE"
    echo "Please create the configuration file and try again"
    exit 1
fi

# Build and start the containers
echo "Starting deployment with compose file: $COMPOSE_FILE"

# Set additional options
COMPOSE_OPTS=""

# Disable database service if not needed
if [ "$USE_DATABASE" = false ]; then
    COMPOSE_OPTS="$COMPOSE_OPTS --scale database=0"
fi

# Force rebuild if requested
if [ "$REBUILD" = true ]; then
    echo "Forcing rebuild of Docker images"
    docker-compose -f "$COMPOSE_FILE" build --no-cache
fi

# Start the services
echo "Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d $COMPOSE_OPTS

# Check if services are running
if [ $? -eq 0 ]; then
    echo "Deployment successful!"
    echo "The dashboard is available at: http://localhost:8501"
else
    echo "Deployment failed. Check the logs for more information:"
    echo "docker-compose -f $COMPOSE_FILE logs"
    exit 1
fi

# Display container status
docker-compose -f "$COMPOSE_FILE" ps
