#!/bin/bash
# Simple script to build the Apptainer container

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Check if apptainer is installed
if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer is not installed or not in PATH"
    echo "Please install Apptainer or load the apptainer module"
    exit 1
fi

# Define paths
CONTAINER_DEF="scripts/container.def"
CONTAINER_IMAGE="ttc.sif"

# Build the container with verbose output
echo "Building Apptainer container..."
echo "Using definition file: $CONTAINER_DEF"
echo "Output image: $CONTAINER_IMAGE"
apptainer build $CONTAINER_IMAGE $CONTAINER_DEF

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Container built successfully: $CONTAINER_IMAGE"
    echo "You can now use it with your SLURM scripts in the scripts directory"
else
    echo "Container build failed"
    exit 1
fi 
