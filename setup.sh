#!/bin/bash
# Try installing via apt-get, but also check for ffmpeg as a Python package

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found, installing via apt-get..."
    apt-get update
    apt-get install -y ffmpeg
else
    echo "ffmpeg is already installed."
fi
