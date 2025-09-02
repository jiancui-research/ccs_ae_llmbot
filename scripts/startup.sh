#!/bin/bash

cd /workspace

# Ensure bash completion and enhanced terminal are available
export TERM=xterm-256color

# Check if measurement_data directory exists and has content
if [ ! -d "measurement_data" ] || [ -z "$(ls -A measurement_data 2>/dev/null)" ]; then
    echo "🔍 Measurement data not found. Downloading..."
    
    # Remove any existing corrupted zip file
    rm -f measurement_data.zip
    
    echo "📥 Downloading measurement data from Google Drive..."
    
    # Try direct download with confirmation parameter
    echo "Attempting direct download with confirmation..."
    wget --no-check-certificate "https://drive.usercontent.google.com/download?id=16y_QrENhjra7lCDrRz7yIJqE-bwrGzXr&export=download&confirm=t" -O measurement_data.zip
    
    # Check if download was successful
    if [ ! -f "measurement_data.zip" ] || [ ! -s "measurement_data.zip" ]; then
        echo "Direct download failed. Trying alternative method..."
        # Alternative: Use gdown if available
        if command -v gdown &> /dev/null; then
            echo "Using gdown as fallback..."
            gdown --id 16y_QrENhjra7lCDrRz7yIJqE-bwrGzXr --output measurement_data.zip
        else
            echo "❌ Download failed. Please download manually or check the Google Drive link."
            echo "Starting JupyterLab without measurement data..."
            exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.token=''
        fi
    fi
    
    # Verify zip file integrity before extraction
    if [ -f "measurement_data.zip" ] && [ -s "measurement_data.zip" ]; then
        echo "🔍 Verifying zip file integrity..."
        if unzip -t measurement_data.zip > /dev/null 2>&1; then
            echo "📦 Extracting measurement data..."
            unzip -q measurement_data.zip
            
            # Clean up zip file to save space
            rm -f measurement_data.zip
            
            echo "✅ Measurement data is now available!"
        else
            echo "❌ Zip file is corrupted. Removing and starting without data..."
            rm -f measurement_data.zip
            echo "Starting JupyterLab without measurement data..."
            exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.token=''
        fi
    else
        echo "❌ Download failed. Starting JupyterLab without measurement data..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.token=''
    fi
else
    echo "✅ Measurement data already exists, skipping download."
fi

echo "🚀 Starting JupyterLab..."
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.allow_origin=* --ServerApp.token=''