#!/bin/bash
# Install dependencies for Drone Embeddings Server - New Architecture
set -e

echo "Installing Drone Embeddings Server dependencies..."

# Update pip
python3 -m pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install torch with CUDA support (if available)
echo "Checking for CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, using CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Verify installation
echo "Verifying installation..."
python3 -c "
import fastapi
import torch
import numpy as np
import PIL
import matplotlib
import pandas
import earthengine as ee
print('✓ All core dependencies imported successfully')

# Check CUDA
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
else:
    print('✓ CPU-only mode (CUDA not available)')
"

echo "✓ Installation completed successfully!"
echo ""
echo "To start the server:"
echo "  python server.py --port 5000"
echo ""
echo "To test the installation:"
echo "  python test_local_simple.py"
