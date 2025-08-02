#!/bin/bash
# Google Earth Engine Key Setup Script

echo "Setting up Google Earth Engine credentials..."

# Create secrets directory
mkdir -p secrets

echo "Please paste your Google Earth Engine service account JSON key below:"
echo "Press Ctrl+D when finished pasting"
echo ""

# Use cat to capture multi-line input
cat > secrets/earth-engine-key.json

echo ""
echo "Key saved to secrets/earth-engine-key.json"

# Validate JSON format
if python3 -c "import json; json.load(open('secrets/earth-engine-key.json'))" 2>/dev/null; then
    echo "JSON format is valid"
else
    echo "Invalid JSON format - please check your key"
fi

echo "You can now start the server with: python server.py --port 5000"
