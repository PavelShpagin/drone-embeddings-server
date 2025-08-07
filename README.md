# Drone Embeddings Server

Real-time GPS localization using DINOv2 embeddings and satellite imagery.

## Quick Start

### Server Setup

```bash
cd server
python server.py --port 5000
```

### Run Tests

```bash
cd server
python test/test_fetch_gps_device.py
python test/test_simulation.py
```

### Remote Testing (AWS)

```bash
export AWS_SERVER_DNS=your-server-dns.com
python test/test_simulation.py --remote
```

## Key Features

- **Automatic Path Visualization**: `fetch_gps` automatically creates and updates red dot path images
- **Device Integration**: Works with remote device client via TCP sockets
- **Session Management**: Persistent storage of maps, embeddings, and GPS paths
- **Real-time Processing**: Fast GPS coordinate matching using DINOv2 embeddings

## API Endpoints

- `POST /init_map` - Initialize map with satellite imagery and create embeddings
- `POST /fetch_gps` - Get GPS coordinates from drone image (auto-updates visualization)
- `POST /visualize_path` - Return current path visualization image
- `GET /list_sessions` - List all active sessions

## Data Flow

1. **Init Map**: Creates satellite map patches and DINOv2 embeddings
2. **Fetch GPS**: Matches drone image → Returns GPS + Updates path visualization
3. **Path Storage**: Red dot images saved in `data/server_paths/`
4. **Session Persistence**: All data stored in `data/sessions.pkl`

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)
- Google Earth Engine credentials
- FastAPI, torch, PIL, numpy

## Installation

```bash
pip install fastapi uvicorn torch torchvision numpy pillow earthengine-api
```
