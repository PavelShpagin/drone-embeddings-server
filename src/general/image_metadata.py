import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
from datetime import datetime
import re

@dataclass
class DroneTelemetry:
    """Class to hold drone telemetry data."""
    fms: Optional[int] = None
    csv: Optional[int] = None
    fps: Optional[Tuple[int, int]] = None
    height: Optional[float] = None
    tilt: Optional[Tuple[float, float]] = None
    mtc: Optional[int] = None
    flight_mode: Optional[str] = None
    quaternion: Optional[Tuple[float, float, float, float]] = None
    position_2d: Optional[Tuple[float, float]] = None
    acceleration: Optional[Tuple[float, float, float]] = None
    climb_rate: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    coefficient: Optional[float] = None
    # Added ground-truth GPS fields if present in metadata tail
    g_lat: Optional[float] = None
    g_lon: Optional[float] = None

@dataclass
class ImageMetadata:
    """Class to hold all metadata extracted from an image."""
    filename: str
    file_size_bytes: int
    basic_info: Dict
    exif_metadata: Dict
    iptc_xmp_metadata: Dict
    telemetry: Optional[DroneTelemetry] = None
    timestamp: Optional[datetime] = None
    image_binary_size: int = 0
    raw_json_metadata: Optional[List] = None

def parse_header(header_str: str) -> Dict:
    """Parse the header string into a dictionary of values."""
    result = {}
    
    # Extract FMS number
    fms_match = re.search(r'FMS:\s*(\d+)', header_str)
    if fms_match:
        result['fms'] = int(fms_match.group(1))
    
    # Extract CSV number
    csv_match = re.search(r'CSV:\s*(\d+)', header_str)
    if csv_match:
        result['csv'] = int(csv_match.group(1))
    
    # Extract FPS
    fps_match = re.search(r'FPS:\s*(\d+)/(\d+)', header_str)
    if fps_match:
        result['fps'] = (int(fps_match.group(1)), int(fps_match.group(2)))
    
    # Extract Height
    height_match = re.search(r'H:\s*([-+]?\d*\.?\d+)', header_str)
    if height_match:
        result['height'] = float(height_match.group(1))
    
    # Extract Tilt
    tilt_match = re.search(r'TLT:\s*([-+]?\d*\.?\d+)/([-+]?\d*\.?\d+)', header_str)
    if tilt_match:
        result['tilt'] = (float(tilt_match.group(1)), float(tilt_match.group(2)))
    
    # Extract MTC
    mtc_match = re.search(r'MTC:\s*(\d+)', header_str)
    if mtc_match:
        result['mtc'] = int(mtc_match.group(1))
    
    return result

def extract_metadata(image_path: str) -> ImageMetadata:
    """
    Extract all metadata from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageMetadata object containing all extracted metadata
    """
    with open(image_path, 'rb') as f:
        full_content = f.read()

    metadata = ImageMetadata(
        filename=image_path,
        file_size_bytes=len(full_content),
        basic_info={},
        exif_metadata={},
        iptc_xmp_metadata={},
        telemetry=None,
        image_binary_size=len(full_content),
        raw_json_metadata=None
    )

    # Extract basic image info and EXIF/other data using Pillow
    try:
        with Image.open(image_path) as img:
            metadata.basic_info = {
                "image_size": img.size,
                "image_height": img.height,
                "image_width": img.width,
                "image_format": img.format,
                "image_mode": img.mode,
                "image_is_animated": getattr(img, "is_animated", False),
                "frames_in_image": getattr(img, "n_frames", 1)
            }

            # Extract EXIF data
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            value = str(value)
                    metadata.exif_metadata[tag_name] = value

            # Extract IPTC/XMP data
            if "icc_profile" in img.info:
                metadata.iptc_xmp_metadata["icc_profile"] = str(img.info["icc_profile"])
            if "photoshop" in img.info:
                metadata.iptc_xmp_metadata["photoshop"] = str(img.info["photoshop"])

    except Exception as e:
        print(f"Warning: Could not extract basic image info or EXIF/IPTC/XMP data: {e}")

    # Extract JSON from end of file
    search_tail_size = min(len(full_content), 8192)
    tail_content_bytes = full_content[-search_tail_size:]

    try:
        decoded_tail = tail_content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_tail = tail_content_bytes.decode('latin-1')
        except UnicodeDecodeError:
            decoded_tail = None

    if decoded_tail:
        json_end_pos = decoded_tail.rfind(']')
        if json_end_pos != -1:
            open_brackets = 0
            json_start_pos = -1
            for idx in range(json_end_pos, -1, -1):
                if decoded_tail[idx] == ']':
                    open_brackets += 1
                elif decoded_tail[idx] == '[':
                    open_brackets -= 1
                if open_brackets == 0:
                    json_start_pos = idx
                    break

            if json_start_pos != -1:
                try:
                    json_str = decoded_tail[json_start_pos:json_end_pos + 1]
                    metadata_array = json.loads(json_str)
                    metadata.image_binary_size = len(full_content) - search_tail_size + json_start_pos
                    
                    # Store raw JSON metadata
                    metadata.raw_json_metadata = metadata_array
                    
                    # Extract telemetry data from the last few objects
                    telemetry = DroneTelemetry()
                    
                    for item in reversed(metadata_array):
                        if isinstance(item, dict):
                            # Parse header string
                            if 'header' in item:
                                header_data = parse_header(item['header'])
                                telemetry.fms = header_data.get('fms')
                                telemetry.csv = header_data.get('csv')
                                telemetry.fps = header_data.get('fps')
                                telemetry.height = header_data.get('height')
                                telemetry.tilt = header_data.get('tilt')
                                telemetry.mtc = header_data.get('mtc')
                            
                            # Parse flight mode
                            if 'comment' in item and 'FlightMode:' in item['comment']:
                                telemetry.flight_mode = item['comment'].split('FlightMode:')[1].strip()
                            
                            # Parse quaternion
                            if 'quat' in item and isinstance(item['quat'], list) and len(item['quat']) == 4:
                                telemetry.quaternion = tuple(item['quat'])
                            
                            # Parse position
                            if 'pos2D' in item and isinstance(item['pos2D'], list) and len(item['pos2D']) == 2:
                                telemetry.position_2d = tuple(item['pos2D'])
                            
                            # Parse acceleration
                            if 'accel' in item and isinstance(item['accel'], list) and len(item['accel']) == 3:
                                telemetry.acceleration = tuple(item['accel'])
                            
                            # Parse other fields
                            if 'height' in item and isinstance(item['height'], (int, float)):
                                telemetry.height = float(item['height'])
                            if 'climb' in item and isinstance(item['climb'], (int, float)):
                                telemetry.climb_rate = float(item['climb'])
                            if 'dx' in item and isinstance(item['dx'], (int, float)):
                                telemetry.dx = float(item['dx'])
                            if 'dy' in item and isinstance(item['dy'], (int, float)):
                                telemetry.dy = float(item['dy'])
                            if 'coef' in item and isinstance(item['coef'], (int, float)):
                                telemetry.coefficient = float(item['coef'])
                            # Ground-truth GPS if provided by synthetic streams
                            if 'g_lat' in item and isinstance(item['g_lat'], (int, float)):
                                telemetry.g_lat = float(item['g_lat'])
                            if 'g_lon' in item and isinstance(item['g_lon'], (int, float)):
                                telemetry.g_lon = float(item['g_lon'])
                    
                    metadata.telemetry = telemetry
                    
                except json.JSONDecodeError:
                    pass

    return metadata

def calculate_speed_between_images(metadata1: ImageMetadata, metadata2: ImageMetadata) -> Optional[float]:
    """
    Calculate speed between two images based on their positions and timestamps.
    
    Args:
        metadata1: Metadata from first image
        metadata2: Metadata from second image
        
    Returns:
        Speed in meters per second if both images have valid position data,
        None otherwise
    """
    if not all([
        metadata1.telemetry and metadata1.telemetry.position_2d,
        metadata2.telemetry and metadata2.telemetry.position_2d,
        metadata1.telemetry.fms is not None,
        metadata2.telemetry.fms is not None
    ]):
        return None
        
    # Calculate time difference in seconds (assuming FMS numbers are sequential)
    fms_diff = metadata2.telemetry.fms - metadata1.telemetry.fms
    if fms_diff <= 0:
        return None
        
    # Calculate distance using Euclidean distance
    pos1 = np.array(metadata1.telemetry.position_2d)
    pos2 = np.array(metadata2.telemetry.position_2d)
    distance = np.linalg.norm(pos2 - pos1)
    
    # Calculate speed in m/s (assuming FMS numbers represent frames at ~50fps)
    fps = metadata1.telemetry.fps[0] if metadata1.telemetry.fps else 50
    time_diff = fms_diff / fps
    speed = distance / time_diff if time_diff > 0 else None
    
    return speed 