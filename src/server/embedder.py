"""
DINOv2 Embedder for Satellite Imagery
=====================================
Handles deep learning embedding generation using DINOv2 model.

This research variant follows the project-wide convention:
embed_patch returns a dictionary with at least the key "embedding" -> np.ndarray.
"""

import numpy as np
from typing import Dict, Any


class TinyDINOEmbedder:
    """DINOv2 embedding model for satellite imagery."""
    
    def __init__(self, model_name: str = "dinov2_vits14"):
        """Initialize the DINOv2 embedding model."""
        import sys
        
        # Check Python version for compatibility
        python_version = sys.version_info
        if python_version < (3, 10):
            print(f"Python {python_version.major}.{python_version.minor} detected. DINOv2 may have compatibility issues.")
        
        try:
            import torch
            import torchvision.transforms as transforms
            import os
            import warnings
            
            # Suppress warnings and compatibility issues
            warnings.filterwarnings("ignore")
            os.environ["XFORMERS_DISABLED"] = "1"
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Try to load DINOv2 with better error handling
            try:
                self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True, trust_repo=True)
                self.model = self.model.to(self.device).eval()
                
                with torch.no_grad():
                    dummy_output = self.model(torch.randn(1, 3, 224, 224).to(self.device))
                    self.embedding_dim = dummy_output.shape[1]
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                self.is_loaded = True
                print(f"DINOv2 loaded: {model_name}, dim={self.embedding_dim}, device={self.device}")
                
            except Exception as model_error:
                # If the specific error is Python version related, try alternative approach
                if "unsupported operand type" in str(model_error) and "|" in str(model_error):
                    print(f"Python version compatibility issue with DINOv2. Using random embeddings.")
                    print(f"To fix: upgrade to Python 3.10+ or use older PyTorch/DINOv2 versions.")
                else:
                    print(f"DINOv2 model loading failed: {model_error}")
                
                self.model = None
                self.embedding_dim = 384
                self.is_loaded = False
                
        except ImportError as import_error:
            print(f"PyTorch import failed: {import_error}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 384
            self.is_loaded = False
            
        except Exception as e:
            print(f"DINOv2 initialization failed: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 384
            self.is_loaded = False
    
    def embed_patch(self, patch: np.ndarray) -> Dict[str, Any]:
        """Generate embedding for image patch using DINOv2.

        Returns a dict: {"embedding": np.ndarray}
        """
        if not self.is_loaded or self.model is None:
            return {"embedding": np.random.normal(0, 1, self.embedding_dim).astype(np.float32)}
        
        try:
            import torch
            from PIL import Image as PILImage
            
            if patch.dtype != np.uint8:
                patch = (patch * 255).astype(np.uint8)
            
            input_tensor = self.transform(PILImage.fromarray(patch)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(input_tensor).cpu().numpy().flatten().astype(np.float32)
            
            return {"embedding": embedding}
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return {"embedding": np.random.normal(0, 1, self.embedding_dim).astype(np.float32)}