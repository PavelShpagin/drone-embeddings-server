"""
DINOv2 Embedder for Satellite Imagery
=====================================
Handles deep learning embedding generation using DINOv2 model.
"""

import numpy as np


class TinyDINOEmbedder:
    """DINOv2 embedding model for satellite imagery."""
    
    def __init__(self, model_name: str = "dinov2_vits14"):
        """Initialize the DINOv2 embedding model."""
        try:
            import torch
            import torchvision.transforms as transforms
            import os
            
            os.environ["XFORMERS_DISABLED"] = "1"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
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
            
        except Exception as e:
            print(f"DINOv2 failed: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 384
            self.is_loaded = False
    
    def embed_patch(self, patch: np.ndarray) -> np.ndarray:
        """Generate embedding for image patch using DINOv2."""
        if not self.is_loaded or self.model is None:
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)
        
        try:
            import torch
            from PIL import Image as PILImage
            
            if patch.dtype != np.uint8:
                patch = (patch * 255).astype(np.uint8)
            
            input_tensor = self.transform(PILImage.fromarray(patch)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(input_tensor).cpu().numpy().flatten()
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.random.normal(0, 1, self.embedding_dim).astype(np.float32)