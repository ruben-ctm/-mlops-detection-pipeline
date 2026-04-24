"""
Description generation module using CLIP
"""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)


class DescriptionGenerator:
    """CLIP-based description generator"""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda"
    ):
        """
        Initialize description generator
        
        Args:
            model_name: HuggingFace CLIP model name
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
        
        # Predefined description templates
        self.templates = [
            "a photo of a {}",
            "a {} in the scene",
            "there is a {} here",
            "this is a {}",
            "{} visible in the image"
        ]
    
    def generate_description(
        self, 
        image: Image.Image,
        candidates: List[str] = None,
        top_k: int = 3
    ) -> str:
        """
        Generate description for cropped object
        
        Args:
            image: PIL Image of cropped object
            candidates: Optional list of candidate descriptions
            top_k: Number of top descriptions to consider
            
        Returns:
            Best description string
        """
        try:
            if candidates is None:
                # Default attribute candidates
                candidates = [
                    "small", "large", "red", "blue", "green", "yellow",
                    "old", "new", "bright", "dark", "colorful",
                    "outdoor", "indoor", "standing", "sitting", "moving"
                ]
            
            # Prepare text prompts
            text_prompts = [f"a {attr} object" for attr in candidates]
            
            # Process inputs
            inputs = self.processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], top_k)
            
            # Build description
            top_attrs = [candidates[idx] for idx in top_indices.cpu().numpy()]
            
            # Simple description: combine top attributes
            if len(top_attrs) >= 2:
                description = f"{top_attrs[0]} and {top_attrs[1]} object"
            else:
                description = f"{top_attrs[0]} object"
            
            return description
            
        except Exception as e:
            logger.error(f"Description generation error: {e}")
            return "object detected"
    
    def classify_attributes(
        self,
        image: Image.Image,
        attribute_categories: dict = None
    ) -> dict:
        """
        Classify multiple attributes of an object
        
        Args:
            image: PIL Image
            attribute_categories: Dict of attribute types and candidates
                e.g., {'size': ['small', 'medium', 'large'],
                       'color': ['red', 'blue', 'green']}
        
        Returns:
            Dict of attribute types to predicted values
        """
        if attribute_categories is None:
            attribute_categories = {
                'size': ['tiny', 'small', 'medium', 'large', 'huge'],
                'color': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
                'state': ['new', 'old', 'clean', 'dirty']
            }
        
        results = {}
        
        try:
            for attr_type, candidates in attribute_categories.items():
                text_prompts = [f"a {cand} object" for cand in candidates]
                
                inputs = self.processor(
                    text=text_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                
                # Get best match
                best_idx = probs[0].argmax().item()
                results[attr_type] = candidates[best_idx]
            
            return results
            
        except Exception as e:
            logger.error(f"Attribute classification error: {e}")
            return {}
