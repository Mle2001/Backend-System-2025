# custom_imagebind.py
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import requests
from PIL import Image
import numpy as np

class CustomImageBind:
    """Custom ImageBind implementation using CLIP as backbone"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model as backbone
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define modality types (similar to ImageBind)
        self.ModalityType = type('ModalityType', (), {
            'VISION': 'vision',
            'TEXT': 'text',
            'AUDIO': 'audio'
        })
        
    def load_and_transform_vision_data(self, image_paths, device=None):
        """Load and transform image data"""
        if device is None:
            device = self.device
            
        images = []
        for path in image_paths:
            if isinstance(path, str):
                if path.startswith('http'):
                    image = Image.open(requests.get(path, stream=True).raw)
                else:
                    image = Image.open(path)
            else:
                image = path
                
            images.append(image)
            
        # Process images
        inputs = self.processor(images=images, return_tensors="pt")
        return {self.ModalityType.VISION: inputs['pixel_values'].to(device)}
    
    def load_and_transform_text(self, text, device=None):
        """Load and transform text data"""
        if device is None:
            device = self.device
            
        if isinstance(text, str):
            text = [text]
            
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return {self.ModalityType.TEXT: inputs['input_ids'].to(device)}
    
    def encode_vision(self, vision_data):
        """Encode vision data to embeddings"""
        with torch.no_grad():
            if self.ModalityType.VISION in vision_data:
                pixel_values = vision_data[self.ModalityType.VISION]
                vision_outputs = self.model.get_image_features(pixel_values=pixel_values)
                return vision_outputs
        return None
    
    def encode_text(self, text_data):
        """Encode text data to embeddings"""
        with torch.no_grad():
            if self.ModalityType.TEXT in text_data:
                input_ids = text_data[self.ModalityType.TEXT]
                text_outputs = self.model.get_text_features(input_ids=input_ids)
                return text_outputs
        return None
    
    def forward(self, inputs):
        """Forward pass for multiple modalities"""
        outputs = {}
        
        if self.ModalityType.VISION in inputs:
            vision_embeds = self.encode_vision({self.ModalityType.VISION: inputs[self.ModalityType.VISION]})
            outputs[self.ModalityType.VISION] = vision_embeds
            
        if self.ModalityType.TEXT in inputs:
            text_embeds = self.encode_text({self.ModalityType.TEXT: inputs[self.ModalityType.TEXT]})
            outputs[self.ModalityType.TEXT] = text_embeds
            
        return outputs
    
    def __call__(self, inputs):
        """Make the model callable"""
        return self.forward(inputs)

# Data loading utilities (similar to ImageBind data module)
class data:
    @staticmethod
    def load_and_transform_vision_data(image_paths, device="cpu"):
        """Static method for loading vision data"""
        custom_bind = CustomImageBind()
        return custom_bind.load_and_transform_vision_data(image_paths, device)
    
    @staticmethod
    def load_and_transform_text(text_list, device="cpu"):
        """Static method for loading text data"""
        custom_bind = CustomImageBind()
        return custom_bind.load_and_transform_text(text_list, device)

# Model loading function
def imagebind_huge(pretrained=True):
    """Load the custom ImageBind model"""
    return CustomImageBind()

# Create ModalityType class
class ModalityType:
    VISION = "vision"
    TEXT = "text"
    AUDIO = "audio"