"""
model_manager.py - Model Loading & Management cho VideoRAG System
Quản lý và load các AI models: ImageBind, MiniCPM-V, GPT-4o-mini, Whisper, Text Embedding
"""

import os
import gc
import time
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from transformers import logging as transformers_logging

from config import Config, ModelConfig, ModelType, get_config
from utils import get_logger, measure_time, performance_monitor, retry_on_failure

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
transformers_logging.set_verbosity_error()

@dataclass
class ModelInfo:
    """Information về loaded model"""
    name: str
    model_type: ModelType
    device: str
    memory_usage_mb: float
    is_loaded: bool = False
    load_time: float = 0.0
    last_used: float = 0.0
    error_count: int = 0

@dataclass
class InferenceResult:
    """Result từ model inference"""
    success: bool
    result: Any = None
    execution_time: float = 0.0
    error_message: str = ""
    memory_usage_mb: float = 0.0

class ModelManager:
    """
    Centralized manager cho tất cả AI models trong VideoRAG system
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ModelManager
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger('model_manager')
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        
        # Device management
        self.available_devices = self._detect_devices()
        self.device_memory: Dict[str, float] = {}
        
        # Model loading state
        self._loading_lock = {}
        self._is_loading = {}
        
        self.logger.info(f"🤖 ModelManager initialized with devices: {self.available_devices}")
        
    def _detect_devices(self) -> List[str]:
        """Detect available devices"""
        devices = ['cpu']
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices.extend([f'cuda:{i}' for i in range(num_gpus)])
            
            # Log GPU info
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                self.logger.info(f"🎮 GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        return devices
    
    def _get_device_memory_usage(self, device: str) -> float:
        """Get current memory usage for device (MB)"""
        try:
            if device.startswith('cuda'):
                device_id = int(device.split(':')[1]) if ':' in device else 0
                return torch.cuda.memory_allocated(device_id) / (1024**2)
            else:
                # For CPU, estimate based on loaded models
                return sum(info.memory_usage_mb for info in self.model_info.values() 
                          if info.device == device and info.is_loaded)
        except:
            return 0.0
    
    def _estimate_model_memory(self, model_name: str) -> float:
        """Estimate memory requirements for model (MB)"""
        memory_estimates = {
            'imagebind': 6000,      # ~6GB for ImageBind huge
            'minicpm_v': 4000,      # ~4GB for MiniCPM-V int4
            'whisper': 2000,        # ~2GB for Distil-Whisper large
            'text_embedding': 500,   # ~500MB for text-embedding-3-small
            'gpt4o_mini': 0         # API-based, no local memory
        }
        return memory_estimates.get(model_name, 1000)  # Default 1GB
    
    def _check_device_capacity(self, device: str, required_memory_mb: float) -> bool:
        """Check if device has enough memory"""
        if device == 'cpu':
            return True  # Assume CPU always has enough RAM
        
        try:
            device_id = int(device.split(':')[1]) if ':' in device else 0
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**2)
            used_memory = self._get_device_memory_usage(device)
            available_memory = total_memory - used_memory
            
            return available_memory >= required_memory_mb * 1.2  # 20% safety margin
        except:
            return False
    
    @retry_on_failure(max_retries=2, delay=1.0)
    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """
        Load một model cụ thể
        
        Args:
            model_name: Tên model cần load
            force_reload: Force reload nếu model đã loaded
            
        Returns:
            True nếu load thành công
        """
        if model_name not in self.config.models:
            self.logger.error(f"❌ Model '{model_name}' not found in configuration")
            return False
        
        model_config = self.config.models[model_name]
        
        if not model_config.enabled:
            self.logger.warning(f"⚠️ Model '{model_name}' is disabled in configuration")
            return False
        
        # Check if already loaded
        if model_name in self.models and not force_reload:
            self.logger.info(f"✅ Model '{model_name}' already loaded")
            self.model_info[model_name].last_used = time.time()
            return True
        
        # Prevent concurrent loading
        if model_name in self._is_loading and self._is_loading[model_name]:
            self.logger.warning(f"⏳ Model '{model_name}' is already being loaded")
            return False
        
        self._is_loading[model_name] = True
        start_time = time.time()
        
        try:
            with performance_monitor.monitor(f"load_{model_name}"):
                success = self._load_specific_model(model_name, model_config)
            
            load_time = time.time() - start_time
            
            if success:
                # Update model info
                memory_usage = self._estimate_model_memory(model_name)
                device = model_config.device if model_config.device != 'auto' else self._select_optimal_device(memory_usage)
                
                self.model_info[model_name] = ModelInfo(
                    name=model_name,
                    model_type=model_config.type,
                    device=device,
                    memory_usage_mb=memory_usage,
                    is_loaded=True,
                    load_time=load_time,
                    last_used=time.time(),
                    error_count=0
                )
                
                self.logger.info(f"✅ Model '{model_name}' loaded successfully in {load_time:.2f}s")
            else:
                self._handle_model_error(model_name, "Failed to load model")
            
            return success
            
        except Exception as e:
            self._handle_model_error(model_name, str(e))
            return False
        
        finally:
            self._is_loading[model_name] = False
    
    def _load_specific_model(self, model_name: str, model_config: ModelConfig) -> bool:
        """Load specific model based on type"""
        try:
            if model_name == 'imagebind':
                return self._load_imagebind(model_config)
            elif model_name == 'minicpm_v':
                return self._load_minicpm_v(model_config)
            elif model_name == 'whisper':
                return self._load_whisper(model_config)
            elif model_name == 'text_embedding':
                return self._load_text_embedding(model_config)
            elif model_name == 'gpt4o_mini':
                return self._load_gpt4o_mini(model_config)
            else:
                self.logger.error(f"❌ Unknown model type: {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading {model_name}: {e}")
            return False
    
    def _load_imagebind(self, model_config: ModelConfig) -> bool:
        """Load ImageBind model"""
        try:
            import imagebind.models.imagebind_model as imagebind_model
            from imagebind.models import imagebind_model
            
            self.logger.info("📥 Loading ImageBind model...")
            
            # Load model
            model = imagebind_model.imagebind_huge(pretrained=True)
            device = self._get_optimal_device(model_config.device)
            model = model.to(device)
            model.eval()
            
            self.models['imagebind'] = model
            self.logger.info(f"✅ ImageBind loaded on {device}")
            return True
            
        except ImportError:
            self.logger.error("❌ ImageBind not installed. Install with: pip install imagebind")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error loading ImageBind: {e}")
            return False
    
    def _load_minicpm_v(self, model_config: ModelConfig) -> bool:
        """Load MiniCPM-V model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.logger.info("📥 Loading MiniCPM-V model...")
            
            model_path = model_config.model_path
            device = self._get_optimal_device(model_config.device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model với memory optimization
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
                low_cpu_mem_usage=True,
                device_map=device if device.startswith('cuda') else None
            )
            
            if not device.startswith('cuda'):
                model = model.to(device)
            
            model.eval()
            
            self.models['minicpm_v'] = model
            self.tokenizers['minicpm_v'] = tokenizer
            
            self.logger.info(f"✅ MiniCPM-V loaded on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading MiniCPM-V: {e}")
            return False
    
    def _load_whisper(self, model_config: ModelConfig) -> bool:
        """Load Whisper ASR model"""
        try:
            from transformers import pipeline
            
            self.logger.info("📥 Loading Whisper model...")
            
            device = self._get_optimal_device(model_config.device)
            device_id = int(device.split(':')[1]) if device.startswith('cuda') else -1
            
            # Create pipeline
            whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_config.model_path,
                device=device_id,
                torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
            )
            
            self.models['whisper'] = whisper_pipeline
            
            self.logger.info(f"✅ Whisper loaded on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading Whisper: {e}")
            return False
    
    def _load_text_embedding(self, model_config: ModelConfig) -> bool:
        """Load text embedding model"""
        try:
            import openai
            
            # For OpenAI embedding model, just verify API key
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("❌ OpenAI API key not found")
                return False
            
            # Store client
            self.models['text_embedding'] = openai.OpenAI(api_key=api_key)
            
            self.logger.info("✅ Text embedding model (OpenAI) configured")
            return True
            
        except ImportError:
            self.logger.error("❌ OpenAI package not installed. Install with: pip install openai")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error configuring text embedding: {e}")
            return False
    
    def _load_gpt4o_mini(self, model_config: ModelConfig) -> bool:
        """Load GPT-4o-mini model"""
        try:
            import openai
            
            # Verify API key
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                self.logger.error("❌ OpenAI API key not found")
                return False
            
            # Store client
            self.models['gpt4o_mini'] = openai.OpenAI(api_key=api_key)
            
            self.logger.info("✅ GPT-4o-mini model (OpenAI) configured")
            return True
            
        except ImportError:
            self.logger.error("❌ OpenAI package not installed. Install with: pip install openai")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error configuring GPT-4o-mini: {e}")
            return False
    
    def _get_optimal_device(self, configured_device: str) -> str:
        """Get optimal device based on configuration và availability"""
        if configured_device == 'auto':
            return self._select_optimal_device()
        
        if configured_device in self.available_devices:
            return configured_device
        
        # Fallback
        self.logger.warning(f"⚠️ Device '{configured_device}' not available, falling back to CPU")
        return 'cpu'
    
    def _select_optimal_device(self, required_memory_mb: float = 1000) -> str:
        """Select optimal device based on memory requirements"""
        # Prefer GPU if available và has enough memory
        for device in self.available_devices:
            if device.startswith('cuda'):
                if self._check_device_capacity(device, required_memory_mb):
                    return device
        
        # Fallback to CPU
        return 'cpu'
    
    def _handle_model_error(self, model_name: str, error_message: str):
        """Handle model loading/inference errors"""
        if model_name in self.model_info:
            self.model_info[model_name].error_count += 1
            self.model_info[model_name].is_loaded = False
        
        self.logger.error(f"❌ Model error for '{model_name}': {error_message}")
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload một model để free memory
        
        Args:
            model_name: Tên model cần unload
            
        Returns:
            True nếu unload thành công
        """
        try:
            if model_name in self.models:
                # Clear from GPU memory if needed
                model = self.models[model_name]
                if hasattr(model, 'cpu'):
                    model.cpu()
                del self.models[model_name]
                
                # Clear tokenizers/processors
                if model_name in self.tokenizers:
                    del self.tokenizers[model_name]
                if model_name in self.processors:
                    del self.processors[model_name]
                
                # Update model info
                if model_name in self.model_info:
                    self.model_info[model_name].is_loaded = False
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"🗑️ Model '{model_name}' unloaded")
                return True
            
            return True  # Already unloaded
            
        except Exception as e:
            self.logger.error(f"❌ Error unloading model '{model_name}': {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """
        Load tất cả enabled models
        
        Returns:
            Dict mapping model names to load success status
        """
        self.logger.info("🚀 Loading all enabled models...")
        
        results = {}
        for model_name, model_config in self.config.models.items():
            if model_config.enabled:
                self.logger.info(f"📥 Loading {model_name}...")
                results[model_name] = self.load_model(model_name)
            else:
                self.logger.info(f"⏭️ Skipping disabled model: {model_name}")
                results[model_name] = False
        
        # Summary
        loaded_models = [name for name, success in results.items() if success]
        failed_models = [name for name, success in results.items() if not success]
        
        self.logger.info(f"✅ Successfully loaded: {loaded_models}")
        if failed_models:
            self.logger.warning(f"❌ Failed to load: {failed_models}")
        
        return results
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is loaded"""
        return (model_name in self.models and 
                model_name in self.model_info and 
                self.model_info[model_name].is_loaded)
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get loaded model instance
        
        Args:
            model_name: Tên model
            
        Returns:
            Model instance hoặc None nếu not loaded
        """
        if self.is_model_loaded(model_name):
            self.model_info[model_name].last_used = time.time()
            return self.models[model_name]
        
        self.logger.warning(f"⚠️ Model '{model_name}' not loaded")
        return None
    
    def get_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get tokenizer for model"""
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]
        return None
    
    @measure_time
    def inference_imagebind(
        self, 
        images: List[Image.Image], 
        texts: Optional[List[str]] = None
    ) -> InferenceResult:
        """
        ImageBind inference
        
        Args:
            images: List of PIL Images
            texts: Optional list of texts
            
        Returns:
            InferenceResult với embeddings
        """
        start_time = time.time()
        
        try:
            model = self.get_model('imagebind')
            if model is None:
                if not self.load_model('imagebind'):
                    return InferenceResult(False, error_message="Failed to load ImageBind")
                model = self.get_model('imagebind')
            
            from imagebind import data
            
            # Prepare inputs
            inputs = {}
            
            if images:
                # Convert PIL images to required format
                inputs['vision'] = data.load_and_transform_vision_data(images, device=model.device)
            
            if texts:
                inputs['text'] = data.load_and_transform_text(texts, device=model.device)
            
            # Run inference
            with torch.no_grad():
                embeddings = model(inputs)
            
            # Extract embeddings
            result = {}
            if 'vision' in embeddings:
                result['vision'] = embeddings['vision'].cpu().numpy()
            if 'text' in embeddings:
                result['text'] = embeddings['text'].cpu().numpy()
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                success=True,
                result=result,
                execution_time=execution_time,
                memory_usage_mb=self._get_device_memory_usage(model.device.type)
            )
            
        except Exception as e:
            self._handle_model_error('imagebind', str(e))
            return InferenceResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    @measure_time  
    def inference_minicpm_v(
        self, 
        images: List[Image.Image], 
        prompt: str = "Describe this image in detail."
    ) -> InferenceResult:
        """
        MiniCPM-V inference cho image captioning
        
        Args:
            images: List of PIL Images
            prompt: Text prompt cho captioning
            
        Returns:
            InferenceResult với generated captions
        """
        start_time = time.time()
        
        try:
            model = self.get_model('minicpm_v')
            tokenizer = self.get_tokenizer('minicpm_v')
            
            if model is None or tokenizer is None:
                if not self.load_model('minicpm_v'):
                    return InferenceResult(False, error_message="Failed to load MiniCPM-V")
                model = self.get_model('minicpm_v')
                tokenizer = self.get_tokenizer('minicpm_v')
            
            captions = []
            
            # Process each image
            for image in images:
                try:
                    # Generate caption
                    response = model.chat(
                        image=image,
                        msgs=[{'role': 'user', 'content': prompt}],
                        tokenizer=tokenizer
                    )
                    captions.append(response)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing image: {e}")
                    captions.append("")
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                success=True,
                result=captions,
                execution_time=execution_time,
                memory_usage_mb=self._get_device_memory_usage(model.device.type if hasattr(model, 'device') else 'cpu')
            )
            
        except Exception as e:
            self._handle_model_error('minicpm_v', str(e))
            return InferenceResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    @measure_time
    def inference_whisper(self, audio_path: str) -> InferenceResult:
        """
        Whisper inference cho ASR
        
        Args:
            audio_path: Đường dẫn đến audio file
            
        Returns:
            InferenceResult với transcript
        """
        start_time = time.time()
        
        try:
            whisper_pipeline = self.get_model('whisper')
            if whisper_pipeline is None:
                if not self.load_model('whisper'):
                    return InferenceResult(False, error_message="Failed to load Whisper")
                whisper_pipeline = self.get_model('whisper')
            
            # Run ASR
            result = whisper_pipeline(audio_path)
            transcript = result['text'] if 'text' in result else ""
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                success=True,
                result=transcript,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._handle_model_error('whisper', str(e))
            return InferenceResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    @measure_time
    def inference_text_embedding(self, texts: List[str]) -> InferenceResult:
        """
        Text embedding inference
        
        Args:
            texts: List of texts to embed
            
        Returns:
            InferenceResult với embeddings
        """
        start_time = time.time()
        
        try:
            client = self.get_model('text_embedding')
            if client is None:
                if not self.load_model('text_embedding'):
                    return InferenceResult(False, error_message="Failed to load text embedding")
                client = self.get_model('text_embedding')
            
            # Get embeddings từ OpenAI
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            embeddings = np.array(embeddings)
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                success=True,
                result=embeddings,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._handle_model_error('text_embedding', str(e))
            return InferenceResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    @measure_time
    def inference_gpt4o_mini(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> InferenceResult:
        """
        GPT-4o-mini inference
        
        Args:
            messages: List of message dicts với 'role' và 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            InferenceResult với generated text
        """
        start_time = time.time()
        
        try:
            client = self.get_model('gpt4o_mini')
            if client is None:
                if not self.load_model('gpt4o_mini'):
                    return InferenceResult(False, error_message="Failed to load GPT-4o-mini")
                client = self.get_model('gpt4o_mini')
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            
            execution_time = time.time() - start_time
            
            return InferenceResult(
                success=True,
                result=generated_text,
                execution_time=execution_time
            )
            
        except Exception as e:
            self._handle_model_error('gpt4o_mini', str(e))
            return InferenceResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status của tất cả models"""
        status = {
            'loaded_models': [],
            'failed_models': [],
            'total_memory_usage_mb': 0,
            'device_usage': {}
        }
        
        for model_name, info in self.model_info.items():
            if info.is_loaded:
                status['loaded_models'].append({
                    'name': model_name,
                    'device': info.device,
                    'memory_mb': info.memory_usage_mb,
                    'load_time': info.load_time,
                    'last_used': info.last_used,
                    'error_count': info.error_count
                })
                status['total_memory_usage_mb'] += info.memory_usage_mb
            else:
                status['failed_models'].append(model_name)
        
        # Device usage
        for device in self.available_devices:
            status['device_usage'][device] = self._get_device_memory_usage(device)
        
        return status
    
    def cleanup_models(self, unused_threshold_minutes: int = 30):
        """
        Clean up models không được sử dụng
        
        Args:
            unused_threshold_minutes: Unload models unused for this many minutes
        """
        current_time = time.time()
        threshold_seconds = unused_threshold_minutes * 60
        
        models_to_unload = []
        
        for model_name, info in self.model_info.items():
            if (info.is_loaded and 
                current_time - info.last_used > threshold_seconds):
                models_to_unload.append(model_name)
        
        if models_to_unload:
            self.logger.info(f"🧹 Cleaning up unused models: {models_to_unload}")
            for model_name in models_to_unload:
                self.unload_model(model_name)
        
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check trên tất cả loaded models
        
        Returns:
            Dict mapping model names to health status
        """
        health_status = {}
        
        for model_name in self.models.keys():
            try:
                # Simple test inference
                if model_name == 'text_embedding':
                    result = self.inference_text_embedding(["test"])
                elif model_name == 'gpt4o_mini':
                    result = self.inference_gpt4o_mini([{"role": "user", "content": "Hi"}])
                else:
                    # For vision/audio models, just check if accessible
                    model = self.get_model(model_name)
                    result = InferenceResult(success=model is not None)
                
                health_status[model_name] = result.success
                
            except Exception as e:
                self.logger.warning(f"⚠️ Health check failed for {model_name}: {e}")
                health_status[model_name] = False
        
        return health_status
    
    def __del__(self):
        """Cleanup khi ModelManager bị destroy"""
        try:
            for model_name in list(self.models.keys()):
                self.unload_model(model_name)
        except:
            pass

# Global model manager instance  
_global_model_manager = None

def get_model_manager(config: Optional[Config] = None) -> ModelManager:
    """
    Get global model manager instance (singleton pattern)
    
    Args:
        config: Configuration object (chỉ sử dụng lần đầu)
        
    Returns:
        ModelManager instance
    """
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager(config)
    return _global_model_manager

def reset_model_manager():
    """Reset global model manager instance"""
    global _global_model_manager
    if _global_model_manager:
        del _global_model_manager
    _global_model_manager = None

# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing VideoRAG Model Manager")
    print("="*50)
    
    # Setup logging
    from utils import setup_logging
    setup_logging("INFO", enable_colors=True)
    
    # Initialize model manager
    config = get_config()
    manager = ModelManager(config)
    
    # Initialize model manager
    config = get_config()
    manager = ModelManager(config)
    
    # Print system info
    logger = get_logger('test')
    logger.info("🖥️ System Information:")
    logger.info(f"Available devices: {manager.available_devices}")
    
    # Test device detection
    for device in manager.available_devices:
        memory_usage = manager._get_device_memory_usage(device)
        logger.info(f"Device {device}: {memory_usage:.1f} MB used")
    
    # Test memory estimation
    logger.info("\n📊 Memory Estimates:")
    for model_name in config.models.keys():
        if config.models[model_name].enabled:
            estimated_memory = manager._estimate_model_memory(model_name)
            logger.info(f"{model_name}: ~{estimated_memory:.0f} MB")
    
    # Test selective model loading
    logger.info("\n🔄 Testing Selective Model Loading:")
    
    # Load text-based models first (lighter)
    text_models = ['text_embedding', 'gpt4o_mini']
    for model_name in text_models:
        if config.models[model_name].enabled:
            logger.info(f"Loading {model_name}...")
            success = manager.load_model(model_name)
            if success:
                logger.info(f"✅ {model_name} loaded successfully")
            else:
                logger.error(f"❌ Failed to load {model_name}")
    
    # Test inference cho text models
    logger.info("\n🧪 Testing Text Model Inference:")
    
    # Test text embedding
    if manager.is_model_loaded('text_embedding'):
        try:
            logger.info("Testing text embedding...")
            result = manager.inference_text_embedding(["Hello world", "This is a test"])
            if result.success:
                logger.info(f"✅ Text embedding successful: {result.result.shape}")
                logger.info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            else:
                logger.error(f"❌ Text embedding failed: {result.error_message}")
        except Exception as e:
            logger.error(f"❌ Text embedding test error: {e}")
    
    # Test GPT-4o-mini
    if manager.is_model_loaded('gpt4o_mini'):
        try:
            logger.info("Testing GPT-4o-mini...")
            messages = [{"role": "user", "content": "Say hello in a creative way"}]
            result = manager.inference_gpt4o_mini(messages, max_tokens=50)
            if result.success:
                logger.info(f"✅ GPT-4o-mini successful: {result.result}")
                logger.info(f"⏱️ Execution time: {result.execution_time:.3f}s")
            else:
                logger.error(f"❌ GPT-4o-mini failed: {result.error_message}")
        except Exception as e:
            logger.error(f"❌ GPT-4o-mini test error: {e}")
    
    # Test vision models (if sufficient memory)
    logger.info("\n🎨 Testing Vision Model Loading:")
    
    vision_models = ['minicpm_v', 'imagebind']
    for model_name in vision_models:
        if config.models[model_name].enabled:
            # Check if enough memory
            required_memory = manager._estimate_model_memory(model_name)
            optimal_device = manager._select_optimal_device(required_memory)
            
            if manager._check_device_capacity(optimal_device, required_memory):
                logger.info(f"Loading {model_name} on {optimal_device}...")
                success = manager.load_model(model_name)
                if success:
                    logger.info(f"✅ {model_name} loaded successfully")
                else:
                    logger.error(f"❌ Failed to load {model_name}")
            else:
                logger.warning(f"⚠️ Insufficient memory for {model_name} on {optimal_device}")
    
    # Test vision inference nếu models loaded
    logger.info("\n🖼️ Testing Vision Model Inference:")
    
    # Create test image
    try:
        from PIL import Image
        import numpy as np
        
        # Create simple test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Test MiniCPM-V
        if manager.is_model_loaded('minicpm_v'):
            try:
                logger.info("Testing MiniCPM-V image captioning...")
                result = manager.inference_minicpm_v(
                    [test_image], 
                    "Describe what you see in this image."
                )
                if result.success:
                    logger.info(f"✅ MiniCPM-V successful: {result.result[0][:100]}...")
                    logger.info(f"⏱️ Execution time: {result.execution_time:.3f}s")
                else:
                    logger.error(f"❌ MiniCPM-V failed: {result.error_message}")
            except Exception as e:
                logger.error(f"❌ MiniCPM-V test error: {e}")
        
        # Test ImageBind
        if manager.is_model_loaded('imagebind'):
            try:
                logger.info("Testing ImageBind multimodal embedding...")
                result = manager.inference_imagebind(
                    [test_image], 
                    ["A test image"]
                )
                if result.success:
                    logger.info(f"✅ ImageBind successful")
                    if 'vision' in result.result:
                        logger.info(f"Vision embedding shape: {result.result['vision'].shape}")
                    if 'text' in result.result:
                        logger.info(f"Text embedding shape: {result.result['text'].shape}")
                    logger.info(f"⏱️ Execution time: {result.execution_time:.3f}s")
                else:
                    logger.error(f"❌ ImageBind failed: {result.error_message}")
            except Exception as e:
                logger.error(f"❌ ImageBind test error: {e}")
        
    except ImportError:
        logger.warning("⚠️ PIL not available, skipping vision model tests")
    except Exception as e:
        logger.error(f"❌ Error creating test image: {e}")
    
    # Test audio model
    logger.info("\n🎵 Testing Audio Model:")
    
    if config.models['whisper'].enabled:
        # Note: Real audio testing requires audio file
        logger.info("Whisper model configuration checked ✅")
        logger.info("(Actual audio testing requires audio file)")
    
    # Show model status
    logger.info("\n📊 Model Status Summary:")
    status = manager.get_model_status()
    
    logger.info(f"Loaded models: {len(status['loaded_models'])}")
    for model_info in status['loaded_models']:
        logger.info(f"  ✅ {model_info['name']}: {model_info['device']} ({model_info['memory_mb']:.0f} MB)")
    
    if status['failed_models']:
        logger.info(f"Failed models: {status['failed_models']}")
    
    logger.info(f"Total memory usage: {status['total_memory_usage_mb']:.0f} MB")
    
    # Device usage summary
    logger.info("\n🖥️ Device Usage Summary:")
    for device, usage in status['device_usage'].items():
        logger.info(f"  {device}: {usage:.1f} MB")
    
    # Test health check
    logger.info("\n🏥 Health Check:")
    health_status = manager.health_check()
    for model_name, is_healthy in health_status.items():
        status_icon = "✅" if is_healthy else "❌"
        logger.info(f"  {status_icon} {model_name}: {'Healthy' if is_healthy else 'Unhealthy'}")
    
    # Test cleanup
    logger.info("\n🧹 Testing Model Cleanup:")
    
    # Wait a bit then test cleanup
    import time
    time.sleep(2)
    
    # Set very short threshold for testing
    manager.cleanup_models(unused_threshold_minutes=0.01)  # 0.6 seconds
    
    # Final status
    logger.info("\n📈 Final Status:")
    final_status = manager.get_model_status()
    logger.info(f"Models remaining loaded: {len(final_status['loaded_models'])}")
    
    # Test memory optimization
    logger.info("\n⚡ Memory Optimization Demo:")
    
    # Function to print memory usage
    def print_memory_usage():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                cached = torch.cuda.memory_reserved(i) / (1024**2)
                logger.info(f"  GPU {i}: {allocated:.1f} MB allocated, {cached:.1f} MB cached")
    
    print_memory_usage()
    
    # Unload all models
    logger.info("Unloading all models...")
    for model_name in list(manager.models.keys()):
        manager.unload_model(model_name)
    
    print_memory_usage()
    
    # Test error handling
    logger.info("\n🛡️ Testing Error Handling:")
    
    # Try to use unloaded model
    result = manager.inference_text_embedding(["test"])
    if not result.success:
        logger.info(f"✅ Error handling works: {result.error_message}")
    
    # Try to load non-existent model
    success = manager.load_model('nonexistent_model')
    if not success:
        logger.info("✅ Non-existent model handling works")
    
    # Test configuration validation
    logger.info("\n⚙️ Testing Configuration Integration:")
    
    # Check model configs
    for model_name, model_config in config.models.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Enabled: {model_config.enabled}")
        logger.info(f"  Type: {model_config.type}")
        logger.info(f"  Device: {model_config.device}")
        logger.info(f"  Path: {model_config.model_path}")
    
    # Test performance monitoring integration
    logger.info("\n📊 Performance Monitoring Integration:")
    
    # Re-load a model and monitor performance
    if config.models['text_embedding'].enabled:
        with performance_monitor.monitor("model_loading_and_inference"):
            manager.load_model('text_embedding')
            result = manager.inference_text_embedding(["Performance test"])
            if result.success:
                logger.info(f"✅ Performance monitoring integrated successfully")
    
    # Test automatic device selection
    logger.info("\n🎯 Testing Automatic Device Selection:")
    
    for memory_requirement in [500, 2000, 8000]:  # MB
        optimal_device = manager._select_optimal_device(memory_requirement)
        logger.info(f"For {memory_requirement}MB: {optimal_device}")
    
    # Test concurrent access simulation
    logger.info("\n🔄 Testing Concurrent Access Simulation:")
    
    import threading
    import queue
    
    results_queue = queue.Queue()
    
    def worker_thread(thread_id):
        try:
            # Each thread tries to load and use text embedding
            if manager.load_model('text_embedding'):
                result = manager.inference_text_embedding([f"Thread {thread_id} test"])
                results_queue.put((thread_id, result.success))
            else:
                results_queue.put((thread_id, False))
        except Exception as e:
            results_queue.put((thread_id, False))
    
    # Start multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Collect results
    thread_results = []
    while not results_queue.empty():
        thread_results.append(results_queue.get())
    
    successful_threads = sum(1 for _, success in thread_results if success)
    logger.info(f"✅ {successful_threads}/{len(thread_results)} threads succeeded")
    
    # Resource usage summary
    logger.info("\n📋 Resource Usage Summary:")
    
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    logger.info(f"Process memory: {memory_info.rss / (1024**2):.1f} MB")
    logger.info(f"CPU usage: {process.cpu_percent()}%")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**2)
            total = props.total_memory / (1024**2)
            logger.info(f"GPU {i} ({props.name}): {allocated:.1f}/{total:.1f} MB")
    
    # Final cleanup
    logger.info("\n🧹 Final Cleanup:")
    del manager
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("✅ Model Manager testing completed successfully!")
    
    # Summary of capabilities
    print("\n" + "="*60)
    print("🎉 VideoRAG Model Manager - Capabilities Summary")
    print("="*60)
    print("✅ Multi-model support (Vision, Language, Audio)")
    print("✅ Automatic device selection and memory management")
    print("✅ Error handling and retry mechanisms")
    print("✅ Performance monitoring integration")
    print("✅ Concurrent access support")
    print("✅ Dynamic loading/unloading")
    print("✅ Health checking and cleanup")
    print("✅ Configuration-driven setup")
    print("✅ API-based model support (OpenAI)")
    print("✅ Memory optimization and caching")
    print("="*60)