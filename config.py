"""
config.py - Configuration Management cho VideoRAG System
Chứa tất cả các cấu hình về models, database, processing parameters, và distributed settings
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ModelType(Enum):
    """Enum cho các loại model được hỗ trợ"""
    VISION_LANGUAGE = "vision_language"
    EMBEDDING = "embedding"
    AUDIO = "audio"
    LLM = "llm"

class DatabaseType(Enum):
    """Enum cho các loại database được hỗ trợ"""
    VECTOR = "vector"
    GRAPH = "graph"
    METADATA = "metadata"

@dataclass
class ModelConfig:
    """Cấu hình cho từng model"""
    name: str
    type: ModelType
    model_path: str
    device: str = "auto"
    max_memory: Optional[str] = None
    quantization: Optional[str] = None
    batch_size: int = 1
    max_length: Optional[int] = None
    temperature: float = 0.1
    top_p: float = 0.9
    enabled: bool = True

@dataclass
class DatabaseConfig:
    """Cấu hình cho database"""
    type: DatabaseType
    connection_string: str
    collection_name: str
    embedding_dimension: Optional[int] = None
    index_type: str = "HNSW"
    metric: str = "cosine"
    max_connections: int = 100

@dataclass
class ProcessingConfig:
    """Cấu hình cho video processing"""
    frame_sampling_rate: int = 5  # frames per 30-second clip
    detailed_sampling_rate: int = 15  # frames cho detailed analysis
    chunk_duration: int = 30  # seconds
    max_frames_per_video: int = 3600  # constraint cho GPU memory
    supported_formats: List[str] = None
    output_resolution: tuple = (224, 224)
    audio_sample_rate: int = 16000

@dataclass
class RetrievalConfig:
    """Cấu hình cho retrieval system"""
    similarity_threshold: float = 0.7
    top_k_text: int = 10
    top_k_visual: int = 5
    top_k_final: int = 15
    enable_cross_modal: bool = True
    enable_temporal_filtering: bool = True
    relevance_filter_threshold: float = 0.6

@dataclass
class DistributedConfig:
    """Cấu hình cho distributed processing"""
    coordinator_host: str = "localhost"
    coordinator_port: int = 8888
    worker_nodes: List[str] = None
    max_workers: int = 4
    heartbeat_interval: int = 30  # seconds
    task_timeout: int = 3600  # seconds
    enable_failover: bool = True
    data_sync_interval: int = 300  # seconds

class Config:
    """Main configuration class cho VideoRAG System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration từ file hoặc default values
        
        Args:
            config_path: Đường dẫn đến config file (yaml/json)
        """
        self.config_path = config_path
        self._load_config()
        self._setup_directories()
        
    def _load_config(self):
        """Load configuration từ file hoặc sử dụng default"""
        if self.config_path and os.path.exists(self.config_path):
            self._load_from_file()
        else:
            self._load_default_config()
            
    def _load_from_file(self):
        """Load config từ yaml/json file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self._parse_config_data(config_data)
            print(f"✅ Loaded config from {self.config_path}")
            
        except Exception as e:
            print(f"❌ Error loading config file: {e}")
            print("🔄 Falling back to default configuration")
            self._load_default_config()
    
    def _parse_config_data(self, config_data: Dict):
        """Parse config data và tạo các object"""
        # Models configuration
        self.models = {}
        for model_name, model_data in config_data.get('models', {}).items():
            self.models[model_name] = ModelConfig(
                name=model_name,
                **model_data
            )
        
        # Database configuration  
        self.databases = {}
        for db_name, db_data in config_data.get('databases', {}).items():
            self.databases[db_name] = DatabaseConfig(**db_data)
        
        # Processing configuration
        proc_data = config_data.get('processing', {})
        self.processing = ProcessingConfig(**proc_data)
        
        # Retrieval configuration
        ret_data = config_data.get('retrieval', {})
        self.retrieval = RetrievalConfig(**ret_data)
        
        # Distributed configuration
        dist_data = config_data.get('distributed', {})
        self.distributed = DistributedConfig(**dist_data)
        
        # System paths
        self.paths = config_data.get('paths', {})
        
    def _load_default_config(self):
        """Load default configuration"""
        
        # Default models configuration
        self.models = {
            'imagebind': ModelConfig(
                name='imagebind',
                type=ModelType.VISION_LANGUAGE,
                model_path='imagebind_huge',
                device='auto',
                batch_size=1,
                enabled=True
            ),
            'minicpm_v': ModelConfig(
                name='minicpm_v',
                type=ModelType.VISION_LANGUAGE,
                model_path='openbmb/MiniCPM-V-2_6-int4',
                device='auto',
                quantization='int4',
                batch_size=1,
                max_length=2048,
                enabled=True
            ),
            'gpt4o_mini': ModelConfig(
                name='gpt4o_mini',
                type=ModelType.LLM,
                model_path='gpt-4o-mini',
                max_length=32768,
                temperature=0.1,
                top_p=0.9,
                enabled=True
            ),
            'text_embedding': ModelConfig(
                name='text_embedding',
                type=ModelType.EMBEDDING,
                model_path='text-embedding-3-small',
                device='auto',
                enabled=True
            ),
            'whisper': ModelConfig(
                name='whisper',
                type=ModelType.AUDIO,
                model_path='distil-whisper/distil-large-v3',
                device='auto',
                batch_size=1,
                enabled=True
            )
        }
        
        # Default database configuration
        self.databases = {
            'vector_db': DatabaseConfig(
                type=DatabaseType.VECTOR,
                connection_string='./data/vector_db',
                collection_name='video_embeddings',
                embedding_dimension=1536,
                index_type='HNSW',
                metric='cosine'
            ),
            'graph_db': DatabaseConfig(
                type=DatabaseType.GRAPH,
                connection_string='./data/graph_db',
                collection_name='knowledge_graph'
            ),
            'metadata_db': DatabaseConfig(
                type=DatabaseType.METADATA,
                connection_string='./data/metadata.db',
                collection_name='video_metadata'
            )
        }
        
        # Default processing configuration
        self.processing = ProcessingConfig(
            frame_sampling_rate=5,
            detailed_sampling_rate=15,
            chunk_duration=30,
            max_frames_per_video=3600,
            supported_formats=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            output_resolution=(224, 224),
            audio_sample_rate=16000
        )
        
        # Default retrieval configuration
        self.retrieval = RetrievalConfig(
            similarity_threshold=0.7,
            top_k_text=10,
            top_k_visual=5,
            top_k_final=15,
            enable_cross_modal=True,
            enable_temporal_filtering=True,
            relevance_filter_threshold=0.6
        )
        
        # Default distributed configuration
        self.distributed = DistributedConfig(
            coordinator_host='localhost',
            coordinator_port=8888,
            worker_nodes=[],
            max_workers=4,
            heartbeat_interval=30,
            task_timeout=3600,
            enable_failover=True,
            data_sync_interval=300
        )
        
        # Default paths
        self.paths = {
            'data_dir': './data',
            'models_dir': './models',
            'temp_dir': './temp',
            'logs_dir': './logs',
            'output_dir': './output'
        }
    
    def _setup_directories(self):
        """Tạo các thư mục cần thiết"""
        for path_name, path_value in self.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Lấy cấu hình của một model cụ thể"""
        return self.models.get(model_name)
    
    def get_database_config(self, db_name: str) -> Optional[DatabaseConfig]:
        """Lấy cấu hình của một database cụ thể"""
        return self.databases.get(db_name)
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Kiểm tra xem model có được enable không"""
        model_config = self.get_model_config(model_name)
        return model_config.enabled if model_config else False
    
    def get_supported_video_formats(self) -> List[str]:
        """Lấy danh sách format video được hỗ trợ"""
        return self.processing.supported_formats
    
    def validate_config(self) -> bool:
        """Validate configuration"""
        try:
            # Kiểm tra essential models
            essential_models = ['gpt4o_mini', 'text_embedding']
            for model_name in essential_models:
                if not self.is_model_enabled(model_name):
                    print(f"⚠️ Warning: Essential model '{model_name}' is disabled")
            
            # Kiểm tra database connections
            for db_name, db_config in self.databases.items():
                if not db_config.connection_string:
                    print(f"❌ Error: Database '{db_name}' missing connection string")
                    return False
            
            # Kiểm tra directories
            for path_name, path_value in self.paths.items():
                if not os.path.exists(path_value):
                    print(f"⚠️ Warning: Directory '{path_value}' does not exist")
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        try:
            config_dict = {
                'models': {name: asdict(config) for name, config in self.models.items()},
                'databases': {name: asdict(config) for name, config in self.databases.items()},
                'processing': asdict(self.processing),
                'retrieval': asdict(self.retrieval),
                'distributed': asdict(self.distributed),
                'paths': self.paths
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.endswith('.yaml') or output_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def update_model_config(self, model_name: str, **kwargs):
        """Update configuration cho một model cụ thể"""
        if model_name in self.models:
            for key, value in kwargs.items():
                if hasattr(self.models[model_name], key):
                    setattr(self.models[model_name], key, value)
            print(f"✅ Updated config for model '{model_name}'")
        else:
            print(f"❌ Model '{model_name}' not found in configuration")
    
    def get_gpu_memory_limit(self) -> Optional[str]:
        """Lấy GPU memory limit từ environment hoặc config"""
        return os.environ.get('GPU_MEMORY_LIMIT', '24GB')
    
    def get_api_keys(self) -> Dict[str, str]:
        """Lấy API keys từ environment variables"""
        return {
            'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
            'huggingface_token': os.environ.get('HUGGINGFACE_TOKEN', '')
        }
    
    def print_config_summary(self):
        """In tóm tắt configuration"""
        print("\n" + "="*50)
        print("📋 VideoRAG System Configuration Summary")
        print("="*50)
        
        print(f"\n🤖 Models ({len(self.models)} configured):")
        for name, config in self.models.items():
            status = "✅" if config.enabled else "❌"
            print(f"  {status} {name}: {config.model_path}")
        
        print(f"\n💾 Databases ({len(self.databases)} configured):")
        for name, config in self.databases.items():
            print(f"  📁 {name}: {config.connection_string}")
        
        print(f"\n🎬 Processing Settings:")
        print(f"  📽️ Frame sampling: {self.processing.frame_sampling_rate}/clip")
        print(f"  ⏱️ Chunk duration: {self.processing.chunk_duration}s")
        print(f"  🎯 Max frames/video: {self.processing.max_frames_per_video}")
        
        print(f"\n🔍 Retrieval Settings:")
        print(f"  🎯 Similarity threshold: {self.retrieval.similarity_threshold}")
        print(f"  📊 Top-K (text/visual): {self.retrieval.top_k_text}/{self.retrieval.top_k_visual}")
        
        print(f"\n🌐 Distributed Settings:")
        print(f"  🖥️ Coordinator: {self.distributed.coordinator_host}:{self.distributed.coordinator_port}")
        print(f"  👥 Max workers: {self.distributed.max_workers}")
        
        print("="*50 + "\n")

# Global config instance
_global_config = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global config instance (singleton pattern)
    
    Args:
        config_path: Đường dẫn đến config file (chỉ sử dụng lần đầu)
        
    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config

def reset_config():
    """Reset global config instance"""
    global _global_config
    _global_config = None

def load_config_from_env() -> Config:
    """
    Load configuration từ environment variables
    Hữu ích cho containerized deployment
    """
    config = Config()
    
    # Override với environment variables nếu có
    if os.environ.get('OPENAI_API_KEY'):
        config.update_model_config('gpt4o_mini', 
                                 model_path=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'))
    
    if os.environ.get('HUGGINGFACE_TOKEN'):
        # Enable HuggingFace models nếu có token
        for model_name in ['imagebind', 'minicpm_v', 'whisper']:
            if model_name in config.models:
                config.models[model_name].enabled = True
    
    # Database overrides
    if os.environ.get('DATABASE_URL'):
        config.databases['vector_db'].connection_string = os.environ.get('DATABASE_URL')
    
    # Processing overrides
    if os.environ.get('MAX_FRAMES_PER_VIDEO'):
        config.processing.max_frames_per_video = int(os.environ.get('MAX_FRAMES_PER_VIDEO'))
    
    if os.environ.get('CHUNK_DURATION'):
        config.processing.chunk_duration = int(os.environ.get('CHUNK_DURATION'))
    
    # Distributed overrides
    if os.environ.get('COORDINATOR_HOST'):
        config.distributed.coordinator_host = os.environ.get('COORDINATOR_HOST')
    
    if os.environ.get('COORDINATOR_PORT'):
        config.distributed.coordinator_port = int(os.environ.get('COORDINATOR_PORT'))
    
    return config

def create_default_config_file(output_path: str = './config.yaml'):
    """
    Tạo file config mặc định để user có thể customize
    
    Args:
        output_path: Đường dẫn output file
    """
    try:
        config = Config()
        config.save_config(output_path)
        print(f"✅ Created default config file at: {output_path}")
        print("📝 You can now customize the configuration and reload it")
        
    except Exception as e:
        print(f"❌ Error creating default config file: {e}")

def validate_system_requirements(config: Config) -> bool:
    """
    Validate system requirements dựa trên configuration
    
    Args:
        config: Config instance
        
    Returns:
        True nếu system đáp ứng requirements
    """
    try:
        import torch
        import transformers
        
        # Check CUDA availability cho GPU models
        gpu_models = [name for name, cfg in config.models.items() 
                     if cfg.device in ['cuda', 'auto'] and cfg.enabled]
        
        if gpu_models and not torch.cuda.is_available():
            print("⚠️ Warning: GPU models configured but CUDA not available")
            print(f"GPU models: {gpu_models}")
            
            # Auto fallback to CPU
            for model_name in gpu_models:
                config.update_model_config(model_name, device='cpu')
                print(f"🔄 Switched {model_name} to CPU")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            required_memory = 8  # Minimum 8GB cho ImageBind + MiniCPM-V
            
            if gpu_memory < required_memory:
                print(f"⚠️ Warning: GPU memory ({gpu_memory:.1f}GB) < recommended ({required_memory}GB)")
        
        # Check API keys
        api_keys = config.get_api_keys()
        if config.is_model_enabled('gpt4o_mini') and not api_keys['openai_api_key']:
            print("⚠️ Warning: OpenAI API key not found. GPT-4o-mini will be disabled")
            config.update_model_config('gpt4o_mini', enabled=False)
        
        # Check HuggingFace token for gated models
        if not api_keys['huggingface_token']:
            print("⚠️ Warning: HuggingFace token not found. Some models may not work")
        
        print("✅ System requirements validation completed")
        return True
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("📦 Please install required dependencies")
        return False
    
    except Exception as e:
        print(f"❌ System requirements validation failed: {e}")
        return False

def get_optimal_device_config() -> Dict[str, str]:
    """
    Determine optimal device configuration dựa trên hardware available
    
    Returns:
        Dict mapping model types to optimal devices
    """
    device_config = {}
    
    try:
        import torch
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = sum(torch.cuda.get_device_properties(i).total_memory 
                             for i in range(num_gpus)) / (1024**3)
            
            print(f"🖥️ Detected {num_gpus} GPU(s) with {total_memory:.1f}GB total memory")
            
            if total_memory >= 24:  # High-end setup
                device_config = {
                    'imagebind': 'cuda:0',
                    'minicpm_v': 'cuda:0',
                    'whisper': 'cuda:0',
                    'text_embedding': 'cuda:0'
                }
            elif total_memory >= 12:  # Mid-range setup
                device_config = {
                    'imagebind': 'cuda:0',
                    'minicpm_v': 'cuda:0',
                    'whisper': 'cpu',  # Move audio to CPU
                    'text_embedding': 'cuda:0'
                }
            else:  # Low memory GPU
                device_config = {
                    'imagebind': 'cpu',
                    'minicpm_v': 'cuda:0',  # Keep most important model on GPU
                    'whisper': 'cpu',
                    'text_embedding': 'cpu'
                }
        else:
            print("🖥️ No GPU detected, using CPU for all models")
            device_config = {
                'imagebind': 'cpu',
                'minicpm_v': 'cpu',
                'whisper': 'cpu',
                'text_embedding': 'cpu'
            }
            
    except ImportError:
        print("⚠️ PyTorch not found, defaulting to CPU")
        device_config = {model: 'cpu' for model in ['imagebind', 'minicpm_v', 'whisper', 'text_embedding']}
    
    return device_config

def apply_performance_optimizations(config: Config):
    """
    Apply performance optimizations dựa trên hardware
    
    Args:
        config: Config instance to modify
    """
    device_config = get_optimal_device_config()
    
    # Apply optimal device assignments
    for model_name, device in device_config.items():
        if model_name in config.models:
            config.update_model_config(model_name, device=device)
    
    # Adjust batch sizes dựa trên available memory
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if gpu_memory >= 24:
                # High memory - increase batch sizes
                config.update_model_config('minicpm_v', batch_size=2)
                config.processing.frame_sampling_rate = 8
            elif gpu_memory < 8:
                # Low memory - reduce batch sizes
                config.update_model_config('minicpm_v', batch_size=1)
                config.processing.frame_sampling_rate = 3
                config.processing.max_frames_per_video = 1800
    
    except ImportError:
        pass
    
    print("🚀 Applied performance optimizations")

# Example usage và testing
if __name__ == "__main__":
    print("🧪 Testing VideoRAG Configuration System")
    
    # Test 1: Default configuration
    print("\n1️⃣ Testing default configuration...")
    config = Config()
    config.print_config_summary()
    
    # Test 2: Validation
    print("\n2️⃣ Testing configuration validation...")
    config.validate_config()
    
    # Test 3: System requirements
    print("\n3️⃣ Testing system requirements...")
    validate_system_requirements(config)
    
    # Test 4: Performance optimization
    print("\n4️⃣ Testing performance optimization...")
    apply_performance_optimizations(config)
    
    # Test 5: Save/load configuration
    print("\n5️⃣ Testing save/load configuration...")
    config.save_config('./test_config.yaml')
    
    # Test 6: Load from file
    print("\n6️⃣ Testing load from file...")
    config2 = Config('./test_config.yaml')
    
    print("\n✅ All configuration tests completed!")
    
    # Clean up
    import os
    if os.path.exists('./test_config.yaml'):
        os.remove('./test_config.yaml')